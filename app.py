from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import base64
import binascii
import io
import requests
from io import BytesIO
from datetime import datetime

# OpenCV is optional — face preprocessing skipped gracefully when absent.
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None

# PIL for sentence overlay
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from dotenv import load_dotenv
import cloudinary_helper
import replicate_helper

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

os.environ['PYTHONUNBUFFERED'] = '1'

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(32)

# In-memory store for active predictions
active_predictions = {}

FACE_TARGET_SIZE = int(os.getenv('FACE_TARGET_SIZE', '1024'))
FACE_CROP_SCALE  = float(os.getenv('FACE_CROP_SCALE', '2.4'))
_face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') if CV2_AVAILABLE else None
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))


# ── Utilities ──────────────────────────────────────────────

def decode_base64_image(image_input):
    """Decode a data URL or raw base64 string into bytes."""
    if not isinstance(image_input, str):
        raise ValueError('photo must be a base64 string')
    payload = image_input.strip()
    if not payload:
        raise ValueError('photo is empty')
    if payload.startswith('data:'):
        if ',' not in payload:
            raise ValueError('photo data URL is malformed')
        payload = payload.split(',', 1)[1].strip()
    if not payload:
        raise ValueError('photo base64 content is empty')
    payload += '=' * (-len(payload) % 4)
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError('Invalid photo base64 payload')


def preprocess_photo(image_bytes):
    """
    Crop and resize to center on the detected face.
    Falls back to original bytes when cv2 is unavailable.
    """
    if not CV2_AVAILABLE:
        print("[INFO] cv2 not available — skipping face preprocessing", flush=True)
        return image_bytes, False

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Unable to decode image bytes')

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        center_x, center_y = x + w // 2, y + h // 2
        crop_size = int(max(w, h) * FACE_CROP_SCALE)
    else:
        center_x, center_y = width // 2, height // 2
        crop_size = int(min(width, height))

    crop_size = max(256, min(crop_size, width, height))
    half = crop_size // 2
    left  = max(0, center_x - half)
    top   = max(0, center_y - half)
    right  = left + crop_size
    bottom = top  + crop_size

    if right  > width:  right  = width;  left = width - crop_size
    if bottom > height: bottom = height; top  = height - crop_size

    cropped = image[top:bottom, left:right]
    if cropped.size == 0:
        raise ValueError('Failed to crop photo for preprocessing')

    interp = cv2.INTER_CUBIC if crop_size < FACE_TARGET_SIZE else cv2.INTER_AREA
    final  = cv2.resize(cropped, (FACE_TARGET_SIZE, FACE_TARGET_SIZE), interpolation=interp)
    ok, encoded = cv2.imencode('.jpg', final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError('Failed to encode preprocessed photo')

    return encoded.tobytes(), len(faces) > 0


def overlay_sentence_on_image(image_bytes, sentence):
    """
    Overlay the character's caption sentence on the bottom of the result image.
    Returns original bytes if PIL is unavailable or overlay fails.
    """
    if not PIL_AVAILABLE or not sentence:
        return image_bytes

    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        width, height = image.size

        font_size = max(48, width // 10)
        font = None
        for fp in [
            # Linux / Vercel paths
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
            '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
            '/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf',
            '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/dejavu/DejaVuSans.ttf',
            # Windows paths
            'arial.ttf', 'Arial.ttf', 'arialbd.ttf',
            'C:/Windows/Fonts/arial.ttf',
            'C:/Windows/Fonts/arialbd.ttf',
        ]:
            try:
                font = ImageFont.truetype(fp, font_size)
                print(f"[OVERLAY] Using font: {fp} size {font_size}", flush=True)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                font = ImageFont.load_default()

        # Measure text
        dummy_draw = ImageDraw.Draw(image)
        bbox        = dummy_draw.textbbox((0, 0), sentence, font=font)
        text_w      = bbox[2] - bbox[0]
        text_h      = bbox[3] - bbox[1]
        pad_v       = int(height * 0.025)
        strip_h     = text_h + pad_v * 2

        # Draw semi-transparent strip at the bottom
        image_rgba = image.convert('RGBA')
        strip      = Image.new('RGBA', (width, strip_h), (0, 0, 0, 0))
        strip_draw = ImageDraw.Draw(strip)
        strip_draw.rectangle([0, 0, width, strip_h], fill=(8, 20, 40, 215))
        image_rgba.paste(strip, (0, height - strip_h), strip)
        image = image_rgba.convert('RGB')

        draw   = ImageDraw.Draw(image)
        text_x = (width - text_w) // 2
        text_y = height - strip_h + pad_v

        # Drop shadow
        draw.text((text_x + 2, text_y + 2), sentence, font=font, fill=(0, 0, 0))
        # White text
        draw.text((text_x,     text_y),     sentence, font=font, fill=(255, 255, 255))

        out = BytesIO()
        image.save(out, format='JPEG', quality=95)
        return out.getvalue()

    except Exception as e:
        print(f"[OVERLAY] Text overlay failed: {e}", flush=True)
        return image_bytes


# ── Middleware ─────────────────────────────────────────────

@app.before_request
def log_request_info():
    try:
        if request.path != '/health':
            print(f"\n[REQUEST] {request.method} {request.path}", flush=True)
            if request.content_length:
                print(f"[REQUEST] Content-Length: {request.content_length} bytes", flush=True)
    except Exception as e:
        print(f"[ERROR] before_request: {e}", flush=True)


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(f"\n[CRITICAL ERROR] {str(e)}", flush=True)
    print(traceback.format_exc(), flush=True)
    return jsonify({'error': f'Server Error: {str(e)}'}), 500


# ── Routes ─────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/swap-face', methods=['POST'])
def swap_face():
    print("=" * 60, flush=True)
    print("PHOTO GENERATION REQUEST", flush=True)
    print("=" * 60, flush=True)

    try:
        data = request.get_json(silent=True)
        if not data or 'child_photo' not in data or 'character' not in data:
            return jsonify({'error': 'Missing required fields: child_photo and character'}), 400

        try:
            image_bytes = decode_base64_image(data['child_photo'])
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        character       = data['character']
        requested_model = (os.getenv('FACE_SWAP_MODEL') or 'google/nano-banana').strip().lower()

        print(f"[INFO] Character: {character}", flush=True)
        print(f"[INFO] Image size: {len(image_bytes)} bytes", flush=True)

        # Preprocess (face crop) only for template-based models
        if requested_model in replicate_helper.GOOGLE_DIRECT_MODELS:
            processed_bytes = image_bytes
            print(f"[INFO] Using original image for Google model (no face crop)", flush=True)
        else:
            processed_bytes, used_crop = preprocess_photo(image_bytes)
            print(f"[INFO] Preprocessed size: {len(processed_bytes)} bytes (face_crop={'yes' if used_crop else 'no'})", flush=True)

        # Step 1: Upload to Cloudinary
        print("[STEP 1] Uploading photo to Cloudinary...", flush=True)
        upload_result = cloudinary_helper.upload_temp_image(processed_bytes)
        if not upload_result:
            return jsonify({'error': 'Failed to upload image to cloud storage'}), 500

        image_url = upload_result['url']
        public_id = upload_result['public_id']
        print(f"[SUCCESS] Uploaded: {image_url[:50]}...", flush=True)

        # Step 2: Start AI generation
        print("[STEP 2] Starting AI generation...", flush=True)
        try:
            prediction_info = replicate_helper.start_face_generation(
                child_image_url=image_url,
                character=character,
                raise_errors=True
            )
        except replicate_helper.FaceGenerationStartError as ai_error:
            cloudinary_helper.delete_temp_image(public_id)
            return jsonify({'error': f'Failed to start AI processing: {str(ai_error)}'}), 502

        if not prediction_info:
            cloudinary_helper.delete_temp_image(public_id)
            return jsonify({'error': 'Failed to start AI processing'}), 502

        prediction_id = prediction_info['prediction_id']
        active_predictions[prediction_id] = {
            'child_cloudinary_id': public_id,
            'character':           character,
            'status':              'processing',
            'model':               prediction_info.get('model'),
        }

        print(f"[SUCCESS] Prediction started: {prediction_id}", flush=True)
        print("=" * 60, flush=True)

        return jsonify({
            'prediction_id': prediction_id,
            'status':        'processing',
            'model':         prediction_info.get('model'),
            'message':       'Processing started. Poll /check-status to get updates.',
        })

    except Exception as e:
        import traceback
        print(f"[ERROR] swap_face: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/check-status/<prediction_id>', methods=['GET'])
def check_status(prediction_id):
    try:
        print(f"[POLL] Checking status for: {prediction_id}", flush=True)

        if prediction_id not in active_predictions:
            return jsonify({'error': 'Prediction not found'}), 404

        status_info = replicate_helper.check_prediction_status(prediction_id)
        if not status_info:
            return jsonify({'error': 'Failed to check prediction status'}), 500

        status          = status_info['status']
        prediction_data = active_predictions[prediction_id]
        prediction_data['status'] = status

        if status == 'succeeded':
            print(f"[SUCCESS] Prediction completed: {prediction_id}", flush=True)
            result_url = status_info.get('result_url')

            if not result_url:
                return jsonify({'error': 'No result URL received from AI model'}), 500

            # Overlay caption sentence on the result image
            character = prediction_data.get('character')
            sentence  = replicate_helper.get_character_sentence(character)

            if sentence:
                try:
                    print(f"[OVERLAY] Adding sentence: {sentence}", flush=True)
                    resp = requests.get(result_url, timeout=30)
                    resp.raise_for_status()
                    overlaid_bytes = overlay_sentence_on_image(resp.content, sentence)
                    final_upload   = cloudinary_helper.upload_temp_image(overlaid_bytes)
                    if final_upload:
                        result_url = final_upload['url']
                        print(f"[OVERLAY] Done. New URL: {result_url[:50]}...", flush=True)
                except Exception as overlay_err:
                    print(f"[OVERLAY] Failed (using original): {overlay_err}", flush=True)

            # Cleanup source image from Cloudinary
            child_id = prediction_data.get('child_cloudinary_id')
            if child_id:
                print(f"[CLEANUP] Deleting source image...", flush=True)
                cloudinary_helper.delete_temp_image(child_id)

            del active_predictions[prediction_id]

            return jsonify({
                'status':     'succeeded',
                'result_url': result_url,
                'model':      prediction_data.get('model'),
            })

        elif status == 'failed':
            error_msg = status_info.get('error', 'Unknown error')
            child_id  = prediction_data.get('child_cloudinary_id')
            if child_id:
                cloudinary_helper.delete_temp_image(child_id)
            del active_predictions[prediction_id]

            return jsonify({'status': 'failed', 'error': error_msg})

        else:
            return jsonify({'status': status, 'model': prediction_data.get('model')})

    except Exception as e:
        import traceback
        print(f"[ERROR] check_status: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({'error': 'Missing image_url parameter'}), 400

        image_url = data['image_url']
        print(f"[QR] Generating QR for: {image_url[:50]}...", flush=True)

        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(image_url)
        qr.make(fit=True)

        img    = qr.make_image(fill_color='black', back_color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        qr_b64 = base64.b64encode(buffer.read()).decode('utf-8')

        print("[QR] Generated successfully", flush=True)
        return jsonify({'qr_code': f'data:image/png;base64,{qr_b64}'})

    except Exception as e:
        print(f"[ERROR] QR generation: {str(e)}", flush=True)
        return jsonify({'error': f'QR generation failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':          'healthy',
        'cloudinary':      'configured' if os.getenv('CLOUDINARY_API_KEY') else 'not configured',
        'gemini':          'configured' if (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')) else 'not configured',
        'face_swap_model': (os.getenv('FACE_SWAP_MODEL') or 'google/nano-banana').strip().lower(),
    })


# ── Entry Point ────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    model = (os.getenv('FACE_SWAP_MODEL') or 'google/nano-banana').strip()
    print(f"OB-GYN ACTIVITY PHOTO BOOTH — {model}")
    print("=" * 60)
    print("Server: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
