"""
Replicate API Helper Module - IP-Adapter Face Inpaint
Handles face blending using lucataco/ip_adapter-face-inpaint model.
"""

import replicate
import os
import time
import json
import io
import uuid
import base64
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import requests
from PIL import Image
import cloudinary_helper

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

# Load environment variables
load_dotenv()

# Configure Replicate API
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
if REPLICATE_API_TOKEN:
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN


GOOGLE_DIRECT_MODELS = {
    'google/nano-banana': 'gemini-2.5-flash-image',
    'google/nano-banana-pro': 'gemini-3-pro-image-preview',
    'google/nano-banana-2': 'gemini-3.1-flash-image-preview',
}

SUPPORTED_FACE_SWAP_MODELS = (
    'yan-ops/face_swap',
    'codeplugtech/face-swap',
    *GOOGLE_DIRECT_MODELS.keys(),
)

GOOGLE_DIRECT_RESULTS: Dict[str, Dict[str, Any]] = {}
LAST_START_ERROR: Optional[str] = None


class FaceGenerationStartError(RuntimeError):
    """Raised when the provider cannot start or complete initial image generation."""


def _is_google_direct_model(model_name: str) -> bool:
    return model_name in GOOGLE_DIRECT_MODELS


def _get_google_api_key() -> Optional[str]:
    return (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or '').strip() or None


def _redact_sensitive_values(message: str) -> str:
    redacted = message or ''
    for key in (
        'GEMINI_API_KEY',
        'GOOGLE_API_KEY',
        'REPLICATE_API_TOKEN',
        'CLOUDINARY_API_KEY',
        'CLOUDINARY_API_SECRET',
    ):
        value = (os.getenv(key) or '').strip()
        if value and len(value) >= 8:
            redacted = redacted.replace(value, '[redacted]')
    return redacted


def _safe_error_message(error: Any, max_length: int = 700) -> str:
    message = _redact_sensitive_values(str(error).strip() or error.__class__.__name__)
    return message[:max_length]


def get_last_start_error() -> Optional[str]:
    return LAST_START_ERROR


def _set_last_start_error(error: Any) -> None:
    global LAST_START_ERROR
    LAST_START_ERROR = _safe_error_message(error)


def _clear_last_start_error() -> None:
    global LAST_START_ERROR
    LAST_START_ERROR = None


def _google_direct_max_attempts() -> int:
    raw = os.getenv('GOOGLE_DIRECT_MAX_ATTEMPTS', '1')
    try:
        attempts = int(raw)
    except ValueError:
        attempts = 1
    return max(1, min(5, attempts))


def _is_retryable_google_error(error: Exception) -> bool:
    message = str(error).lower()
    retryable_markers = (
        '500',
        '502',
        '503',
        '504',
        'deadline',
        'timeout',
        'temporarily',
        'unavailable',
        'internal',
    )
    return any(marker in message for marker in retryable_markers)

def _clamp_weight(value: float) -> float:
    return max(0.5, min(1.0, value))


def _parse_weight_overrides(raw: str) -> Dict[str, float]:
    """
    Parse FACE_SWAP_WEIGHT_OVERRIDES in the format:
    "doctor_boy=0.82,teacher_girl=0.84,default=0.85"
    """
    overrides: Dict[str, float] = {}
    if not raw:
        return overrides

    for chunk in raw.split(','):
        item = chunk.strip()
        if not item or '=' not in item:
            continue
        key, value = item.split('=', 1)
        key = key.strip().lower()
        if not key:
            continue
        try:
            parsed = float(value.strip())
        except ValueError:
            continue
        overrides[key] = _clamp_weight(parsed)
    return overrides


def _get_swap_weight(character: Optional[str] = None, style_config: Optional[Dict[str, Any]] = None) -> float:
    """Get swap weight with override priority: style_config -> env override -> env default."""
    raw_weight = os.getenv('FACE_SWAP_WEIGHT', '0.85')
    try:
        parsed = float(raw_weight)
    except ValueError:
        parsed = 0.85
    selected = _clamp_weight(parsed)

    raw_overrides = os.getenv('FACE_SWAP_WEIGHT_OVERRIDES', '')
    overrides = _parse_weight_overrides(raw_overrides)
    character_key = (character or '').strip().lower()
    if character_key and character_key in overrides:
        selected = overrides[character_key]
    elif 'default' in overrides:
        selected = overrides['default']

    if style_config and isinstance(style_config.get('swap_weight'), (int, float)):
        selected = _clamp_weight(float(style_config['swap_weight']))

    return selected


def _get_face_swap_model() -> str:
    raw = (os.getenv('FACE_SWAP_MODEL') or 'google/nano-banana').strip().lower()
    if raw in SUPPORTED_FACE_SWAP_MODELS:
        return raw
    return 'google/nano-banana'


def _get_fallback_model(primary_model: str) -> Optional[str]:
    raw = (os.getenv('FACE_SWAP_FALLBACK_MODEL') or '').strip().lower()
    if raw in SUPPORTED_FACE_SWAP_MODELS and raw != primary_model:
        return raw
    return None


def _build_nano_banana_prompt(character: str, style_config: Optional[Dict[str, Any]] = None) -> str:
    # If the character style has a full_prompt, use it as-is (bypasses shared constraints).
    if style_config and isinstance(style_config.get('full_prompt'), str) and style_config['full_prompt'].strip():
        return style_config['full_prompt'].strip()

    shared_constraints = (
        "Identity lock: keep the exact same face from the input photo "
        "(same eyes, nose, mouth, jawline, skin tone, and natural age appearance). "
        "Do not change facial structure or age. "
        "Photorealistic, high detail, natural skin texture, "
        "clean lighting, sharp focus. No text, no watermark, no logo, no cartoon, no anime, "
        "no extra people, no face distortion."
    )
    job_prompt = style_config.get('prompt') if style_config else None

    # Prefer external prompt pack if available.
    try:
        prompt_file = os.path.join(os.path.dirname(__file__), 'nano_banana_prompts.json')
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            prompts = config.get('prompts') or {}
            loaded_shared = config.get('shared_constraints')
            if isinstance(loaded_shared, str) and loaded_shared.strip():
                shared_constraints = loaded_shared.strip()
            key = (character or '').strip().lower()
            # Only exact key match — avoid false fallbacks for new doctor keys.
            loaded_prompt = prompts.get(key)
            if isinstance(loaded_prompt, str) and loaded_prompt.strip():
                job_prompt = loaded_prompt.strip()
    except Exception:
        pass

    if not job_prompt:
        job_prompt = (
            f"Edit this photo into a realistic {character.replace('_', ' ')} portrait. "
            "Dress the person in role-appropriate professional clothing and set a matching real-world background."
        )

    return f"{job_prompt} {shared_constraints}"


def _build_input_candidates(
    model_name: str,
    source_image: str,
    target_image: Optional[str],
    weight: float,
    character: str,
    style_config: Optional[Dict[str, Any]] = None
) -> list:
    if model_name == 'yan-ops/face_swap':
        return [{
            "source_image": source_image,
            "target_image": target_image,
            "weight": weight
        }]

    if model_name == 'codeplugtech/face-swap':
        # Try common schema variants defensively to avoid breaking on model-side changes.
        return [
            {"input_image": target_image, "swap_image": source_image},
            {"target_image": target_image, "source_image": source_image},
            {"image": target_image, "swap_image": source_image},
            {"input": target_image, "swap": source_image},
        ]

    if _is_google_direct_model(model_name):
        prompt = _build_nano_banana_prompt(character=character, style_config=style_config)
        return [
            {"prompt": prompt, "image_input": [source_image], "output_format": "jpg"},
            {"prompt": prompt, "image_input": source_image, "output_format": "jpg"},
            {"prompt": prompt, "images": [source_image], "output_format": "jpg"},
            {"prompt": prompt, "image": source_image, "output_format": "jpg"},
        ]

    return [{
        "source_image": source_image,
        "target_image": target_image,
        "weight": weight
    }]


def _create_prediction_with_candidates(model_name: str, input_candidates: list) -> Optional[Any]:
    model = replicate.models.get(model_name)
    version = model.latest_version
    print(f"[Replicate] Using model: {model_name} (version: {version.id[:12]}...)", flush=True)

    last_error = None
    for idx, input_params in enumerate(input_candidates, start=1):
        try:
            print(f"[Replicate] Trying input schema #{idx}: {list(input_params.keys())}", flush=True)
            return replicate.predictions.create(
                version=version.id,
                input=input_params
            )
        except Exception as e:
            last_error = e
            print(f"[Replicate] Schema #{idx} failed: {str(e)}", flush=True)

    if last_error:
        raise last_error
    return None


def _download_image_for_google(image_url: str) -> Image.Image:
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    return image.convert('RGB')


def _iter_google_response_parts(response: Any) -> List[Any]:
    parts = getattr(response, 'parts', None)
    if parts:
        return list(parts)

    candidates = getattr(response, 'candidates', None) or []
    collected: List[Any] = []
    for candidate in candidates:
        content = getattr(candidate, 'content', None)
        candidate_parts = getattr(content, 'parts', None) if content else None
        if candidate_parts:
            collected.extend(candidate_parts)
    return collected


def _extract_google_response_text(parts: List[Any]) -> str:
    text_chunks = []
    for part in parts:
        text = getattr(part, 'text', None)
        if text:
            text_chunks.append(str(text))
    return ' '.join(text_chunks).strip()


def _extract_google_image_bytes(response: Any) -> bytes:
    parts = _iter_google_response_parts(response)
    for part in parts:
        inline_data = getattr(part, 'inline_data', None) or getattr(part, 'inlineData', None)
        if not inline_data:
            continue

        data = getattr(inline_data, 'data', None)
        if data:
            if isinstance(data, str):
                return base64.b64decode(data)
            return bytes(data)

        as_image = getattr(part, 'as_image', None)
        if callable(as_image):
            image = as_image()
            image_bytes = getattr(image, 'image_bytes', None)
            if image_bytes:
                return bytes(image_bytes)

            output = io.BytesIO()
            try:
                image.save(output, format='PNG')
            except TypeError:
                image.save(output)
            return output.getvalue()

    response_text = _extract_google_response_text(parts)
    if response_text:
        raise RuntimeError(f"Google returned text but no image: {response_text[:500]}")
    raise RuntimeError("Google returned no image output")


def _remember_google_direct_result(result_url: str, result_public_id: Optional[str], model_name: str) -> str:
    now = time.time()
    for old_id, old_result in list(GOOGLE_DIRECT_RESULTS.items()):
        if now - float(old_result.get('created_at', now)) > 3600:
            GOOGLE_DIRECT_RESULTS.pop(old_id, None)

    prediction_id = f"google-direct-{uuid.uuid4().hex}"
    GOOGLE_DIRECT_RESULTS[prediction_id] = {
        'prediction_id': prediction_id,
        'status': 'succeeded',
        'result_url': result_url,
        'result_public_id': result_public_id,
        'created_at': now,
        'model': model_name,
        'provider': 'google'
    }
    return prediction_id


def _create_google_direct_generation(
    model_name: str,
    source_image: str,
    character: str,
    style_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if genai is None or genai_types is None:
        raise RuntimeError("google-genai package is required for Google direct image generation")

    api_key = _get_google_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Google direct image generation")

    google_model = GOOGLE_DIRECT_MODELS[model_name]
    prompt = _build_nano_banana_prompt(character=character, style_config=style_config)

    print(f"[Google] Using direct model: {google_model} for {model_name}", flush=True)
    print(f"[Google] Downloading source image: {source_image[:50]}...", flush=True)

    source = _download_image_for_google(source_image)
    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
    max_attempts = _google_direct_max_attempts()
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[Google] Generation attempt {attempt}/{max_attempts}", flush=True)
            response = client.models.generate_content(
                model=google_model,
                contents=[prompt, source],
                config=config
            )
            image_bytes = _extract_google_image_bytes(response)
            break
        except Exception as e:
            last_error = e
            print(f"[Google] Generation attempt {attempt} failed: {_safe_error_message(e)}", flush=True)
            if attempt >= max_attempts or not _is_retryable_google_error(e):
                raise
            time.sleep(min(2 ** (attempt - 1), 4))
    else:
        raise RuntimeError(f"Google image generation failed: {_safe_error_message(last_error)}")

    print(f"[Google] Generated image size: {len(image_bytes)} bytes", flush=True)
    upload_result = cloudinary_helper.upload_temp_image(image_bytes)
    if upload_result:
        result_url = upload_result['url']
        result_public_id = upload_result.get('public_id')
    else:
        print("[Google] Cloudinary result upload failed; returning inline data URL fallback", flush=True)
        result_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        result_public_id = None

    prediction_id = _remember_google_direct_result(
        result_url=result_url,
        result_public_id=result_public_id,
        model_name=model_name
    )
    print(f"[Google] Direct generation completed: {prediction_id}", flush=True)

    return {
        'prediction_id': prediction_id,
        'status': 'succeeded',
        'created_at': time.time(),
        'model': model_name,
        'provider': 'google',
        'google_model': google_model
    }


# Character style mapping for SDXL IP-Adapter FaceID
CHARACTER_STYLES = {
    'superman': {
        'prompt': '''Preserve the superhero's original head shape, jawline, skull structure,
hair, hairstyle, costume, pose, and lighting exactly as in the base image.

Subtly blend the child's facial characteristics into the face,
including eyes, eyebrows, nose, mouth, and expression.

Child face, young facial proportions, soft facial features.

No face swap. No replacement of head shape. Maintain superhero identity.

Photorealistic. Cinematic lighting. Clean studio background. High detail.''',
        
        'negative_prompt': '''face swap, different jawline, different hair, adult face, aging,
distorted face, cartoon, anime, exaggerated features, deformed,
brown costume, gray costume, desaturated colors, muted colors''',
        
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1768395277/templates/superman_template_vibrant_v4.png'
    },
    'batman': {
        'prompt': '''Preserve the superhero's original head shape, jawline, skull structure,
hair, hairstyle, costume, pose, and lighting exactly as in the base image.

Subtly blend the child's facial characteristics into the face,
including eyes, eyebrows, nose, mouth, and expression.

Child face, young facial proportions, soft facial features.

No face swap. No replacement of head shape. Maintain superhero identity.

Photorealistic. Cinematic lighting. Clean studio background. High detail.''',
        
        'negative_prompt': '''face swap, different jawline, different hair, adult face, aging,
distorted face, cartoon, anime, exaggerated features, deformed''',
        
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1768659434/templates/batman_template_backend.jpg'
    },
    'spiderman': {
        'prompt': '''Preserve the superhero's original head shape, jawline, skull structure,
hair, hairstyle, costume, pose, and lighting exactly as in the base image.

Subtly blend the child's facial characteristics into the face,
including eyes, eyebrows, nose, mouth, and expression.

Child face, young facial proportions, soft facial features.

No face swap. No replacement of head shape. Maintain superhero identity.

Photorealistic. Cinematic lighting. Clean studio background. High detail.''',
        
        'negative_prompt': '''face swap, different jawline, different hair, adult face, aging,
distorted face, cartoon, anime, exaggerated features, deformed''',
        
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1768659435/templates/spiderman_template_backend.jpg'
    },
    'wonderwoman': {
        'prompt': '''Preserve the superhero's original head shape, jawline, skull structure,
hair, hairstyle, costume, pose, and lighting exactly as in the base image.

Subtly blend the child's facial characteristics into the face,
including eyes, eyebrows, nose, mouth, and expression.

Child face, young facial proportions, soft facial features.

No face swap. No replacement of head shape. Maintain superhero identity.

Photorealistic. Cinematic lighting. Clean studio background. High detail.''',
        
        'negative_prompt': '''face swap, different jawline, different hair, adult face, aging,
distorted face, cartoon, anime, exaggerated features, deformed''',
        
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1768659435/templates/wonderwoman_template_backend.jpg'
    },
    'ironman': {
        'prompt': '''Preserve the superhero's original head shape, jawline, skull structure,
hair, hairstyle, costume, pose, and lighting exactly as in the base image.

Subtly blend the child's facial characteristics into the face,
including eyes, eyebrows, nose, mouth, and expression.

Child face, young facial proportions, soft facial features.

No face swap. No replacement of head shape. Maintain superhero identity.

Photorealistic. Cinematic lighting. Clean studio background. High detail.''',
        
        'negative_prompt': '''face swap, different jawline, different hair, adult face, aging,
distorted face, cartoon, anime, exaggerated features, deformed''',
        
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1768659436/templates/ironman_template_backend.jpg'
    },
    'captainamerica': {
        'prompt': '''Preserve the superhero's original head shape, jawline, skull structure,
hair, hairstyle, costume, pose, and lighting exactly as in the base image.

Subtly blend the child's facial characteristics into the face,
including eyes, eyebrows, nose, mouth, and expression.

Child face, young facial proportions, soft facial features.

No face swap. No replacement of head shape. Maintain superhero identity.

Photorealistic. Cinematic lighting. Clean studio background. High detail.''',
        
        'negative_prompt': '''face swap, different jawline, different hair, adult face, aging,
distorted face, cartoon, anime, exaggerated features, deformed''',
        
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1768659437/templates/captainamerica_template_backend.jpg'
    },
    'saudi_central_male': {
        'prompt': 'Saudi man wearing traditional bisht and thobe, photorealistic, cinematic lighting',
        'negative_prompt': 'cartoon, drawing, anime, low quality',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/templates/saudi_central_male_v7.png'
    },
    'saudi_traditional_daglah': {
        'prompt': 'Saudi man wearing traditional daglah with golden embroidered patterns and black bandolier, white shemagh with black agal, photorealistic, cinematic lighting, traditional Saudi heritage setting',
        'negative_prompt': 'cartoon, drawing, anime, low quality, modern clothing, western clothing',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1770104351/templates/saudi_traditional_daglah.jpg'
    },
    'jeddah_character_updated_1770487272655': {
        'prompt': 'Young Saudi man wearing white bisht over black thobe, white shemagh with gold-striped agal, clean-shaven face with very light mustache, Jeddah cityscape background, photorealistic, professional photography, natural lighting',
        'negative_prompt': 'cartoon, drawing, anime, low quality, modern clothing, western clothing, old man, elderly, goatee, beard, heavy mustache',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1770487272/templates/jeddah_character_updated.jpg'
    },
    'daglah_child_character_1770488439465': {
        'prompt': 'Saudi Arabian boy child aged 8-12 years old wearing traditional daglah with golden embroidery and black bandolier, white shemagh with black agal, child face, young boy, photorealistic, professional photography, natural lighting',
        'negative_prompt': 'cartoon, drawing, anime, low quality, modern clothing, western clothing, adult, teenager, facial hair, beard, mustache',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1770488439/templates/daglah_child_character.jpg'
    },
    'sharqawi_dress_character_1770578762613': {
        'prompt': 'Saudi Arabian woman wearing traditional Eastern Province (Sharqiyah) black dress with intricate gold embroidery, black hijab with gold trim, elegant appearance, photorealistic, professional photography, natural lighting, traditional Saudi heritage setting',
        'negative_prompt': 'cartoon, drawing, anime, low quality, modern clothing, western clothing, niqab',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1770578762/templates/sharqawi_dress_character.jpg'
    },
    'jeddah_character_updated_1770660835227': {
        'prompt': 'Fit athletic Saudi Arabian man wearing traditional white bisht over black thobe, white shemagh with gold-striped agal, well-groomed full light beard with mustache, natural neutral expression, historical Saudi heritage architecture background (old Jeddah Al-Balad style), photorealistic, professional photography, natural lighting',
        'negative_prompt': 'cartoon, drawing, anime, low quality, modern clothing, western clothing, smile, goatee only, clean shaven',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/templates/jeddah_updated_v2.jpg'
    },
    'northern_woman_v1_1770658383334': {
        'prompt': 'Saudi Arabian woman wearing traditional Northern Saudi dress with burgundy/maroon embroidered vest featuring vertical striped patterns and gold coin necklace decorations, black hijab with burgundy and gold coin headband, black waist sash, elegant appearance, photorealistic, professional photography, natural lighting, traditional Saudi heritage setting',
        'negative_prompt': 'cartoon, drawing, anime, low quality, modern clothing, western clothing, niqab',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/templates/northern_woman_v1.jpg'
    },
    'sharqawi_girl_child': {
        'prompt': 'Beautiful 8-12 year old Saudi Arabian girl wearing a traditional black Sharqawi dress with intricate gold embroidery and a matching sheer veil, gentle closed-mouth smile, photorealistic, cinematic lighting, traditional Saudi heritage architecture background',
        'negative_prompt': 'cartoon, drawing, anime, low quality, modern clothing, western clothing, teeth, open mouth',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1771073577/templates/sharqawi_girl_child.jpg'
    },
    'southern_asiri_adult': {
        'prompt': 'Beautiful 25-32 year old Saudi Arabian woman with elegant features and fit body shape, wearing traditional Southern Saudi (Asiri) black dress with vibrant colorful geometric embroidery on the chest and sleeves, yellow headscarf, large ornate golden coin bib necklace, gentle closed-mouth smile, photorealistic, cinematic lighting, historical Saudi courtyard background',
        'negative_prompt': 'cartoon, drawing, anime, low quality, modern clothing, western clothing, teeth, open mouth, overweight, bulky',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1771073578/templates/southern_asiri_adult.jpg'
    },
    'astronaut_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a astronaut, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625719/templates/dream_jobs/astronaut_boy.png'
    },
    'astronaut_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a astronaut, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625721/templates/dream_jobs/astronaut_girl.png'
    },
    'doctor_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a doctor, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625722/templates/dream_jobs/doctor_boy.png'
    },
    'doctor_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a doctor, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625723/templates/dream_jobs/doctor_girl.png'
    },
    'engineer_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a engineer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625724/templates/dream_jobs/engineer_boy.png'
    },
    'engineer_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a engineer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625725/templates/dream_jobs/engineer_girl.png'
    },
    'firefighter_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a firefighter, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625726/templates/dream_jobs/firefighter_boy.png'
    },
    'firefighter_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a firefighter, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625727/templates/dream_jobs/firefighter_girl.png'
    },
    'lawyer_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a lawyer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625728/templates/dream_jobs/lawyer_boy.png'
    },
    'lawyer_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a lawyer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'nurse_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a nurse, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625729/templates/dream_jobs/nurse_boy.png'
    },
    'nurse_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a nurse, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625730/templates/dream_jobs/nurse_girl.png'
    },
    'police_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a police, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625731/templates/dream_jobs/police_boy.png'
    },
    'police_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a police, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625732/templates/dream_jobs/police_girl.png'
    },
    'software_engineer': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a software engineer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625732/templates/dream_jobs/software_engineer.png'
    },
    'software_engineer_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a software engineer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'teacher_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a teacher, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625734/templates/dream_jobs/teacher_boy.png'
    },
    'teacher_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a teacher, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
        'template_image': 'https://res.cloudinary.com/dfcqp8igu/image/upload/v1773625735/templates/dream_jobs/teacher_girl.png'
    },
    # ── New Dream Jobs ──────────────────────────────────────────────────────────
    'architect_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as an architect, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'architect_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as an architect, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'businessman_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a businessman, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'businesswoman_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a businesswoman, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'cook_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a cook, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'cook_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a cook, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'driver_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a driver, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'driver_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a driver, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'fashion_designer_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a fashion designer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'fashion_designer_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a fashion designer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'farmer_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a farmer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'farmer_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a farmer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'flight_attendant_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a flight attendant, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'flight_attendant_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a flight attendant, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'journalist_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a journalist, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'journalist_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a journalist, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'manager_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a manager, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'manager_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a manager, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'mechanic_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a mechanic, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'mechanic_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a mechanic, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'photographer_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a photographer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'photographer_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a photographer, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'pilot_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a pilot, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'pilot_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a pilot, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'waiter_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), working as a waiter, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'waitress_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), working as a waitress, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'football_player_boy': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (boy), playing as a professional football player, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    'football_player_girl': {
        'prompt': 'A photorealistic portrait of a young Saudi Arabian child (girl), playing as a professional football player, cinematic lighting.',
        'negative_prompt': 'cartoon, drawing, anime, low quality, adult face, aging, distorted, ugly',
    },
    # ── OB-GYN Doctor Characters ────────────────────────────────────────────────
    'before_residency_male': {
        'sentence': 'Smiling before discovering labor ward reality.',
        'full_prompt': (
            "Transform this male doctor's photo into a portrait of a bright-eyed, fresh male medical graduate "
            "on his very first day. He wears a spotless white coat and proudly holds a stethoscope, "
            "wearing a wide optimistic smile full of hope. Background: a clean, modern hospital corridor "
            "with soft natural daylight. Magazine cover style, cinematic lighting, photorealistic, "
            "high-detail professional medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'before_residency_female': {
        'sentence': 'Smiling before discovering labor ward reality.',
        'full_prompt': (
            "Transform this female doctor's photo into a portrait of a bright-eyed, fresh female medical graduate "
            "on her very first day. She wears a spotless white coat (with optional hijab if appropriate), "
            "proudly holds a stethoscope, wearing a wide optimistic smile full of hope. "
            "Background: a clean, modern hospital corridor with soft natural daylight. "
            "Magazine cover style, cinematic lighting, photorealistic, "
            "high-detail professional medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'after_24h_call_male': {
        'sentence': 'Running on 2% battery.',
        'full_prompt': (
            "Transform this male doctor's photo to show extreme exhaustion after a brutal 24-hour hospital shift. "
            "Deep dark circles under heavy tired eyes, slightly disheveled scrubs, holding an empty "
            "crushed coffee cup. Background: a hospital break room under harsh fluorescent lighting at night. "
            "Realistically exhausted yet still standing. Cinematic moody lighting, photorealistic, "
            "high-detail professional medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'after_24h_call_female': {
        'sentence': 'Running on 2% battery.',
        'full_prompt': (
            "Transform this female doctor's photo to show extreme exhaustion after a brutal 24-hour hospital shift. "
            "Deep dark circles under heavy tired eyes, slightly disheveled scrubs and hijab (if appropriate), "
            "holding an empty crushed coffee cup. Background: a hospital break room under harsh fluorescent "
            "lighting at night. Realistically exhausted yet still standing. Cinematic moody lighting, "
            "photorealistic, high-detail professional medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'emergency_csection_male': {
        'sentence': 'Activated survival mode.',
        'full_prompt': (
            "Transform this male doctor's photo into an OB-GYN surgeon in the middle of an emergency C-section. "
            "Full surgical scrubs, surgical cap, mask pulled down to chin revealing intense laser-focused eyes "
            "and an expression of total concentration and calm under pressure. "
            "Background: bright surgical operating room lights, sterile OR environment. "
            "High-stakes adrenaline atmosphere, dramatic cinematic lighting, photorealistic, "
            "high-detail professional medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'emergency_csection_female': {
        'sentence': 'Activated survival mode.',
        'full_prompt': (
            "Transform this female doctor's photo into an OB-GYN surgeon in the middle of an emergency C-section. "
            "Full surgical scrubs, surgical cap covering hair (and hijab if appropriate), mask pulled down to chin "
            "revealing intense laser-focused eyes and an expression of total concentration and calm under pressure. "
            "Background: bright surgical operating room lights, sterile OR environment. "
            "High-stakes adrenaline atmosphere, dramatic cinematic lighting, photorealistic, "
            "high-detail professional medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'after_coffee_male': {
        'sentence': 'Vital signs restored.',
        'full_prompt': (
            "Transform this male doctor's photo to show a visibly refreshed and revitalized doctor "
            "right after drinking his first coffee of the day. Warm satisfied smile, holding a steaming "
            "coffee cup with both hands, clearly renewed energy in his posture and expression. "
            "Background: cozy hospital break room with warm soft lighting. "
            "Photorealistic lifestyle medical photography, cinematic warm tones, professional. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'after_coffee_female': {
        'sentence': 'Vital signs restored.',
        'full_prompt': (
            "Transform this female doctor's photo to show a visibly refreshed and revitalized doctor "
            "right after drinking her first coffee of the day. Warm satisfied smile, holding a steaming "
            "coffee cup with both hands, clearly renewed energy in her posture and expression. "
            "Background: cozy hospital break room with warm soft lighting. "
            "Photorealistic lifestyle medical photography, cinematic warm tones, professional. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'coffee_consultant_male': {
        'sentence': 'Fueled by caffeine and confidence.',
        'full_prompt': (
            "Transform this male doctor's photo into a supremely confident senior OB-GYN consultant "
            "who runs entirely on caffeine. Sharp pressed white coat, commanding and energetic posture, "
            "holding a large coffee cup raised triumphantly like a trophy, bold confident smile. "
            "Background: sleek modern hospital corridor or office. Power stance, full authority. "
            "Cinematic professional medical photography, dramatic lighting. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'coffee_consultant_female': {
        'sentence': 'Fueled by caffeine and confidence.',
        'full_prompt': (
            "Transform this female doctor's photo into a supremely confident senior OB-GYN consultant "
            "who runs entirely on caffeine. Sharp pressed white coat, commanding and energetic posture, "
            "holding a large coffee cup raised triumphantly like a trophy, bold confident smile. "
            "Background: sleek modern hospital corridor or office. Power stance, full authority. "
            "Cinematic professional medical photography, dramatic lighting. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'delivery_commander_male': {
        'sentence': 'Born to manage labor ward drama.',
        'full_prompt': (
            "Transform this male doctor's photo into a commanding OB-GYN delivery room chief who owns every situation. "
            "Strong confident posture with arms crossed or hands ready, calm but intensely authoritative expression, "
            "surgical scrubs or white coat, surrounded by the controlled chaos of a busy labor ward. "
            "Background: delivery room or labor ward with medical equipment visible. "
            "Leadership aura radiating, dramatic cinematic lighting, photorealistic medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'delivery_commander_female': {
        'sentence': 'Born to manage labor ward drama.',
        'full_prompt': (
            "Transform this female doctor's photo into a commanding OB-GYN delivery room chief who owns every situation. "
            "Strong confident posture with arms crossed or hands ready, calm but intensely authoritative expression, "
            "surgical scrubs or white coat, surrounded by the controlled chaos of a busy labor ward. "
            "Background: delivery room or labor ward with medical equipment visible. "
            "Leadership aura radiating, dramatic cinematic lighting, photorealistic medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'documentation_ninja_male': {
        'sentence': 'Fights unfinished notes daily.',
        'full_prompt': (
            "Transform this male doctor's photo into a documentation-obsessed OB-GYN doctor surrounded by "
            "towering stacks of medical charts, papers, and printed lab results. A glowing EMR computer screen "
            "full of unfinished notes dominates the background. Slightly overwhelmed but stubbornly determined "
            "expression, optional glasses perched on nose. Hospital workstation or nurse station environment. "
            "Realistic, slightly dramatic, slightly humorous tone. Photorealistic medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'documentation_ninja_female': {
        'sentence': 'Fights unfinished notes daily.',
        'full_prompt': (
            "Transform this female doctor's photo into a documentation-obsessed OB-GYN doctor surrounded by "
            "towering stacks of medical charts, papers, and printed lab results. A glowing EMR computer screen "
            "full of unfinished notes dominates the background. Slightly overwhelmed but stubbornly determined "
            "expression, optional glasses perched on nose. Hospital workstation or nurse station environment. "
            "Realistic, slightly dramatic, slightly humorous tone. Photorealistic medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'nightshift_survivor_male': {
        'sentence': "Hasn't seen sunlight in days.",
        'full_prompt': (
            "Transform this male doctor's photo into a night-shift survival legend who hasn't seen daylight "
            "in days. Pale, slightly hollowed face with haunted but resilient eyes that have seen too much. "
            "Wrinkled scrubs, tired posture but still standing with quiet pride. "
            "Background: a dark empty hospital corridor at 3am, minimal cold blue-tinted fluorescent lighting, "
            "eerie quiet atmosphere. Brave against all odds. Dramatic moody cinematic lighting, "
            "photorealistic medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
    'nightshift_survivor_female': {
        'sentence': "Hasn't seen sunlight in days.",
        'full_prompt': (
            "Transform this female doctor's photo into a night-shift survival legend who hasn't seen daylight "
            "in days. Pale, slightly hollowed face with haunted but resilient eyes that have seen too much. "
            "Wrinkled scrubs and hijab (if appropriate), tired posture but still standing with quiet pride. "
            "Background: a dark empty hospital corridor at 3am, minimal cold blue-tinted fluorescent lighting, "
            "eerie quiet atmosphere. Brave against all odds. Dramatic moody cinematic lighting, "
            "photorealistic medical photography. "
            "Keep the exact same face, eyes, nose, mouth, jawline, and skin tone from the input photo. "
            "Do not change facial structure. No cartoon, no anime, no watermark, no text overlay."
        ),
    },
}


def get_character_sentence(character: Optional[str]) -> Optional[str]:
    """Return the caption sentence for a given character key, or None if not found."""
    if not character:
        return None
    style = CHARACTER_STYLES.get(character.lower())
    if not style:
        return None
    return style.get('sentence')


def start_face_generation(
    child_image_url: str,
    character: str = 'superman',
    raise_errors: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Start face swap using yan-ops/face_swap model.
    
    Args:
        child_image_url: URL of child's photo (source face)
        character: Character name
        
    Returns:
        Dict with prediction_id and status, or None if failed
    """
    try:
        _clear_last_start_error()
        print(f"[Replicate] Starting Face Swap for character: {character}", flush=True)
        print(f"[Replicate] Child/Source image: {child_image_url[:50]}...", flush=True)
        
        primary_model = _get_face_swap_model()
        fallback_model = _get_fallback_model(primary_model)
        uses_template = not _is_google_direct_model(primary_model)

        # Get character-specific settings
        style_config = CHARACTER_STYLES.get(character.lower(), CHARACTER_STYLES['superman'])
        
        # Get template image URL (target face)
        template_url = style_config.get('template_image')

        # Auto-fallback: if no template image available, switch to nano-banana (prompt-only)
        if uses_template and not template_url:
            print(f"[Replicate] No template for '{character}' — auto-switching to google/nano-banana.", flush=True)
            primary_model = 'google/nano-banana'
            fallback_model = None
            uses_template = False
        
        swap_weight = _get_swap_weight(character=character, style_config=style_config)

        print(f"[Replicate] Preferred model: {primary_model}", flush=True)
        if fallback_model:
            print(f"[Replicate] Fallback model: {fallback_model}", flush=True)
        print(f"  Source (child): {child_image_url[:50]}...", flush=True)
        if template_url:
            print(f"  Target (template): {template_url[:50]}...", flush=True)
        print(f"  Weight: {swap_weight}", flush=True)

        prediction = None
        model_used = primary_model
        try:
            if _is_google_direct_model(primary_model):
                return _create_google_direct_generation(
                    model_name=primary_model,
                    source_image=child_image_url,
                    character=character,
                    style_config=style_config
                )

            input_candidates = _build_input_candidates(
                model_name=primary_model,
                source_image=child_image_url,
                target_image=template_url,
                weight=swap_weight,
                character=character,
                style_config=style_config
            )
            prediction = _create_prediction_with_candidates(primary_model, input_candidates)
        except Exception as primary_error:
            if not fallback_model:
                raise primary_error
            print(f"[Replicate] Primary model failed, trying fallback. Error: {primary_error}", flush=True)
            model_used = fallback_model
            if _is_google_direct_model(fallback_model):
                return _create_google_direct_generation(
                    model_name=fallback_model,
                    source_image=child_image_url,
                    character=character,
                    style_config=style_config
                )

            fallback_candidates = _build_input_candidates(
                model_name=fallback_model,
                source_image=child_image_url,
                target_image=template_url,
                weight=swap_weight,
                character=character,
                style_config=style_config
            )
            prediction = _create_prediction_with_candidates(fallback_model, fallback_candidates)
        
        prediction_id = prediction.id
        print(f"[Replicate] Prediction started: {prediction_id} (model: {model_used})", flush=True)
        
        _clear_last_start_error()
        return {
            'prediction_id': prediction_id,
            'status': prediction.status,
            'created_at': time.time(),
            'model': model_used
        }
        
    except Exception as e:
        _set_last_start_error(e)
        safe_error = _safe_error_message(e)
        print(f"[Replicate] Failed to start prediction: {safe_error}", flush=True)
        import traceback
        traceback.print_exc()
        if raise_errors:
            raise FaceGenerationStartError(safe_error) from e
        return None


def check_prediction_status(prediction_id: str) -> Optional[Dict[str, Any]]:
    """
    Check the status of a face generation prediction.
    
    Args:
        prediction_id: The prediction ID from start_face_generation
        
    Returns:
        Dict with status and result URL if complete, None if failed
    """
    try:
        if prediction_id in GOOGLE_DIRECT_RESULTS:
            result = dict(GOOGLE_DIRECT_RESULTS[prediction_id])
            print(f"[Google] Returning cached direct result: {prediction_id}", flush=True)
            return result

        # Check status via API
        prediction = replicate.predictions.get(prediction_id)
        
        status = prediction.status
        
        result = {
            'prediction_id': prediction_id,
            'status': status,
        }
        
        if status == 'succeeded':
            output = prediction.output
            print(f"[Replicate] Raw output type: {type(output).__name__}", flush=True)
            print(f"[Replicate] Raw output value: {output}", flush=True)
            
            if output:
                # yan-ops/face_swap returns a dictionary with 'cache_url' and 'msg'
                # Other models might return a URL string or list of URLs
                if isinstance(output, dict):
                    # Dictionary format: {'cache_url': 'https://...', 'msg': 'succeed'}
                    # Try multiple possible key names
                    result_url = (output.get('cache_url') or 
                                output.get('url') or 
                                output.get('output_url') or
                                output.get('image') or
                                output.get('result'))
                    if not result_url:
                        print(f"[Replicate] ERROR: No URL found in output dict. Keys: {list(output.keys())}", flush=True)
                        print(f"[Replicate] Full output: {output}", flush=True)
                        result_url = None
                    else:
                        print(f"[Replicate] Extracted URL from dict: {result_url}", flush=True)
                elif isinstance(output, list):
                    # List format: ['https://...']
                    result_url = output[0] if output else None
                    print(f"[Replicate] Extracted URL from list: {result_url}", flush=True)
                else:
                    # String format: 'https://...'
                    result_url = output
                    print(f"[Replicate] URL is string: {result_url}", flush=True)
                
                if result_url:
                    result['result_url'] = result_url
                    print(f"[Replicate] ✓ Result URL ready: {result_url[:50]}...", flush=True)
                else:
                    print(f"[Replicate] ✗ ERROR: Could not extract URL from output", flush=True)
                
        elif status == 'failed':
            result['error'] = prediction.error
            print(f"[Replicate] Prediction failed: {prediction.error}", flush=True)
            
        return result
        
    except Exception as e:
        print(f"[Replicate] Failed to check status: {str(e)}", flush=True)
        return None


def test_connection() -> bool:
    """
    Test Replicate API connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Try to list models (requires valid API token)
        models = replicate.models.list()
        print("[Replicate] Connection successful!", flush=True)
        return True
    except Exception as e:
        print(f"[Replicate] Connection failed: {str(e)}", flush=True)
        print("[Replicate] Please check REPLICATE_API_TOKEN in .env file", flush=True)
        return False


if __name__ == "__main__":
    # Test the connection
    print("Testing Replicate connection...")
    test_connection()
