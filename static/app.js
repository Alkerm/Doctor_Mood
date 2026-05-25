// ── State ──────────────────────────────────────────────────
let capturedPhoto    = null;
let selectedCharacter = null;
let videoStream      = null;

// ── DOM References ─────────────────────────────────────────
const cameraScreen     = document.getElementById('camera-screen');
const characterScreen  = document.getElementById('character-screen');
const processingScreen = document.getElementById('processing-screen');
const resultScreen     = document.getElementById('result-screen');

const cameraFeed    = document.getElementById('camera-feed');
const photoCanvas   = document.getElementById('photo-canvas');
const captureBtn    = document.getElementById('capture-btn');
const uploadBtn     = document.getElementById('upload-btn');
const fileInput     = document.getElementById('file-input');
const cameraError   = document.getElementById('camera-error');

const capturedPreview = document.getElementById('captured-preview');
const characterCards  = document.querySelectorAll('.character-card');
const retakeBtn1      = document.getElementById('retake-btn-1');
const retakeBtn2      = document.getElementById('retake-btn-2');
const resultImage     = document.getElementById('result-image');
const loadingText     = document.getElementById('loading-text');

// ── Image Quality ──────────────────────────────────────────
const BURST_FRAME_COUNT    = 3;
const BURST_FRAME_DELAY_MS = 140;

const faceDetector = ('FaceDetector' in window)
    ? new FaceDetector({ fastMode: true, maxDetectedFaces: 1 })
    : null;

// ── Init ───────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initCamera();
});

// ── Helpers ────────────────────────────────────────────────
function isValidImageDataUrl(value) {
    return typeof value === 'string' && /^data:image\/[a-zA-Z0-9.+-]+;base64,/.test(value);
}

async function parseJsonResponse(response) {
    const text = await response.text();
    if (!text) return null;
    try {
        return JSON.parse(text);
    } catch {
        throw new Error(`Server returned invalid JSON (HTTP ${response.status})`);
    }
}

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function updateLoadingText(text) {
    if (loadingText) loadingText.textContent = text;
}

// ── Camera ─────────────────────────────────────────────────
async function initCamera() {
    const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (!isSecure) {
        console.warn('Camera may not work on non-secure origin:', location.origin);
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoStream = stream;
        cameraFeed.srcObject = stream;
        cameraFeed.muted = true;
        cameraFeed.classList.add('flipped');
        cameraFeed.onloadedmetadata = () => {
            cameraFeed.play().catch(e => console.error('Video play error:', e));
        };
        cameraError.style.display = 'none';
        captureBtn.disabled = false;
    } catch (error) {
        console.error('Camera error:', error);
        cameraError.style.display = 'block';
        let msg = '⚠️ Camera access denied. Please allow camera permissions and refresh.';
        if (error.name === 'NotAllowedError')  msg = '⚠️ Camera blocked by user. Enable it in browser settings.';
        if (error.name === 'NotFoundError')    msg = '⚠️ No camera found on this device.';
        if (error.name === 'NotReadableError') msg = '⚠️ Camera is in use by another app.';
        cameraError.innerHTML = `<p>${msg}</p>`;
        captureBtn.disabled = true;
    }
}

// ── Image Quality Metrics ──────────────────────────────────
function computeImageQualityMetrics(imageData) {
    const { data, width, height } = imageData;
    const gray = new Float32Array(width * height);
    let brightnessSum = 0;

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        const v = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        gray[p] = v;
        brightnessSum += v;
    }

    const brightness = brightnessSum / gray.length;
    let sharpnessSum = 0, count = 0;

    for (let y = 1; y < height; y += 2) {
        for (let x = 1; x < width; x += 2) {
            const idx = y * width + x;
            sharpnessSum += Math.abs(gray[idx] - gray[idx - 1]) + Math.abs(gray[idx] - gray[idx - width]);
            count++;
        }
    }

    return { brightness, sharpness: count > 0 ? sharpnessSum / count : 0 };
}

async function detectFaceMetrics(canvas) {
    if (!faceDetector) return { available: false, detected: false };
    try {
        const faces = await faceDetector.detect(canvas);
        if (!faces || faces.length === 0) return { available: true, detected: false };
        const box = faces[0].boundingBox;
        const faceAreaRatio = (box.width * box.height) / (canvas.width * canvas.height);
        const dx = Math.abs(box.x + box.width / 2 - canvas.width / 2) / canvas.width;
        const dy = Math.abs(box.y + box.height / 2 - canvas.height / 2) / canvas.height;
        return { available: true, detected: true, faceAreaRatio, centerOffset: Math.sqrt(dx * dx + dy * dy) };
    } catch {
        return { available: false, detected: false };
    }
}

async function evaluateCanvasQuality(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const basic = computeImageQualityMetrics(imageData);
    const face  = await detectFaceMetrics(canvas);

    let score = basic.sharpness - (Math.abs(basic.brightness - 128) / 128 * 18);
    if (face.available) {
        score += face.detected ? (14 - face.centerOffset * 15) : -30;
    }
    return { score, brightness: basic.brightness, sharpness: basic.sharpness, face };
}

function drawFrameToCanvas(canvas, video) {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    // Flip horizontally to undo the mirror effect shown in the preview
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();
}

async function captureBestFrame(video, frameCount = BURST_FRAME_COUNT) {
    const frames = [];
    for (let i = 0; i < frameCount; i++) {
        drawFrameToCanvas(photoCanvas, video);
        const metrics = await evaluateCanvasQuality(photoCanvas);
        frames.push({ dataUrl: photoCanvas.toDataURL('image/png'), metrics });
        if (i < frameCount - 1) await wait(BURST_FRAME_DELAY_MS);
    }
    frames.sort((a, b) => b.metrics.score - a.metrics.score);
    return frames[0];
}

// ── Capture Photo ──────────────────────────────────────────
async function capturePhoto() {
    if (!cameraFeed.videoWidth || !cameraFeed.videoHeight) {
        alert('Camera is not ready yet. Please wait and try again.');
        return;
    }
    const originalLabel = captureBtn.innerHTML;
    captureBtn.disabled = true;
    captureBtn.innerHTML = '<span class="btn-icon">⏳</span> Capturing...';

    try {
        const best = await captureBestFrame(cameraFeed);
        capturedPhoto = best.dataUrl;
        capturedPreview.src = capturedPhoto;
        if (videoStream) videoStream.getTracks().forEach(t => t.stop());
        switchScreen('character');
    } catch (err) {
        console.error('Capture failed:', err);
        alert('Could not capture photo. Please try again.');
    } finally {
        captureBtn.innerHTML = originalLabel;
        captureBtn.disabled = false;
    }
}

// ── File Upload ────────────────────────────────────────────
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
        capturedPhoto = ev.target.result;
        capturedPreview.src = capturedPhoto;
        if (videoStream) videoStream.getTracks().forEach(t => t.stop());
        switchScreen('character');
    };
    reader.readAsDataURL(file);
}

// ── Gender Toggle ──────────────────────────────────────────
function setGender(gender) {
    const btnMale   = document.getElementById('btn-male');
    const btnFemale = document.getElementById('btn-female');
    const gridMale  = document.getElementById('grid-male');
    const gridFemale = document.getElementById('grid-female');

    if (gender === 'male') {
        btnMale.className   = 'gender-btn active';
        btnFemale.className = 'gender-btn inactive';
        gridMale.style.display   = 'grid';
        gridFemale.style.display = 'none';
    } else {
        btnMale.className   = 'gender-btn inactive';
        btnFemale.className = 'gender-btn active';
        gridMale.style.display   = 'none';
        gridFemale.style.display = 'grid';
    }

    // Clear selection when toggling
    characterCards.forEach(c => c.classList.remove('selected'));
    selectedCharacter = null;
}

// ── Character Selection ────────────────────────────────────
function selectCharacter(card) {
    characterCards.forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    selectedCharacter = card.dataset.character;
    setTimeout(() => performFaceSwap(), 500);
}

// ── Face Swap ─────────────────────────────────────────────
async function performFaceSwap() {
    if (!capturedPhoto || !selectedCharacter) {
        alert('Please capture a photo and select a character.');
        return;
    }
    if (!isValidImageDataUrl(capturedPhoto)) {
        alert('Your photo data is invalid. Please retake or upload again.');
        return;
    }

    switchScreen('processing');
    updateLoadingText('Uploading your photo...');

    try {
        const response = await fetch('/swap-face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                child_photo: capturedPhoto,
                character: selectedCharacter
            })
        });

        const data = await parseJsonResponse(response);

        if (!response.ok) {
            throw new Error((data && data.error) || `Request failed (HTTP ${response.status})`);
        }

        if (!data || !data.prediction_id) {
            throw new Error('Server did not return a prediction ID.');
        }

        const predictionId = data.prediction_id;
        updateLoadingText('AI is generating your character...');

        const resultUrl = await pollForResult(predictionId);

        if (resultUrl) {
            resultImage.src = resultUrl;
            generateQRCode(resultUrl);
            switchScreen('result');
        } else {
            throw new Error('Failed to generate result image.');
        }

    } catch (error) {
        console.error('Face swap error:', error);
        let message = error && error.message ? error.message : 'Unknown error';
        if (message === 'Failed to fetch') {
            message = 'Cannot connect to server. Make sure Flask is running and try again.';
        }
        alert(`Something went wrong:\n${message}\n\nPlease try again.`);
        switchScreen('character');
    }
}

// ── Polling ────────────────────────────────────────────────
async function pollForResult(predictionId, maxAttempts = 60) {
    let attempts = 0;

    while (attempts < maxAttempts) {
        try {
            const response = await fetch(`/check-status/${predictionId}`);
            if (!response.ok) throw new Error('Failed to check status');

            const data = await parseJsonResponse(response);
            if (!data) throw new Error('Empty status response from server');

            const status = data.status;
            console.log(`[Poll ${attempts + 1}] Status: ${status}`);

            if (status === 'succeeded') return data.result_url;
            if (status === 'failed')    throw new Error(data.error || 'Generation failed');

            const elapsed = attempts * 2;
            if (elapsed < 10)       updateLoadingText('Starting AI processing...');
            else if (elapsed < 25)  updateLoadingText('Analyzing your medical personality...');
            else if (elapsed < 45)  updateLoadingText('Adding the finishing touches...');
            else                    updateLoadingText('Almost done...');

            await wait(2000);
            attempts++;

        } catch (error) {
            console.error('Polling error:', error);
            throw error;
        }
    }

    throw new Error('Timeout: AI took too long. Please try again.');
}

// ── QR Code ────────────────────────────────────────────────
async function generateQRCode(imageUrl) {
    try {
        const response = await fetch('/generate-qr', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_url: imageUrl })
        });
        if (!response.ok) return;
        const data = await response.json();
        const qrImg = document.getElementById('qr-code');
        if (qrImg && data.qr_code) {
            qrImg.src = data.qr_code;
            qrImg.style.display = 'block';
        }
    } catch (err) {
        console.error('QR generation failed:', err);
    }
}

// ── Retake ─────────────────────────────────────────────────
function retakePhoto() {
    capturedPhoto     = null;
    selectedCharacter = null;
    characterCards.forEach(c => c.classList.remove('selected'));
    initCamera();
    switchScreen('camera');
}

// ── Screen Switcher ────────────────────────────────────────
function switchScreen(screen) {
    [cameraScreen, characterScreen, processingScreen, resultScreen]
        .forEach(s => s.classList.remove('active'));

    const map = {
        camera:     cameraScreen,
        character:  characterScreen,
        processing: processingScreen,
        result:     resultScreen,
    };
    if (map[screen]) map[screen].classList.add('active');
}

// ── Event Listeners ────────────────────────────────────────
function setupEventListeners() {
    captureBtn.addEventListener('click', capturePhoto);
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);

    characterCards.forEach(card => {
        card.addEventListener('click', () => selectCharacter(card));
    });

    retakeBtn1.addEventListener('click', retakePhoto);
    retakeBtn2.addEventListener('click', retakePhoto);
}

// ── Cleanup ────────────────────────────────────────────────
window.addEventListener('beforeunload', () => {
    if (videoStream) videoStream.getTracks().forEach(t => t.stop());
});
