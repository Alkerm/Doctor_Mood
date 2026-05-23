# What Kind of OB-GYN Are You? 👨‍⚕️🩺

A premium, interactive AI-powered photo booth web application built for OB-GYN doctors. The app captures a live photo via the webcam or allows an image upload, applies a state-of-the-art AI transformation based on 16 hilarious doctor personalities (8 male / 8 female), overlays dynamic magazine-style captions, and outputs a downloadable character card complete with a shareable QR code.

---

## ✨ Features

- **📸 Dual Input Channels**: Real-time high-quality webcam capture with custom burst-frame evaluation for the best picture quality, or instant file uploads.
- **🩺 16 OB-GYN Personalities**: Harmonious male and female card selections spanning 8 archetypes:
  1. *Before OB Residency* ("Smiling before discovering labor ward reality.")
  2. *After 24-Hour Call* ("Running on 2% battery.")
  3. *Emergency C-Section* ("Activated survival mode.")
  4. *After Coffee* ("Vital signs restored.")
  5. *Coffee-Powered Consultant* ("Fueled by caffeine and confidence.")
  6. *Delivery Room Commander* ("Born to manage labor ward drama.")
  7. *Documentation Ninja* ("Fights unfinished notes daily.")
  8. *Night-Shift Survivor* ("Hasn't seen sunlight in days.")
- **🎭 Intelligent AI Face Blending**: Integrates Gemini AI and Replicate's high-fidelity image models to flawlessly blend faces onto stylized custom character card layouts.
- **🎨 Dark Medical Glassmorphism Design**: Designed with standard HSL tailored colors, gorgeous glow states, micro-interactions, responsive grids, and beautiful visual feedback.
- **💬 Magazine-Style Overlays**: High-fidelity Pil-based overlay engine that draws semi-transparent backdrops and high-contrast captions on the final output image.
- **📱 QR Code Generation**: Instantly generates scan-ready QR codes of the output images, allowing users to save results to their smartphones immediately.
- **🚀 Vercel-Ready Deployment**: Configured to run out-of-the-box on Vercel Serverless Functions with zero databases or local state dependencies.

---

## 🛠️ Tech Stack

- **Backend**: Python + Flask + PIL/Pillow
- **AI Processing**: Gemini AI (`gemini-2.5-flash` / `google/nano-banana`) & Replicate
- **Media Hosting**: Cloudinary (temp asset storage)
- **Frontend**: Vanilla HTML5 + Custom Modern CSS + JavaScript (Modern Camera WebRTC + Canvas + QR)
- **Deployment**: Vercel (Configured via `vercel.json` and serverless Python rules)

---

## 📦 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root of the project:
```env
# ── Gemini AI (REQUIRED) ───────────────────────────────────
GEMINI_API_KEY=your_gemini_api_key_here

# ── Cloudinary (REQUIRED) ──────────────────────────────────
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret

# ── Advanced Config (Optional) ─────────────────────────────
FACE_SWAP_MODEL=google/nano-banana
GOOGLE_DIRECT_MAX_ATTEMPTS=1
```

### 4. Run Locally
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000` 🚀

---

## ⚡ Deployment to Vercel

This repository is pre-configured for instant **Vercel** serverless deployment:

1. **Push your code** to your GitHub repository (e.g., `https://github.com/Alkerm/Doctor_Mood.git`).
2. Log into your **Vercel** account, click **Add New Project**, and import your repository.
3. In the project **Environment Variables** settings, add the exact variables from your `.env` file:
   - `GEMINI_API_KEY`
   - `CLOUDINARY_CLOUD_NAME`
   - `CLOUDINARY_API_KEY`
   - `CLOUDINARY_API_SECRET`
   - `FACE_SWAP_MODEL=google/nano-banana`
4. Click **Deploy**. Vercel will build the serverless functions and serve the static files instantly!

---

## 📂 Project Structure
```
Doctor_Mood/
├── app.py                 # Core Flask Server / API Endpoints
├── replicate_helper.py    # AI Generation & Gemini Prompting Engine
├── cloudinary_helper.py   # Temporary Cloudinary Upload Handler
├── vercel.json           # Vercel Serverless Routing Config
├── requirements.txt       # Production Dependencies
├── templates/
│   └── index.html        # Glassmorphic Frontend Dashboard
└── static/
    ├── app.js            # Camera capture, status polling, and rendering logic
    └── style.css         # Beautiful responsive dark medical theme styling
```

---

## 📜 License
MIT License. Feel free to use and customize for your events and activities!
