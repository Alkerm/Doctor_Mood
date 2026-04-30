# Photo Booth Face Swap Application

A web-based photo booth application that captures live photos via camera, swaps faces with superhero characters using AI, and enables printing.

## Features

- 📸 **Live Camera Preview** - Real-time camera feed with WebRTC
- 🦸 **Character Selection** - Choose from Superman, Batman, Spider-Man, Wonder Woman
- ✨ **AI Face Swap** - Powered by Hugging Face API (free tier)
- 🖨️ **Print Functionality** - Print your superhero transformation
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Create `.env` file for API configuration:
```bash
cp .env.example .env
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Allow camera permissions when prompted

4. Follow the on-screen instructions:
   - Position yourself in the camera
   - Click "Capture Photo"
   - Select a superhero character
   - Wait for processing
   - Print or retake!

## Requirements

- Python 3.8+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Camera/webcam access
- Internet connection (for AI processing)

## API Usage

The application uses Hugging Face's free inference API:
- **Free tier**: 30,000 requests/month
- **Perfect for**: 10-15 devices with moderate usage
- **No GPU required** on your server

## Project Structure

```
KHAL/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── templates/
│   └── index.html        # Main HTML interface
├── static/
│   ├── style.css         # CSS styling
│   ├── app.js            # JavaScript logic
│   └── characters/       # Superhero character images
│       ├── superman.png
│       ├── batman.png
│       ├── spiderman.png
│       └── wonderwoman.png
└── README.md             # This file
```

## Deployment

For production deployment:
1. Use a production WSGI server (gunicorn, waitress)
2. Set `FLASK_ENV=production` in `.env`
3. Configure HTTPS for camera access
4. Use a reverse proxy (nginx, Apache)

### Render quick deploy

1. Push this repo to GitHub (already done).
2. On Render, create a new Web Service from the repo.
3. Render will use:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app`
4. Set env vars from `.env.example` (at least `FLASK_SECRET_KEY`, `ADMIN_API_KEY`, `GEMINI_API_KEY`, Cloudinary keys).
5. Open:
   - `/` for user app
   - `/admin` for admin panel

Important: current app uses SQLite (`DB_PATH=app.db`). On many free cloud instances, local disk may reset after redeploy/restart. For persistent user/trial data, move DB to managed PostgreSQL.

## Authentication And Trial Limits

This app now includes:
- Email/password login/signup (`/auth/signup`, `/auth/login`)
- Per-user usage limit (default `0`)
- Admin endpoint to add more trials (`/admin/add-trials`)
- Admin page at `/admin` to list users and add trials

### Required/Important environment variables

- `FLASK_SECRET_KEY`: Strong random string for signing sessions
- `DB_PATH`: SQLite DB path (default `app.db`)
- `DEFAULT_TRIALS`: New users initial allowed uses (default `0`)
- `ADMIN_API_KEY`: Secret key for admin API
- `GEMINI_API_KEY`: Google AI Studio API key for `FACE_SWAP_MODEL=google/nano-banana`
- `FACE_SWAP_MODEL`: Defaults to `google/nano-banana`, which calls Google's direct `gemini-2.5-flash-image` model

### Add more trials for a user

`POST /admin/add-trials`

Headers:
- `X-Admin-Key: <ADMIN_API_KEY>`

JSON body:
```json
{
  "email": "user@example.com",
  "additional_uses": 20
}
```

## License

MIT License - Feel free to use and modify!
