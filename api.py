"""
api.py -- REST API for the counterfeit note detector, for your friend's
frontend/backend to call over HTTP.

USAGE (run locally):
    pip install fastapi "uvicorn[standard]" python-multipart pyngrok
    python api.py

This starts a server on http://localhost:8000
Interactive docs (useful for testing): http://localhost:8000/docs

TO GET A PUBLIC URL for your friend (fastest option, no cloud account needed):
    Option A -- pyngrok (built into this script, see bottom):
        Just run `python api.py` -- it will print a public https:// URL
        automatically if ngrok is installed (see setup below).

    Option B -- manual ngrok:
        1. Download ngrok: https://ngrok.com/download
        2. In a separate terminal: ngrok http 8000
        3. It prints a URL like https://xxxx.ngrok-free.app -- give this to
           your friend. Their frontend calls this URL + "/analyze"

NOTE: ngrok's free tier URL changes every time you restart it, and only
works while your laptop + this script are running. For a permanent link
you'd need real cloud hosting (Render/Railway/etc), which takes longer to
set up -- ngrok is the right choice for "deploy this today" urgency.
"""

import os
import tempfile

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from main import analyze_note

app = FastAPI(title="Counterfeit Note Detector API")

# Allow requests from any frontend origin -- fine for a hackathon demo.
# For production you'd restrict this to your friend's actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _save_upload(upload: UploadFile):
    if upload is None:
        return None
    suffix = os.path.splitext(upload.filename)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await upload.read()
    tmp.write(content)
    tmp.close()
    return tmp.name


@app.get("/")
def root():
    return {"status": "ok", "message": "Counterfeit Note Detector API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(
    denomination: str = Form(..., description="One of: 10, 20, 50, 100, 200, 500"),
    image: UploadFile = File(..., description="Straight-on photo of the note"),
    tilt: UploadFile = File(None, description="Optional: tilted photo for thread color-shift check"),
    backlit: UploadFile = File(None, description="Optional: backlit photo for see-through register check"),
):
    """
    Main endpoint. Your friend's frontend sends a multipart/form-data POST
    request here with the image (and denomination), gets back JSON like:

    {
      "final_score": 0.72,
      "verdict": "GENUINE",
      "reasons": [...],
      "agent_scores": {...},
      "weights_used": {...}
    }
    """
    image_path = await _save_upload(image)
    tilt_path = await _save_upload(tilt)
    backlit_path = await _save_upload(backlit)

    try:
        result = analyze_note(image_path, denomination, tilt_path, backlit_path)
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        for p in [image_path, tilt_path, backlit_path]:
            if p and os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    import uvicorn

    PORT = 8000

    # Try to auto-start ngrok and print a public URL. If pyngrok isn't
    # installed or ngrok isn't set up, this just skips it silently --
    # you can still run ngrok manually in another terminal (see docstring).
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(PORT)
        print(f"\n{'='*60}")
        print(f"PUBLIC URL for your friend: {public_url}")
        print(f"They should call: {public_url}/analyze")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n(ngrok auto-start skipped: {e})")
        print("Run 'ngrok http 8000' in another terminal to get a public URL.\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
    