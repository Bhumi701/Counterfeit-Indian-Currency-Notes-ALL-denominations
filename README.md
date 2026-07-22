# Counterfeit Indian Currency Notes Detector

This project detects counterfeit Indian currency notes using a multi-agent pipeline with:
- pattern analysis
- OCR-based checks
- texture/print-quality analysis
- optional thread and feature-detection checks

## Run the API locally

```bash
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate
pip install -r requirements.txt
python api.py
```

The API will be available at:
- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

## Main endpoints
- GET /health
- POST /analyze

## Notes
- For a quick demo, start the server and call the `/analyze` endpoint with a note image and denomination.
- If TensorFlow model files are not available, the system falls back to a neutral score for the texture agent so the API still starts.

## Windows: Tesseract OCR (required for OCR checks)

This project uses Tesseract for OCR. On Windows you must install the Tesseract binary and ensure `tesseract.exe` is on your PATH.

1. Download the installer (recommended build): https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and accept defaults (it typically installs to `C:\Program Files\Tesseract-OCR`).
3. If the installer doesn't add Tesseract to your PATH, open PowerShell as your user and run:

```powershell
setx PATH "$env:PATH;C:\Program Files\Tesseract-OCR"
# Close and reopen your terminal (or run: $env:PATH += ';C:\Program Files\Tesseract-OCR')
tesseract --version
```

If `tesseract --version` prints the version, restart your API server and retry the `/analyze` endpoint.

## Streamlit Cloud deployment notes

- Streamlit Cloud may use a newer Python runtime where some binary wheels (notably OpenCV) are not available. If your deployed app fails with "No module named 'cv2'", the app now shows a friendly error in the UI explaining this. Options to resolve:
	1. Deploy using a Python runtime that matches the wheel availability (e.g., Python 3.11) if the platform allows choosing a runtime.
	2. Supply a custom Docker image for deployment with OpenCV pre-installed.
	3. Run the app locally (recommended for immediate demos) where you can control the Python version and install `opencv-python-headless`.

### Quick: deploy with Docker (recommended for Streamlit-hosted or other container hosts)

Build and run the included Docker image (this forces a compatible Python runtime and installs system libs for OpenCV/Tesseract):

```bash
docker build -t counterfeit-notes .
docker run -p 8501:8501 counterfeit-notes
```

Then open http://localhost:8501 to use the app.



