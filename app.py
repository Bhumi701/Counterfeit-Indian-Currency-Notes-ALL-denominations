"""
app.py -- Streamlit web app for demoing the counterfeit note detector.
This is the "deploy" layer -- wraps main.py's analyze_note() in a simple
upload-and-check UI that works as a hackathon demo.

USAGE:
    pip install streamlit
    streamlit run app.py

This opens a browser tab where you can upload a note photo and get the
verdict instantly. Works fully offline/locally, no cloud deployment needed
for a demo.
"""

import streamlit as st
import numpy as np
import tempfile
import os

# Delay heavy imports (cv2, main) until runtime so the app can start
# even if the deployment environment lacks OpenCV. When a user clicks
# "Analyze Note" we attempt to import the pipeline and show a helpful
# error if the import fails (e.g. no `cv2` on the platform).

st.set_page_config(page_title="Counterfeit Note Detector", page_icon="\U0001F4B5", layout="centered")

st.title("Counterfeit Currency Detector")
st.caption("AI-powered verification for Indian currency notes (Mahatma Gandhi New Series, 2016+)")

denomination = st.selectbox("Denomination", ["10", "20", "50", "100", "200", "500"])
uploaded_file = st.file_uploader("Upload a note photo (straight-on, well-lit)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    tilt_file = st.file_uploader("Optional: tilted photo (for thread color-shift check)", type=["jpg", "jpeg", "png"], key="tilt")
with col2:
    backlit_file = st.file_uploader("Optional: backlit photo (for see-through register check)", type=["jpg", "jpeg", "png"], key="backlit")

def save_temp(uploaded):
    if uploaded is None:
        return None
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    return tmp.name

if uploaded_file is not None:
    image_path = save_temp(uploaded_file)
    tilt_path = save_temp(tilt_file)
    backlit_path = save_temp(backlit_file)

    st.image(uploaded_file, caption="Uploaded note", width=700)

    if st.button("Analyze Note", type="primary"):
        with st.spinner("Running checks..."):
            try:
                # import the analysis entrypoint lazily
                try:
                    from main import analyze_note
                    _can_run = True
                except Exception as ie:
                    st.error(
                        "Cannot run analysis: required native dependency missing (e.g. OpenCV). "
                        "On Streamlit Cloud some binary wheels (opencv) may be unavailable. "
                        "To continue, either deploy using a Python runtime that supports OpenCV or "
                        "modify requirements / use a custom Docker image.\n\n"
                        f"Import error: {ie}"
                    )
                    _can_run = False

                if not _can_run:
                    # Skip running the pipeline when imports failed.
                    result = {"error": "analysis skipped due to missing native dependencies"}
                else:
                    result = analyze_note(image_path, denomination, tilt_path, backlit_path)

                # Handle error responses from the analysis pipeline
                if not isinstance(result, dict):
                    st.error("Analysis failed: unexpected result type")
                elif "error" in result:
                    st.error(f"Analysis error: {result['error']}")
                else:
                    verdict = result.get("verdict")
                    if not verdict:
                        st.error("Analysis produced no `verdict` — check server logs")
                    else:
                        if verdict == "GENUINE":
                            st.success(f"**Verdict: {verdict}**")
                        elif "SUSPECT" in verdict:
                            st.warning(f"**Verdict: {verdict}**")
                        else:
                            st.error(f"**Verdict: {verdict}**")

                        st.metric("Confidence score", result.get("final_score", "N/A"))

                        st.subheader("Why this verdict")
                        for reason in result.get("reasons", []):
                            st.write(f"- {reason}")

                        with st.expander("Agent-by-agent scores"):
                            st.json(result.get("agent_scores", {}))
                            st.caption(f"Weights used: {result.get('weights_used', {})}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
            finally:
                for p in [image_path, tilt_path, backlit_path]:
                    if p and os.path.exists(p):
                        os.remove(p)
else:
    st.info("Upload a note photo to begin.")