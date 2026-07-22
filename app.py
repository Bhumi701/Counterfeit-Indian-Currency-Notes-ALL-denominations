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
import cv2
import tempfile
import os

from main import analyze_note

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

    st.image(uploaded_file, caption="Uploaded note", use_column_width=True)

    if st.button("Analyze Note", type="primary"):
        with st.spinner("Running checks..."):
            try:
                result = analyze_note(image_path, denomination, tilt_path, backlit_path)

                verdict = result["verdict"]
                if verdict == "GENUINE":
                    st.success(f"**Verdict: {verdict}**")
                elif "SUSPECT" in verdict:
                    st.warning(f"**Verdict: {verdict}**")
                else:
                    st.error(f"**Verdict: {verdict}**")

                st.metric("Confidence score", result["final_score"])

                st.subheader("Why this verdict")
                for reason in result["reasons"]:
                    st.write(f"- {reason}")

                with st.expander("Agent-by-agent scores"):
                    st.json(result["agent_scores"])
                    st.caption(f"Weights used: {result['weights_used']}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
            finally:
                for p in [image_path, tilt_path, backlit_path]:
                    if p and os.path.exists(p):
                        os.remove(p)
else:
    st.info("Upload a note photo to begin.")