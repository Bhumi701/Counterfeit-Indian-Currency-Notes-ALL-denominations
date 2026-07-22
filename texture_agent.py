"""
texture_agent.py -- Inference wrapper for the trained texture/print-quality
CNN (see train_texture_agent.py). Loads the saved model once and scores
new note images.
"""

import numpy as np
import cv2

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - runtime fallback for lightweight deployments
    tf = None

IMG_SIZE = (160, 160)


class TextureAgent:
    def __init__(self, model_path="texture_agent_model.keras"):
        self.model = None
        self.loaded = False
        if tf is None:
            return

        try:
            self.model = tf.keras.models.load_model(model_path)
            self.loaded = True
        except (OSError, ValueError, ImportError):
            self.model = None
            self.loaded = False

    def run(self, image_bgr):
        if not self.loaded:
            return {
                "agent": "texture",
                "score": 0.5,
                "checks": {"note": f"Model not trained yet -- run train_texture_agent.py first. "
                                    f"Using neutral score for now."},
            }

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=0).astype("float32")

        pred = self.model.predict(img, verbose=0)[0][0]   # 0 = authentic, 1 = tampered
        authentic_score = 1.0 - float(pred)

        return {
            "agent": "texture",
            "score": round(authentic_score, 3),
            "checks": {"raw_model_output": round(float(pred), 3),
                       "interpretation": "closer to 0 = authentic, closer to 1 = tampered"},
        }