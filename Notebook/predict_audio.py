import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image

# ===========================================
# Load model ONCE
# ===========================================
MODEL_PATH = r"D:\Programming\Alzheimers Detection\model.hdf5"
model = load_model(MODEL_PATH)

# ===========================================
# Match TRAINING mel-spectrogram processing
# ===========================================
def create_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = librosa.util.normalize(S_db)   # <<< IMPORTANT
    return S_db


# ===========================================
# Prediction function
# ===========================================
def predict_audio(filepath):

    # Load audio
    y, sr = librosa.load(filepath, sr=None)

    # Convert to mel-spectrogram (normalized)
    spec = create_mel_spectrogram(y, sr)

    # Render spectrogram EXACTLY like training
    fig = plt.figure(figsize=(3, 3))
    plt.axis("off")
    librosa.display.specshow(spec, cmap="magma")
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Resize & preprocess for Xception
    img = image.smart_resize(img, (250, 250))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0]
    label = "impaired" if pred[0] > pred[1] else "normal"

    return label, pred


# ===========================================
# Example test call
# ===========================================
if __name__ == "__main__":
    test_path = r"D:\Programming\Alzheimers Detection\dataset_audio\test\impaired\impaired_80.wav"
    label, raw = predict_audio(test_path)
    print("Prediction:", label)
    print("Raw:", raw)
