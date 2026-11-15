import os, tempfile, threading
from flask import Flask, request
from use_model import load_classifier, load_detector, detect_and_classify

MODEL   = os.environ.get("CV_MODEL", "mobilenet")
CLASSES = os.environ.get("CV_CLASSES", "./class_names.json")
WEIGHTS = os.environ.get("CV_WEIGHTS", "combined_mobilenet.weights.h5")

# load once at startup, not per request
classifier, img_size, label_map = load_classifier(MODEL, CLASSES, WEIGHTS)
detector = load_detector()
lock = threading.Lock()  # the backend fires one thread per capture

app = Flask(__name__)

@app.post("/analyze")
def analyze():
    file = request.files.get("image")
    if file is None:
        return {"error": "missing 'image' field"}, 400

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    try:
        file.save(tmp.name)
        with lock:
            predictions = detect_and_classify(
                tmp.name, detector, classifier, img_size, label_map
            )
    finally:
        os.remove(tmp.name)

    items = [
        {"name": label, "confidence": round(p["confidence"], 4)}
        for label, p in predictions.items()
    ]
    return {"items": items}
