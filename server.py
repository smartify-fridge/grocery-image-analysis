import os, threading
from flask import Flask, request
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from use_model_server import DEFAULT_SETTINGS, make_settings, load_models, analyze_image

settings = make_settings(
    weights=os.environ.get("CV_WEIGHTS", DEFAULT_SETTINGS["weights"]),
    classifier_classes=os.environ.get("CV_CLASSES", DEFAULT_SETTINGS["classifier_classes"]),
    crop_mode=os.environ.get("CV_CROP_MODE", DEFAULT_SETTINGS["crop_mode"]),
)

# phone photos arrive as HEIC; same handling as convert_img_format.py
register_heif_opener()

# load once at startup, not per request
models = load_models(settings)
lock = threading.Lock()  # the backend fires one thread per capture

app = Flask(__name__)

@app.post("/analyze")
def analyze():
    file = request.files.get("image")
    if file is None:
        return {"error": "missing 'image' field"}, 400

    try:
        image = ImageOps.exif_transpose(Image.open(file.stream)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return {"error": "'image' is not a readable image"}, 400

    with lock:
        kept = analyze_image(image, settings, **models)

    # kept is sorted by confidence, so the first of each label is its best
    items = {}
    for p in kept:
        items.setdefault(p["predicted_label"], {
            "name": p["predicted_label"],
            "confidence": round(p["classifier_confidence"], 4)
        })
    return {"items": list(items.values())}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8002)))
