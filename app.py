import os
from flask import Flask, render_template, request, url_for
from model import GarbagePercentagePredictor

app = Flask(__name__)
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

predictor = GarbagePercentagePredictor(
    model_path="models/garbage_model.h5",
    image_dir="data/raw/images",
    labels_csv="data/labels.csv"
)
predictor.load()

@app.route("/", methods=["GET", "POST"])
def index():
    img_url, percent, confidence = None, None, None
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename:
            save_path = os.path.join(UPLOAD_DIR, f.filename)
            f.save(save_path)
            percent, confidence = predictor.predict_percent_and_confidence(save_path)
            img_url = url_for("static", filename=f"uploads/{f.filename}")
    return render_template("index.html", img_url=img_url, percent=percent, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
