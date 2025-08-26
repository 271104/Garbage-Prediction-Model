from model import GarbagePercentagePredictor

predictor = GarbagePercentagePredictor()
predictor.load()

img_path = "data/raw/images/your_test_image.jpg"  # change to any sample image
pct, conf = predictor.predict_percent_and_confidence(img_path)

print(f"Predicted Garbage %: {pct:.2f}")
print(f"Confidence: {conf:.2f}%")
