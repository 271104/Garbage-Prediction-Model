from model import GarbagePercentagePredictor

if __name__ == "__main__":
    predictor = GarbagePercentagePredictor(
        model_path="models/garbage_model.h5",
        image_dir="data/raw/images",
        labels_csv="data/labels.csv"
    )
    predictor.create_model()
    predictor.train_model(epochs=50)
