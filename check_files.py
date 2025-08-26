import os, csv

with open("data/labels.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        path = os.path.join("data/raw/images", row["filename"])
        if not os.path.exists(path):
            print("❌ Missing:", path)
