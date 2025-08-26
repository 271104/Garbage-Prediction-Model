import os
import csv
import glob
from tkinter import Tk, Label, Scale, HORIZONTAL, Button, StringVar
from PIL import Image, ImageTk

class GarbageDatasetPreparer:
    def __init__(self, root="data"):
        self.root = root
        self.raw_images = os.path.join(root, "raw", "images")

    def create_dataset_structure(self):
        os.makedirs(self.raw_images, exist_ok=True)


class ManualLabelingTool:
    def __init__(self, images_dir="data/raw/images", labels_csv="data/labels.csv"):
        self.images_dir = images_dir
        self.labels_csv = labels_csv
        self.images = sorted(
            [p for p in glob.glob(os.path.join(images_dir, "*")) 
             if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))]
        )
        self.idx = 0
        self.current_img = None
        self.tk_img = None
        self.window = None
        self.percent_scale = None
        self.path_label_text = None   # <-- will be created later with Tk()

        os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
        if not os.path.exists(labels_csv):
            with open(labels_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "percentage"])

    def save_current_label(self):
        if not self.images:
            return
        img_path = self.images[self.idx]
        pct = self.percent_scale.get()
        with open(self.labels_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(img_path), pct])
        self.next_image()

    def next_image(self):
        self.idx += 1
        if self.idx >= len(self.images):
            self.path_label_text.set("✅ All images labeled. You can close the window.")
            return
        self.load_and_show()

    def load_and_show(self):
        path = self.images[self.idx]
        self.path_label_text.set(path)
        img = Image.open(path).convert("RGB")
        img.thumbnail((800, 600))
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)

    def run(self):
        if not self.images:
            print("⚠️ No images found in:", self.images_dir)
            return

        self.window = Tk()
        self.window.title("Garbage % Labeling Tool")

        # ✅ Create StringVar only after Tk() exists
        self.path_label_text = StringVar()

        self.image_label = Label(self.window)
        self.image_label.pack()

        Label(self.window, textvariable=self.path_label_text).pack(pady=4)

        self.percent_scale = Scale(
            self.window, from_=0, to=100, orient=HORIZONTAL,
            length=400, label="Garbage percentage"
        )
        self.percent_scale.set(50)
        self.percent_scale.pack(pady=8)

        Button(self.window, text="Save & Next", command=self.save_current_label).pack(pady=8)

        self.idx = 0
        self.load_and_show()
        self.window.mainloop()
