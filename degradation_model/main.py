import os
from configs import *
from PIL import Image
from model import DegradationModel
from torchvision import transforms
from utils.image import crop_and_show_images
import matplotlib.pyplot as plt


def main():
    PATH = "benchmarks/Set14/HR"
    out_dir = "benchmarks/Set14/LR/"
    os.makedirs(out_dir, exist_ok=True)

    for path in os.listdir(PATH):
        if not path.endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(PATH, path)
        # PATH = "images/barbara.png"
        img = Image.open(path).convert("RGB")
        image_tensor = transforms.ToTensor()(img)
        degradation_model = DegradationModel(SETTINGS)
        lr_img = degradation_model.random_shuffle_degradations(
            image_tensor.unsqueeze(0), scale_factor=0.25
        )
        lr_img = transforms.ToPILImage()(lr_img.squeeze(0))
        lr_img.save(out_dir + os.path.basename(path))

        # (x1, y1, x2, y2)
        # boxes = [(70, 100, 200, 250)]
        # crop_and_show_images(img, bounding_boxes=boxes, colors=["red", "green"])
        # crop_and_show_images(lr_img, bounding_boxes=boxes, colors=["red", "green"])


if __name__ == "__main__":
    main()
