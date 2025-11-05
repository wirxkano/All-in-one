import math
import itertools
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def crop_and_show_images(
    image: Image.Image, bounding_boxes: list, colors: list = ["red"], cols=1
):
    cropped = []
    colors = itertools.cycle(colors)
    draw = ImageDraw.Draw(image)

    for box, color in zip(bounding_boxes, colors):
        draw.rectangle(box, outline=color, width=2)
        cropped.append(image.crop(box))

    rows = math.ceil(len(cropped) / cols)
    fig = plt.figure(figsize=(4 + cols * 3, rows * 3))
    ax_main = plt.subplot2grid((rows, cols + 2), (0, 0), rowspan=rows, colspan=2)
    ax_main.imshow(image)
    ax_main.axis("off")

    for i, crop in enumerate(cropped):
        r, c = divmod(i, cols)
        ax = plt.subplot2grid((rows, cols + 2), (r, 2 + c))
        ax.imshow(crop)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
