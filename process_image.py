import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import TinyImageNetDataset


# dataset is 64x64
dataset = TinyImageNetDataset('./data/tiny-imagenet-200', preload=False)

num_patches = 4
image_res = 64
block_size = image_res // num_patches

for idx, data in enumerate(dataset):
    img, label = data['image'], data['label']

    xs = np.arange(0, image_res, block_size)
    ys = np.arange(0, image_res, block_size)
    splits = [img[xi:xi+block_size, yi:yi+block_size] for xi in xs for yi in ys]

    for image_arr in splits:
        image = Image.fromarray(image_arr)
        plt.imshow(image)
        plt.show()

    break
