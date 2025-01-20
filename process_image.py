import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dataset import TinyImageNetDataset


dataset = TinyImageNetDataset('./data/tiny-imagenet-200', preload=False)

for idx, data in enumerate(dataset):
    img, label = data['image'], data['label']
    image = Image.fromarray(img)
    plt.imshow(image)
    plt.show()
    print(label)
