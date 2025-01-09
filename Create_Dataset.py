import pandas as pd
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from PIL import Image
from io import BytesIO
import os

def create_parquet(dataset, path):
    images, labels = [], []
    for img, label in dataset:
        buffer = BytesIO()
        img.save(buffer, format="PNG")  # Save as PNG
        images.append(buffer.getvalue())
        labels.append(label)
    df = pd.DataFrame({"image": images, "label": labels})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def prepare_dataset():
    transform = Compose([Resize((32, 32)), ToTensor(), ToPILImage()])
    train_dataset = MNIST(root="./datasets", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./datasets", train=False, download=True, transform=transform)

    create_parquet(train_dataset, "mnist/train-00000-of-00001.parquet")
    create_parquet(test_dataset, "mnist/test-00000-of-00001.parquet")
