import argparse
import os

from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir")
    parser.add_argument("--save-to", dest="save_to")
    parser.add_argument("--size", dest="size", type=int)
    args = parser.parse_args()

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)

    resize = Resize(args.size)
    for fname in tqdm(os.listdir(args.dir)):
        img = Image.open(os.path.join(args.dir, fname))
        img_resized = resize(img)
        img_resized.save(os.path.join(args.save_to, fname))
