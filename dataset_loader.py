import json
from pathlib import Path
import os
import cv2
from datasets.features import Image
import numpy as np
from tqdm.asyncio import tqdm
from PIL import Image
import helper

with open("kaggle.json") as f:
            creds = json.load(f)
            os.environ["KAGGLE_USERNAME"] = creds["username"]
            os.environ["KAGGLE_KEY"] = creds["key"]

import kaggle
kaggle.api.authenticate()

class DataLoader:
    def __init__(self):
        self.IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        self.KAGGLE_DATASET = "maxwinkelmann/kite-tracking"   # <owner>/<dataset-slug>
        self.KAGGLE_CACHE_DIR = os.path.join(".cache", "kaggle", "kite-tracking")

    def load(self) -> list:
        print("Loading dataset...")

        if os.path.exists(self.KAGGLE_CACHE_DIR):
            print("Dataset already exists in cache. Loading from cache...")
        else:
            print("Dataset not found in cache. Downloading from Kaggle...")
            self.__download_dataset()

        img_files = [f for f in Path(self.KAGGLE_CACHE_DIR).rglob("*") if f.suffix.lower() in self.IMG_EXTENSIONS]
        if img_files:
            print(f"Found {len(img_files)} image files in the dataset.")
            return sorted([f for f in img_files], key=lambda x: (x.parent, x.name))
        
        return []

    def __download_dataset(self):
        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(self.KAGGLE_DATASET, 
                                          path=self.KAGGLE_CACHE_DIR, 
                                          unzip=True, 
                                          quiet=False)
        

class preprocessor:
    def __init__(self):
        print("Preprocessing dataset...")

    # This method loads the images, converts them to grayscale, 
    # and resizes them to a maximum dimension of 320 pixels 
    # while maintaining the aspect ratio. It returns a list of '
    # tuples containing pairs of consecutive grayscale frames 
    # along with their corresponding RGB frames.
    def convertToGrayScaleAndResize(self, images, max_pairs: int = 100) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        print(f"Loading frame pairs from dataset (max {max_pairs} pairs)...")
        img_files = images
        if len(img_files) < 2:
            print("Not enough images to form frame pairs.")
            return []
        
        limit     = (max_pairs + 1) if max_pairs else len(img_files)
        img_files = img_files[:limit]
        print(f"  → Found {len(img_files)} image file(s) building frame pairs …")

        gray_frames : list[np.ndarray] = []
        rgb_frames : list[np.ndarray] = []

        for path in tqdm(img_files, desc="Loading images", unit="img"):
            try:
                image = Image.open(path).convert("RGB")
                rgbImage = np.array(image)
                grayImage = helper.resize_gray(np.array(image.convert("L")))
                rgbImage = cv2.resize(rgbImage, (grayImage.shape[1], grayImage.shape[0]), interpolation=cv2.INTER_AREA)
                gray_frames.append(grayImage)
                rgb_frames.append(rgbImage)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue

        if len(gray_frames) < 2:
            print("Not enough valid images to form frame pairs.")
            return []
        
        pairs = [(gray_frames[i], gray_frames[i + 1], rgb_frames[i]) for i in range(len(gray_frames) - 1)]
        print(f"  → {len(pairs)} consecutive frame-pair(s) prepared.")
        return pairs
