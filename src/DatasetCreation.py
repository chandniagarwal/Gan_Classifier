import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Optional
import cv2
from tqdm import tqdm

class DatasetCreation:
    def __init__(self, x: list, label: list, label_type: str, image_processing: Optional = None):
        label_types = {
            "binary": self.__binary_transformer,
            "one_hot": self.__one_hot_transformer
        }
        print("_________Loading Images__________")
        self.labels = label_types[label_type](label)
        self.images = self.__load_image(x, image_processing)

    def __load_image(self, image_loc, image_processing) -> np.array:
        list_images = []

        for img_path in tqdm(image_loc):
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))
            image = image_processing(image) if image_processing else self.__normalise_image(image)
            arr = np.array(image)
            list_images.append(arr)
        list_images = np.array(list_images)
        return list_images

    def __normalise_image(self, image: np.array) -> np.array:
        MEAN = 255 * np.array([0.485, 0.456, 0.406])
        STD = 255 * np.array([0.229, 0.224, 0.225])
        x = image
        x = x.transpose(-1, 0, 1)
        x = (x - MEAN[:, None, None]) / STD[:, None, None]
        x = x.transpose(1, -1, 0)
        return x

    def __one_hot_transformer(self, label) -> np.array:
        # One hot encoding
        label = np.array(label).reshape(-1, 1)
        one_hot_encoder = OneHotEncoder(
            sparse_output=False,
        )
        label = one_hot_encoder.fit_transform(label)
        return label

    def __binary_transformer(self, label) -> np.array:
        # binary encoding
        label = np.array(label)
        label_encoder = LabelEncoder()

        label = label_encoder.fit_transform(label)
        return label