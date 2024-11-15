from pathlib import Path, PosixPath
from typing import List, Union
from PIL import JpegImagePlugin
import shutil
import cv2
import numpy as np


class Splitter(object):

    def __init__(self, overlap_pixels=0):
        self.overlap_pixels = overlap_pixels
        self.split_images = {}
        self.remove_later = {}
        self.preprocessed_img_folder = None

    def split(self, image_data: Union[List[JpegImagePlugin.JpegImageFile], List[PosixPath]], outputfolder: str = None, max_splitting_steps: int = 5, image_paths: List[str] = None):
        """
        Preprocesses the images. This includes:
        - Splitting large images into smaller ones
        :param image_data: List of PIL JpegImageFile objects or image paths.
        :param outputfolder: Outputfolder for the preprocessed images.
        :param max_splitting_steps: Maximum number of splitting steps.
        :param image_paths: List of image paths.
        """
        if image_data is None:
            raise ValueError('No image data set. Use set_image_paths() or use a dataset to set them.')
        if outputfolder is None:
            raise ValueError('No outputfolder was set. Use set_image_paths() to set them.')

        # Create a new folder for the preprocessed images
        self.preprocessed_img_folder = Path(outputfolder) / 'preprocessed'
        if self.preprocessed_img_folder.exists():
            shutil.rmtree(self.preprocessed_img_folder)
        self.preprocessed_img_folder.mkdir()

        # Iterate over all images and look up if they are too large
        new_image_paths = []
        self.split_images = {}
        self.remove_later = []
        split_index = 0
        images = []

        for i, img in enumerate(image_data):

            if img and isinstance(image_data[0], PosixPath):
                image_path = Path(img)
                img = cv2.imread(str(img))
            else:
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_path = Path(image_paths[i])

            # Split the image into smaller ones
            split_index += len(images)
            images = self.split_image(img, max_splitting_steps)

            ids = {}
            for j, image in enumerate(images):
                new_image_path = self.preprocessed_img_folder / (image_path.stem + f'_{j:03d}' + image_path.suffix)
                self.remove_later.append(new_image_path)
                cv2.imwrite(str(new_image_path), image)
                new_image_paths.append(new_image_path)
                ids[split_index + j] = image.shape[:2]

            self.split_images[image_path] = ids

        #new_image_paths.sort(key=lambda x: x.name)

        return new_image_paths

    def split_image(self, image, remaining_steps):
        """
        Splits the image if its too large, recursively checks if the new images are too large, too.
        :param image: image to split
        :return: list of images
        """
        h, w = image.shape[:2]
        if (h > 2000 or w > 2000) and remaining_steps > 0:
            remaining_steps -= 1
            images = self.split_image_into_four(image)
            new_images = []
            for image in images:
                new_images.extend(self.split_image(image, remaining_steps))
            return new_images
        else:
            return [image]

    def split_image_into_four(self, image):
        """
        Splits a large image into four smaller ones.
        :param image: large image
        :return: list of four smaller images
        """
        h, w = image.shape[:2]
        h2 = h // 2
        w2 = w // 2
        return [image[:h2 + self.overlap_pixels, :w2 + self.overlap_pixels], image[:h2 + self.overlap_pixels, w2 - self.overlap_pixels:],
                image[h2 - self.overlap_pixels:, :w2 + self.overlap_pixels], image[h2 - self.overlap_pixels:, w2 - self.overlap_pixels:]]

    def clean(self):
        # Delete the preprocessed images from the disk
        shutil.rmtree(self.preprocessed_img_folder)

    def clean_up(self, image_paths, keys_to_add):
        # Delete the preprocessed images from the disk and the image paths
        self.clean()
        new_image_paths = []
        for path in image_paths:
            if path not in self.remove_later:
                new_image_paths.append(path)
        for key in keys_to_add:
            new_image_paths.append(key)

        #new_image_paths.sort(key=lambda x: x.name)
        return new_image_paths

    def get_split_images(self):
        return self.split_images
