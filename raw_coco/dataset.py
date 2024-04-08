import os
import cv2
import json
from tqdm import tqdm
from pathlib import Path
import random
import copy
import numpy as np
import shutil
import utils as u


class Dataset:
    """
    This class represents a dataset. It is constructed using a folder path for the image roots and
    an annotation file path. The annotation file must be in the COCO format.
    TODO Support for other datasets not needed currently
    """

    def __init__(self, img_folder, ann_file):
        """
        :param img_folder: Path to the folder containing the images
        :param ann_file: Path to the annotation file
        """
        self.image_paths = None
        self.img_folder = Path(img_folder)
        self.ann_file = Path(ann_file)

        # read the annotation json file
        if os.path.isfile(ann_file):
            self.annotations = u.read_json(ann_file)
        else:
            self.annotations = u.empty_coco_ann()
            # save json file
            with open(ann_file, 'w') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=4)

        # create a dictionary mapping image_id and file_name to a list of images
        self.images = {}
        self.update_images()

        # create a dictionary mapping image_id to a list of bboxes
        self.bboxes = {}
        self.update_bboxes()

        self.check_for_missing()

    def get_number_of_images(self):
        return len(self.annotations['images'])

    def get_images_without_annotations(self):
        """
        Returns a list of images without annotations.
        """

        images_without_annotations = []
        for image in self.annotations['images']:
            if image['id'] not in [ann['image_id'] for ann in self.annotations['annotations']]:
                images_without_annotations.append(image)

        return images_without_annotations

    def split_dataset(self, split=(80, 10, 10), split_type='random', perc_exclude=90):
        """
        Splits the dataset into a training, validation and test set.
        :param split: The percentage of the dataset used for training, validation and test.
        :param split_type: The type of split. 'random' splits the dataset randomly. 'sequential' splits the dataset in order
        sequentially.
        :param perc_exclude: The percentage of images without annotations to exclude from the dataset set.
        """
        if split_type == 'random':

            # images without annotations
            images_without_annotations = self.get_images_without_annotations()

            # Calculate the number of images to exclude
            num_exlude = int(np.floor(len(images_without_annotations) * perc_exclude / 100))

            exlude_images = random.sample(images_without_annotations, num_exlude)

            # Remove excluded images from the dataset
            self.annotations['images'] = [img for img in self.annotations['images'] if img not in exlude_images]

            # split the dataset
            n = len(self.annotations['images'])
            train_split = int(np.floor(n * split[0] / 100))
            val_split = int(np.floor(n * split[1] / 100))

            indices = list(range(n))
            random.shuffle(indices)

            train_indices = indices[:train_split]
            val_indices = indices[train_split:train_split + val_split]
            test_indices = indices[train_split + val_split:]

            train_images = [self.annotations['images'][i] for i in train_indices]
            val_images = [self.annotations['images'][i] for i in val_indices]
            test_images = [self.annotations['images'][i] for i in test_indices]

            train_anns = [ann for ann in self.annotations['annotations'] if
                          ann['image_id'] in [img['id'] for img in train_images]]
            val_anns = [ann for ann in self.annotations['annotations'] if
                        ann['image_id'] in [img['id'] for img in val_images]]
            test_anns = [ann for ann in self.annotations['annotations'] if
                         ann['image_id'] in [img['id'] for img in test_images]]

            train_ann = {'images': train_images, 'categories': self.annotations['categories'],
                         'annotations': train_anns}
            val_ann = {'images': val_images, 'categories': self.annotations['categories'], 'annotations': val_anns}
            test_ann = {'images': test_images, 'categories': self.annotations['categories'], 'annotations': test_anns}

            with open(self.ann_file.parent / 'train.json', 'w') as f:
                json.dump(train_ann, f, ensure_ascii=False, indent=4)
            with open(self.ann_file.parent / 'val.json', 'w') as f:
                json.dump(val_ann, f, ensure_ascii=False, indent=4)
            with open(self.ann_file.parent / 'test.json', 'w') as f:
                json.dump(test_ann, f, ensure_ascii=False, indent=4)

        elif split_type == 'sequential':
            raise NotImplementedError('Sequential split not implemented yet.')
        else:
            raise ValueError('Split type must be either \'random\' or \'sequential\'.')

    def check_for_missing(self):
        """
        Checks if there are any missing images or annotations and deletes them from the annotation file.
        """
        missing_files = self.get_missing_files()
        missing_annotations = self.get_missing_annotations()
        # prompt user to delete missing files
        if len(missing_files) > 0:
            print('Missing files: {}'.format(missing_files))
            print('Do you want to delete them from the annotation file?')
            if input('y/n: ') == 'y':
                #raise NotImplementedError('No you do not. It is not tested enough. Coming soonTM')
                u.delete_from_json_images(self.ann_file, missing_files)
                print('Deleted missing files from annotation file.')
        if len(missing_annotations) > 0:
            print('Missing annotations: {}'.format(missing_annotations))
            print('Do you want to delete them from the annotation file?')
            if input('y/n: ') == 'y':
                print('Removing missing annotations from annotation file...')
                u.delete_from_json_annotation(self.ann_file, missing_annotations)
        self.annotations = u.read_json(self.ann_file)
        self.update_images()
        self.update_bboxes()

    def get_missing_files(self):
        """
        Checks if the images in the annotation file are present in the image folder.
        :return:
        """
        missing_files = []
        for img in self.annotations['images']:
            # if file not exists
            if not os.path.isfile(os.path.join(self.img_folder, img['file_name'])):
                missing_files.append(img['file_name'])
        return missing_files

    def get_missing_annotations(self):
        """
        Checks if the images in the annotations are present in the annotation file.
        :return:
        """
        missing_annotations = []
        for img in self.bboxes.keys():
            if img not in self.images:
                missing_annotations.append(img)
        return missing_annotations

    def add_dataset(self, dataset, copy_images=True):
        """
        Adds another dataset to this dataset. The images and annotations of the other dataset are
        added to this dataset. The image ids are updated to be unique. The image names are changes to
        the new image ids. The annotation file is updated accordingly.
        :param dataset: The dataset to be added.
        :param copy_images: If True, the images of the other dataset are copied to this dataset.
        """

        # The Image IDs and ANN IDs are resseted to be unique
        dataset.reset_img_ids()
        dataset.reset_ann_ids()
        # Get the last image ID of this dataset, doesn't have to be ordered
        last_img_id = 0
        last_ann_id = 0
        if self.images:
            last_img_id = max(self.images.keys())
        if self.annotations['annotations']:
            last_ann_id = max([ann['id'] for ann in self.annotations['annotations']])

        # save old annotations
        root_ann_path = Path(
            os.path.join(self.ann_file.parent, str(self.ann_file.stem) + "_root" + self.ann_file.suffix))
        with open(root_ann_path, 'w') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=4)

        new_ann_path = Path(
            os.path.join(self.ann_file.parent, str(dataset.ann_file.stem) + "_added" + dataset.ann_file.suffix))
        new_ann = u.empty_coco_ann()

        # update the annotation file
        for img in tqdm(dataset.annotations['images'], desc='Adding images to dataset...'):
            old_file_name = img['file_name']
            img['id'] += last_img_id
            img['file_name'] = str(img['id']).zfill(6) + '.' + img['file_name'].split('.')[-1].lower()
            self.annotations['images'].append(img)
            new_ann['images'].append(img)
            # Copies the images to the correct folder if move_images is True
            if copy_images:
                img_path = os.path.join(dataset.img_folder, old_file_name)
                new_img_path = os.path.join(self.img_folder, img['file_name'])
                shutil.copy(img_path, new_img_path)
        for ann in dataset.annotations['annotations']:
            ann['id'] += last_ann_id
            ann['image_id'] += last_img_id
            self.annotations['annotations'].append(ann)
            new_ann['annotations'].append(ann)

        # save the new annotations
        with open(new_ann_path, 'w') as f:
            json.dump(new_ann, f, ensure_ascii=False, indent=4)

        # update the categories
        self.reset_img_ids()
        self.reset_ann_ids()
        self.update_json()
        self.rename_images()

    def reset_img_ids(self):
        """
        Resets the image IDs. The ids are set to the index of the image in the images list.
        Also checks the image IDs in the annotations and updates them accordingly.
        """
        id_mapping = {}
        for i, img in enumerate(self.annotations['images']):
            old_id = img['id']
            new_id = i + 1
            id_mapping[old_id] = new_id
            img['id'] = new_id

        for ann in self.annotations['annotations']:
            ann['image_id'] = id_mapping[ann['image_id']]

        self.update_images()
        self.update_bboxes()

    def reset_ann_ids(self):
        """
        Resets the annotation IDs. The IDs are set to the index of the annotation in the annotations list.
        """
        for i, ann in enumerate(self.annotations['annotations']):
            ann['id'] = i + 1

        self.update_images()
        self.update_bboxes()

    def rename_images(self):
        """
        Renames all images in the dataset. The new names are the image IDs.
        """
        for img_id in reversed(self.images.keys()):
            img_name = self.images[img_id]
            ext = img_name.split('.')[-1].lower()
            img_path = os.path.join(self.img_folder, img_name)
            new_img_path = os.path.join(self.img_folder, str(img_id).zfill(6) + '.' + ext)
            os.rename(img_path, new_img_path)
            self.images[img_id] = str(img_id).zfill(6) + '.' + ext
            # update the annotation file
            for img in self.annotations['images']:
                if img['file_name'] == img_name:
                    img['file_name'] = str(img_id).zfill(6) + '.' + ext
        self.update_json()
        self.update_images()
        self.update_bboxes()

    def copy_images(self, path):
        """
        Copies all images to the given path.
        :param path:
        :return:
        """
        for img_id in self.images.keys():
            img_name = self.images[img_id]
            img_path = os.path.join(self.img_folder, img_name)
            new_img_path = os.path.join(path, img_name)
            shutil.copyfile(img_path, new_img_path)

    def update_json(self):
        """
        Updates the json file.
        """
        with open(self.ann_file, 'w') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=4)

    def update_images(self):
        """
        Updates the image dictionary.
        """
        self.images = {}
        for img in self.annotations['images']:
            self.images[img['id']] = img['file_name']

    def update_bboxes(self):
        """
        Updates the bboxes dictionary.
        """
        self.bboxes = {}
        for ann in self.annotations['annotations']:
            if ann['image_id'] not in self.bboxes:
                self.bboxes[ann['image_id']] = {}
            if ann['category_id'] not in self.bboxes[ann['image_id']]:
                self.bboxes[ann['image_id']][ann['category_id']] = []
            self.bboxes[ann['image_id']][ann['category_id']].append(ann['bbox'])

    def display_images(self, img_ids=None, category_ids=None, bboxes=True, show=True, save=False, save_path=None,
                       shuffle=False, compare_dataset=None, show_classes=True):
        """
        Displays the images with the given ids.
        :param img_ids: The ids of the images to display
        :param category_ids: The ids of the categories to display
        :param bboxes: Whether to display the bounding boxes
        :param show: Whether to show the images
        :param save: Whether to save the images
        :param save_path: The path to save the images to
        """
        if save_path is not None:
            # Create a  folder for the images
            if not Path(save_path).exists():
                Path(save_path).mkdir()
        if img_ids is None:
            img_ids = self.images.keys()
        if show:
            # show image at the same position on the screen
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Image', 0, 0)
        if shuffle:
            img_ids = list(img_ids)
            random.shuffle(img_ids)
        for img_id in img_ids:
            img = self.get_image(img_id)
            if img is None:
                continue
            if bboxes and img_id in self.bboxes:
                for category_id in self.bboxes[img_id].keys():
                    bboxes_ = self.bboxes[img_id][category_id]
                    img = u.draw_bboxes(img, bboxes_, [category_id - 1 for _ in range(len(bboxes_))], show_classes)
            if show:
                if compare_dataset is not None:
                    img_name = self.images[img_id]
                    img2 = self.get_image(img_id)
                    compare_img_id = list(compare_dataset.images.keys())[list(compare_dataset.images.values()).index(img_name)]
                    if bboxes and compare_img_id in compare_dataset.bboxes:
                        for category_id in compare_dataset.bboxes[compare_img_id].keys():
                            bboxes_ = compare_dataset.bboxes[compare_img_id][category_id]
                            img2 = u.draw_bboxes(img2, bboxes_, [category_id - 1 for _ in range(len(bboxes_))], show_classes)
                # img = cv2.resize(img, (1920, 1080))
                if compare_dataset is None:
                    cv2.imshow('Image', img)
                else:
                    text_position = (int(0), int(10))
                    cv2.putText(img, "Root Dataset", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(img2, "Compare Dataset", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.imshow('Image', np.concatenate((img, img2), axis=1))
                cv2.setWindowTitle('Image', self.images[img_id])
                key = cv2.waitKey(0)
                if key == ord('q'):
                    return
            if save:
                if save_path is not None:
                    # gt = str(self.images[img_id]).replace(str(Path(self.images[img_id]).stem), str(str(Path(self.images[img_id]).stem)+"_gt"))
                    # cv2.imwrite(os.path.join(save_path, gt), img)
                    cv2.imwrite(os.path.join(save_path, self.images[img_id]), img)

    def update_annotations(self, coco_data, new_images, intersection_threshold=0.15):
        """
        A function that updates the annotations of a COCO dataset normally called after the dataset was splitted.
        :param coco_data: loaded JSON file of a COCO dataset in a dictionary
        :param new_images: list of images to add that where created by splitting the original images
                           [("00000_1.jpg", x1, y1, width1, height1), ...]
        :return: new COCO dataset
        """
        new_annotations = []
        new_images_data = []
        starting_image_id = len(coco_data["images"])
        starting_annotation_id = len(coco_data["annotations"])

        for new_image_name, x, y, width, height in new_images:
            # Find the original image ID by its file name
            original_image_id = None
            for i, image in enumerate(coco_data["images"]):
                filename = ""
                for s in Path(new_image_name).stem.split("_")[:-1]:
                    filename += s + "_"
                filename = filename[:-1]
                if Path(coco_data["images"][i]["file_name"]).stem == filename:
                    original_image_id = image["id"]
                    break

            if original_image_id is None:
                print(f"Original image not found for {new_image_name}")
                continue

            # Create new image data entry
            new_image_data = copy.deepcopy(image)
            new_image_data["file_name"] = new_image_name
            new_image_data["id"] = starting_image_id + len(new_images_data) + 1
            new_image_data["width"] = width
            new_image_data["height"] = height
            new_images_data.append(new_image_data)

            # Update the annotations for the new image
            for annotation in coco_data["annotations"]:
                if annotation["image_id"] == original_image_id:
                    new_annotation = copy.deepcopy(annotation)
                    new_annotation["id"] = starting_annotation_id + len(new_annotations) + 1
                    new_annotation["image_id"] = new_image_data["id"]

                    # Update the bounding box coordinates
                    old_x, old_y, old_w, old_h = new_annotation["bbox"]
                    new_x = max(0, old_x - x)
                    new_y = max(0, old_y - y)
                    new_w = min(width - new_x, min(old_w, old_x + old_w - x))
                    new_h = min(height - new_y, min(old_h, old_y + old_h - y))

                    # Check if the bounding box intersects the new image by more than the threshold
                    # That's not exactly using old_x and old_y here, but shouldn't be a broblem
                    intersection_ratio = u.calc_iou([old_x, old_y, new_w, new_h], new_annotation['bbox'])

                    if intersection_ratio >= intersection_threshold:
                        new_annotation["bbox"] = [new_x, new_y, new_w, new_h]
                        new_annotations.append(new_annotation)

        # Update the COCO data with the new image data and annotations
        coco_data["images"] = new_images_data
        coco_data["annotations"] = new_annotations

        return coco_data

    def generate_small_dataset(self, new_data_folder):
        """
        This method takes this dataset, cuts each image into smaller ones, if necessary,
        and saves the smaller images to a new folder. It also updates the annotations
        associated with each image to reflect the changes.

        If an image doesn't need to be split, it's just copied to the new directory with a new name.

        The image annotations are then updated based on these new images and saved to a new JSON file.

        Parameters:
            new_data_folder (str): The path to the directory where the new images will be saved.

        The method works as follows:
        1. Creates a new folder for the preprocessed images.
        2. Iterates over all images in the original dataset.
        3. If an image is too large, it's split into smaller ones. Each smaller image is saved with a unique name.
        4. The image's new name, coordinates, width, and height are recorded.
        5. The annotations associated with the original image are updated to correspond to the smaller images.
        6. If an image doesn't need to be split, it's copied to the new directory with a new name.
        7. The new image's name, coordinates, width, and height are recorded.
        8. The annotations associated with the original image are updated to correspond to the new image.
        9. Once all images have been processed, the updated annotations are saved to a new JSON file.
        """

        # Create a new folder for the preprocessed images
        new_img_folder = Path(new_data_folder) / 'images'
        if new_img_folder.exists():
            shutil.rmtree(new_img_folder)
        os.makedirs(new_img_folder)

        # Iterate over all images and look up if they are too large
        new_image_paths = []
        split_images = {}
        splitted_images = 0
        images = []
        new_images = []
        for i, oimage in enumerate(tqdm(self.images.values(), desc="Images")):
            image_path = Path(self.img_folder) / oimage
            if not image_path.exists():
                continue
            img = cv2.imread(str(image_path))

            # Split the image into smaller ones
            splitted_images += len(images)
            image_coordinates = self.split_image(img)
            if len(image_coordinates) > 1:
                ids = {}
                for j, (image, x, y) in enumerate(image_coordinates):
                    new_image_path = new_img_folder / (image_path.stem + f'_{j}' + image_path.suffix)
                    new_image = (image_path.stem + f'_{j}' + image_path.suffix, x, y, image.shape[1], image.shape[0])
                    new_images.append(new_image)
                    # self.remove_later.append(new_image_path)
                    cv2.imwrite(str(new_image_path), image)
                    new_image_paths.append(new_image_path)
                    ids[splitted_images + j] = (image.shape[:2], x, y)

                split_images[image_path] = ids
            else:
                new_image_paths.append(image_path)
                new_image = (image_path.stem + f'_{0}' + image_path.suffix, 0, 0, img.shape[1], img.shape[0])
                new_images.append(new_image)
                cv2.imwrite(str(new_img_folder / (image_path.stem + f'_{0}' + image_path.suffix)), img)

        new_coco_data = self.update_annotations(new_images)
        # Save the modified annotations to a new JSON file
        with open(os.path.join(new_data_folder, "modified_annotations.json"), "w") as f:
            json.dump(new_coco_data, f, indent=2)

        self.image_paths = new_image_paths

    def split_image(self, image):
        """
        Splits the image if it's too large, recursively checks if the new images are too large, too.
        :param image: image to split
        :return: list of images along with their coordinates in the original image
        """
        h, w = image.shape[:2]
        if h > 1200 or w > 1200:
            image_coordinates = self.split_image_into_four(image)
            new_images = []
            for img, x, y in image_coordinates:
                for sub_img, sub_x, sub_y in self.split_image(img):
                    new_images.append((sub_img, x + sub_x, y + sub_y))
            return new_images
        else:
            return [(image, 0, 0)]

    def split_image_into_four(self, image):
        """
        Splits a large image into four smaller ones.
        :param image: large image
        :return: list of four smaller images along with their coordinates in the original image
        """
        h, w = image.shape[:2]
        overlap = 0
        h2 = h // 2
        w2 = w // 2

        return [
            (image[:h2 + overlap, :w2 + overlap], 0, 0),
            (image[:h2 + overlap, w2 - overlap:], w2 - overlap, 0),
            (image[h2 - overlap:, :w2 + overlap], 0, h2 - overlap),
            (image[h2 - overlap:, w2 - overlap:], w2 - overlap, h2 - overlap),
        ]


    def get_image(self, img_id):
        """
        Returns the image with the given id.
        :param img_id: The id of the image
        :return: The image
        """
        return cv2.imread(os.path.join(self.img_folder, self.images[img_id]))

    def get_bboxes(self, img_id):
        """
        Returns the bounding boxes for the given image id.
        :param img_id: The id of the image
        :return: A dictionary mapping category ids to a list of bounding boxes
        """
        return self.bboxes[img_id]

    def get_image_ids(self):
        """
        Returns a list of all image ids.
        :return: A list of all image ids
        """
        return list(self.images.keys())

    def get_image_names(self):
        """
        Returns a list of all image names.
        :return: A list of all image names
        """
        return list(self.images.values())

    def get_image_path(self, img_id):
        """
        Returns the path to the image with the given id.
        :param img_id: The id of the image
        :return: The path to the image
        """
        return self.img_folder + self.images[img_id]

    def get_image_name(self, img_id):
        """
        Returns the name of the image with the given id.
        :param img_id: The id of the image
        :return: The name of the image
        """
        return self.images[img_id]

    def get_image_id(self, img_name):
        """
        Returns the id of the image with the given name.
        :param img_name: The name of the image
        :return: The id of the image
        """
        for img_id, name in self.images.items():
            if name == img_name:
                return img_id
        return None
