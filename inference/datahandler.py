import copy
import os
import tempfile
import re
import cv2
import shutil
import warnings
import numpy as np
from typing import Union
from pathlib import Path
from .splitter import Splitter
from .merger import Merger
from utils import draw_bboxes
from .annotationhandler import AnnotationHandler


class DataHandler(object):

    def __init__(self, args=None):

        self.args = self.__setup_args(args)
        self.outputfolder = self.__get_output_folder()
        self.input_path = Path(self.args.inputfolder) if self.args.inputfolder is not None else None
        self.dataset, self.images, self.indices = self.__setup_dataset_images()
        self.image_paths = self.__setup_image_paths()

        # For splitting images into multiple smaller ones
        self.splitter = Splitter(overlap_pixels=0)
        self.merger = Merger(overlap_pixels=0)
        self.preprocess()

    def __del__(self):
        if self.args.outputfolder is None and Path(self.outputfolder).exists():
            shutil.rmtree(self.outputfolder)

    def __len__(self):
        return len(self.image_paths) if not self.__source_is_huggingface() else len(self.dataset)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        return img_path.__str__()

    def set_image_paths(self, images_path: Union[str, list]):
        """
        Set the image paths. Can be either a folder or a list of image paths.
        :param images_path: list of image paths or path to the image folder
        """
        if isinstance(images_path, str):
            self.input_path = Path(images_path)
            self.image_paths = self.__setup_image_paths()
        elif isinstance(images_path, list):
            self.input_path = None
            self.image_paths = [Path(_) for _ in images_path]
            self.image_paths.sort(key=lambda x: x.name)

    def get_data(self):
        """
        Get the data from the DataHandler, ready for inference.
        :return: image paths
        """
        if self.__source_is_huggingface() and not self.args.split:
            return self.images
        else:
            return self.__get_image_paths_str()

    def preprocess(self):
        """
        Preprocess the images. This includes:
        - Splitting large images into smaller ones
        :param preprocess: bool to preprocess or not to preprocess
        :param max_splitting_steps: maximum number of splitting steps
        """
        if self.args.split:
            print("Option --split is set, big images will be split into smaller ones. This increases inference time alot.")
            if self.__source_is_huggingface():
                self.image_paths = self.splitter.split(self.images, self.outputfolder, self.args.max_splitting_steps, self.image_paths)
            else:
                self.image_paths = self.splitter.split(image_data=self.image_paths, outputfolder=self.outputfolder, max_splitting_steps=self.args.max_splitting_steps)

    def postprocess(self, inference_results):
        """
        Returns the results of the inference after merging split images and/or multiple models.
        :param inference_results: results from the inference
        :return: merged results: list of dictionaries with bounding boxes, labels and scores
        """
        results = copy.deepcopy(inference_results)
        if self.args.split:
            results, keys_to_add = self.merger.merge(results, self.splitter.get_split_images())
            self.image_paths = self.splitter.clean_up(self.image_paths, keys_to_add)

        merged_bboxes = self.merger.merge_bboxes(results)

        return merged_bboxes

    def show(self, results):
        """
        Show the results in a window with classes and boxxes and colors and stuff.
        :param results: results from the inference
        """
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)

        if self.__source_is_huggingface():
            images = self.images
        else:
            images = self.image_paths

        for i, img_src in enumerate(images):
            # if not results[i]:
            #     continue

            if not self.__source_is_huggingface():
                image = cv2.imread(str(img_src))
            else:
                image = np.array(img_src)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if self.args.print_results:
                print(f"Results for image {self.image_paths[i]}")
                for detection in results[i]:
                    score = detection["score"]
                    label = detection["label"]
                    box = [round(i, 2) for i in detection["box"].values()]
                    print(
                        f"Detected {label} with confidence "
                        f"{round(score, 3)} at location {box}"
                    )
                print("----------------------------------------\n")

            boxxed_img = draw_bboxes(image, [detection["box"] for detection in results[i]], [detection["label"] for detection in results[i]], True)

            cv2.imshow("Image", boxxed_img)
            cv2.waitKey(0)

    def get_annotations_as_json(self, results):
        """
        Get the annotations in json format
        :param results: results from the inference
        :return: json annotation
        """
        annotation_handler = AnnotationHandler(self.args, self.image_paths, self.images)
        return annotation_handler.get_json(results)

    def save_annotation(self, result):
        """
        Save the annotations from the inference in a json file.
        """
        if self.args.create_coco:
            annotation_handler = AnnotationHandler(self.args, self.image_paths, self.images)
            annotation_handler.save_results_in_json(result)

    def set_split(self, split):
        self.args.split = split

    def set_ann_path(self, ann_path):
        self.args.ann_path = ann_path

    def __setup_args(self, args):
        """
        Setup the arguments for the DataHandler. If no arguments are given, the default arguments are used.
        """
        default_args = {
            'inputfolder': None,
            'extensions': ['.jpg', '.png'],
            'outputfolder': None,
            'pattern': '',
            'include_subdirs': False,
            'create_coco': True,
            'ann_path': 'ann.json',
            'dataset': None,
            'split': False,
            'print_results': False,
            'score_thr': 0.4, # Used for filtering in the annotation handler
            'max_splitting_steps': 1
        }
        if args is None:
            from argparse import Namespace
            args = Namespace(**default_args)
        return args  # Store args for further use in the class

    def __get_output_folder(self):
        """
        Get the output folder for the preprocessed images. If no output folder is set, a temporary folder is created.
        :return: output folder
        """
        outpath = str(Path(self.args.outputfolder).expanduser()) if self.args.outputfolder is not None else None

        if outpath is None:
            outpath = tempfile.mkdtemp()
        else:
            if not Path(outpath).exists():
                Path(outpath).mkdir()

        return outpath

    def __setup_dataset_images(self):
        """
        Setup the dataset and images. If a dataset is set, the images are loaded from the dataset. If no dataset is set, nothing is loaded.
        :return: dataset, images, indices or None, None, None
        """
        if self.args.dataset is not None:
            from datasets import load_dataset
            print(f"Loading dataset {self.args.dataset_name} from {self.args.dataset}...")
            dataset = load_dataset(path=str(self.args.dataset), name=str(self.args.dataset_name), trust_remote_code=True)
            subset = dataset[self.args.subset]
            if hasattr(self.args, 'num_images') and self.args.num_images:
                random_indices = np.random.choice(len(subset), self.args.num_images, replace=False)
                images = subset.select(random_indices)['image']
                indices = random_indices
            else:
                images = subset['image']
                indices = [i for i in range(len(subset))]
        else:
            subset, images, indices = None, None, None
        return subset, images, indices

    def __setup_image_paths(self):
        """
        Setup the image paths. If no image paths are set, the image paths are loaded from the dataset or the input folder.
        :return: image paths
        """
        if not self.__source_is_huggingface() and not self.__source_is_disk():
            warnings.warn("Neither dataset or inputfolder is set. This DataHandler is empty and you need to set the image paths manually. Use set_image_paths() to set them.", stacklevel=2)
            return None
        if self.__source_is_huggingface():
            image_paths = [Path(_) for _ in self.dataset[self.indices]['filename']]
        elif self.__source_is_disk():
            image_paths = self.__get_image_paths()
            image_paths.sort(key=lambda x: x.name)

        return image_paths

    def __source_is_huggingface(self):
        return self.dataset is not None

    def __source_is_disk(self):
        return self.input_path is not None

    def __yield_images(self):
        """
        Yield images from the input folder. If include_subdirs is set, all images from all subdirectories are yielded.
        :return: image paths
        """
        path = self.input_path.expanduser()
        for root, dirs, files in os.walk(path):
            for file in files:
                basename = os.path.basename(file).lower()
                ext = os.path.splitext(basename)[-1].lower()
                basename = basename.replace(ext, '')
                if ext in self.args.extensions and bool(re.compile(self.args.pattern).match(basename)):
                    if self.args.include_subdirs:
                        yield Path(os.path.join(root, file))
                    elif root == str(path):
                        yield Path(os.path.join(root, file))

    def __get_image_paths(self):
        if not self.__source_is_huggingface():
            image_paths = list(self.__yield_images())
            if len(image_paths) == 0:
                raise ValueError("No images found in %s" % str(self.input_path))
        else:
            image_paths = [Path(_) for _ in self.dataset[self.indices]['filename']]

        return image_paths

    def __get_image_paths_str(self):
        return [path.__str__() for path in self.image_paths]

