import copy
import os
import sys
import re
import cv2
import numpy as np
from pathlib import Path
from splitter import Splitter
from merger import Merger
from utils import draw_bboxes


class DataHandler(object):

    def __init__(self, args):

        self.args = self.setup_args(args)
        self.img_prefix = self.get_img_prefix()
        self.input_path = Path(args.inputfolder) if args.inputfolder is not None else None
        self.dataset, self.images, self.indices = self.setup_dataset_images()
        self.image_paths = self.setup_image_paths()

        # For splitting images into multiple smaller ones
        self.splitter = Splitter(overlap_pixels=0)
        self.merger = Merger(overlap_pixels=0)
        if self.args.split:
            print("Option --split is set, big images will be split into smaller ones. This increases inference time alot.")
            if self.args.max_splitting_steps:
                self.preprocess(max_splitting_steps=args.max_splitting_steps)
            else:
                self.preprocess()

    def setup_args(self, args):
        default_args = {
            'inputfolder': None,
            'extensions': ['.jpg', '.png'],
            'outputfolder': '/tmpdetoutput',
            'pattern': '',
            'include_subdirs': False,
            'batch_size': 1,
            'dataset': None,
            'split': False
        }
        if args is None:
            from argparse import Namespace
            args = Namespace(**default_args)
        return args  # Store args for further use in the class

    def get_img_prefix(self):
        prefix = str(Path(self.args.outputfolder).expanduser()) if self.args.outputfolder is not None else None

        if not Path(prefix).exists():
            Path(prefix).mkdir()

        return prefix

    def setup_dataset_images(self):
        if self.args.dataset is not None:
            from datasets import load_dataset
            dataset = load_dataset(path=str(self.args.dataset), name=str(self.args.dataset_name))
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

    def setup_image_paths(self):
        if not self.source_is_huggingface():
            image_paths = self.get_image_paths()
        else:
            image_paths = [Path(_) for _ in self.dataset[self.indices]['filepath']]
        image_paths.sort(key=lambda x: x.name)
        return image_paths

    def source_is_huggingface(self):
        return self.dataset is not None

    def __len__(self):
        return len(self.image_paths) if not self.source_is_huggingface() else len(self.dataset)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        return img_path.__str__()

    def yield_images(self):
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

    def set_image_paths(self, image_paths):
        """
        Set the image paths and batches
        :param image_paths: list of image paths
        """
        self.image_paths = image_paths
        self.image_paths.sort(key=lambda x: x.name)
        self.img_prefix = str(Path(image_paths[0]).expanduser().parent)

    def get_image_paths(self):
        if self.dataset is None:
            image_paths = list(self.yield_images())
            if len(image_paths) == 0:
                print("No images found in %s" % str(self.input_path))
                sys.exit(1)

        return image_paths

    def get_image_paths_str(self):
        return [path.__str__() for path in self.image_paths]

    def get_data(self):
        if self.source_is_huggingface() and not self.args.split:
            return self.images
        else:
            return self.get_image_paths_str()

    def preprocess(self, max_splitting_steps=8):
        """
        Preprocess the images. This includes:
        - Splitting large images into smaller ones
        :param preprocess: bool to preprocess or not to preprocess
        :param max_splitting_steps: maximum number of splitting steps
        """
        if self.source_is_huggingface():
            self.image_paths = self.splitter.split(self.images, self.img_prefix, max_splitting_steps, self.image_paths)
        else:
            self.image_paths = self.splitter.split(image_data=self.image_paths, outputfolder=self.img_prefix, max_splitting_steps=max_splitting_steps)

    def postprocess(self, inference_results):
        """
        Postprocess the inference results. This includes:
        - Merging the results of the split images into the original image
        - Deleting the preprocessed images
        :param inference_results: results from the inference
        :return: merged results
        """
        merged_results, keys_to_add = self.merger.merge(inference_results, self.splitter.get_split_images())
        self.image_paths = self.splitter.clean_up(self.image_paths, keys_to_add)

        return merged_results

    def get_results(self, inference_results):
        """
        Returns the results of the inference after merging split images and/or multiple models.
        :param inference_results: results from the inference
        :return: merged results: list of dictionaries with bounding boxes, labels and scores
        """
        results = copy.deepcopy(inference_results)
        if self.args.split:
            results = self.postprocess(results)

        merged_bboxes = self.merger.merge_bboxes(results)

        return merged_bboxes

    def show(self, results):
        """
        Show the results in a window with classes and boxxes and colors and stuff.
        :param results: results from the inference
        """
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)

        if self.source_is_huggingface():
            images = self.images
        else:
            images = self.image_paths

        for i, img_src in enumerate(images):
            # if not results[i]:
            #     continue

            if not self.source_is_huggingface():
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



