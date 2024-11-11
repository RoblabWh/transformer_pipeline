import os
from typing import Optional, Union

from huggingface_hub import HfFileSystem
from joblib.externals.cloudpickle import instance

import utils
from utils import read_json, get_repository_root, get_models_json
from transformers import pipeline
from PIL import Image
import numpy as np
import albumentations as A
import warnings
import torch


class Inferencer(object):

    def __init__(self, checkpoints: Optional[Union[str, list]] = None, score_thr: Optional[float] = 0.5, batch_size: Optional[int] = None):
        """
        The InferenceEngine class is used to run the inference of a MMDetection Net on a folder of images.

        :param checkpoints: Checkpoint folder for Huggingface model
        """
        self.max_size = 1333
        self.batch_size = None if batch_size is None else int(batch_size)
        self.data = []
        self.transfrom_params = []
        self.basic_transform = A.Compose([
            A.LongestMaxSize(max_size=self.max_size),
            A.PadIfNeeded(min_height=self.max_size, min_width=self.max_size, border_mode=0, value=(128, 128, 128), position="top_left"),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

        if isinstance(checkpoints, str):
            checkpoints = [checkpoints]

        self.checkpoints = checkpoints

        if self.checkpoints is None:
            warnings.warn("No checkpoints provided. Please add a checkpoint to the inference engine.", UserWarning, stacklevel=2)
            self.models = []
            self.checkpoints = []
        else:
            self.models = [self.__load_model(checkpoint) for checkpoint in self.checkpoints]

        self.score_thr = score_thr

    def which_models_are_available(self):
        """
        Returns the models which are currently available.
        :return: List of models as strings
        """
        # Find base repository path
        repo_root = get_repository_root(os.path.abspath(__file__))
        transformers_path = get_models_json(repo_root)
        return read_json(transformers_path.joinpath("models.json"))['models']

    def __load_model(self, checkpoint):
        """
        Loads a model from the checkpoint file.
        :param checkpoint: Checkpoint file for Huggingface model
        :return:
        """
        print(f"Loading model from {checkpoint}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            fs = HfFileSystem()
            files = fs.ls(checkpoint)
            preprocessor_path = next((file for file in files if "preprocessor_config.json" in file), None)
            with fs.open(preprocessor_path, "r") as f:
                data = read_json(f)
                self.max_size = data["size"]["longest_edge"]
        except:
            abspath = os.path.abspath(checkpoint)
            # Check and load preprocessor config
            if not os.path.exists(os.path.join(abspath, "preprocessor_config.json")):
                import warnings
                warnings.warn(f"Preprocessor config not found in {abspath}. Using default size {self.max_size}x{self.max_size}. This might be wrong,", UserWarning, stacklevel=2)
            else:
                data = utils.read_json(os.path.join(abspath, "preprocessor_config.json"))
                self.max_size = data["size"]["longest_edge"]

        self.basic_transform = A.Compose([
            A.LongestMaxSize(max_size=self.max_size),
            A.PadIfNeeded(min_height=self.max_size, min_width=self.max_size, border_mode=0, value=(128, 128, 128),
                          position="top_left"),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"])
        )
        return pipeline(task="object-detection", model=checkpoint, device=device, batch_size = self.batch_size)

    def add_model(self, checkpoint):
        """
        Adds a model to the inference engine.
        :param network_folder:
        :param out_folder:
        :return:
        """
        self.checkpoints.append(checkpoint)
        self.models.append(self.__load_model(checkpoint))

    def remove_model(self, checkpoint: Union[str, int]):
        """
        Removes a model from the inference engine.
        :param checkpoint: Checkpoint file for Huggingface model or index of the model
        :return:
        """
        if isinstance(checkpoint, int):
            self.models.pop(checkpoint)
            self.checkpoints.pop(checkpoint)
        else:
            index = self.checkpoints.index(checkpoint)
            self.models.pop(index)
            self.checkpoints.pop(index)

    def batch_generator(self):
        for image in self.data:
            if isinstance(image, str):
                image_path = image
                image = np.array(Image.open(image_path).convert("RGB"))
            else:
                image = np.array(image.convert("RGB"))
            original_height, original_width = image.shape[:2]

            dummybbox = [[0, 0, original_width, original_height]]
            labels = ["dummy"]

            transform = self.basic_transform(image=image, bboxes=dummybbox, labels=labels)
            processed_image = transform["image"]
            transform_params = transform["bboxes"][0]
            self.transfrom_params.append(transform_params)

            yield Image.fromarray(processed_image)

    def adjust_bboxes_to_original(self, bboxes, transform_params, original_size):
        # Calculate scaling factors for width and height
        original_width, original_height = original_size
        width_scale = original_width / transform_params[2]
        height_scale = original_height / transform_params[3]

        x_offset = transform_params[0]
        y_offset = transform_params[1]

        adjusted_bboxes = []
        for bbox in bboxes:
            # Scale bounding box coordinates back to the original image size
            adjusted_bbox = {
                "box": {
                    "xmin": int((bbox["box"]["xmin"] - x_offset) * width_scale),
                    "ymin": int((bbox["box"]["ymin"] - y_offset) * height_scale),
                    "xmax": int((bbox["box"]["xmax"] - x_offset) * width_scale),
                    "ymax": int((bbox["box"]["ymax"] - y_offset) * height_scale),
                },
                "label": bbox["label"],
                "score": bbox["score"],
            }
            adjusted_bboxes.append(adjusted_bbox)

        return adjusted_bboxes

    def transform_bboxes(self, results):
        """
        Transform the bounding boxes from the inference to the original image size.
        :param results: results from the inference
        :return: transformed bounding boxes
        """
        for j, model_results in enumerate(results):
            for i, result in enumerate(model_results):
                # if self.__source_is_huggingface():
                #     original_width, original_height = self.dataset[self.indices[i]]['width'], self.dataset[self.indices[i]]['height']
                # else:
                if isinstance(self.data[i], str):
                    image = Image.open(self.data[i])
                else:
                    image = self.data[i]
                transform_params = self.transfrom_params[i]

                results[j][i] = self.adjust_bboxes_to_original(result, transform_params, (image.width, image.height))


    def __call__(self, data):
        """
        Runs the inference on a batch of images for all models loaded.
        :param batches:
        :return:
        """

        results = []
        self.data = data
        for model in self.models:
            print(f"Running inference on {len(data)} images with model {model.model.name_or_path}")
            with torch.no_grad():
                result = [output for output in model(self.batch_generator(), threshold=self.score_thr)]
                #result = model(data, threshold=self.score_thr)
            results.append(result)
        self.transform_bboxes(results)
        self.data, self.transfrom_params = [], []

        return results

