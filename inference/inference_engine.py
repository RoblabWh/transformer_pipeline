from typing import Optional, Union
from transformers import pipeline
from utils import read_json, get_git_root, get_models_json
from pathlib import Path
import warnings
import torch

class Inferencer(object):

    def __init__(self, checkpoints: Optional[Union[str, list]] = None, score_thr: Optional[float] = 0.5):
        """
        The InferenceEngine class is used to run the inference of a MMDetection Net on a folder of images.

        :param checkpoints: Checkpoint file for Huggingface model
        """

        if isinstance(checkpoints, str):
            checkpoints = [checkpoints]

        self.checkpoints = checkpoints

        if self.checkpoints is None:
            warnings.warn("No checkpoints provided. Please add a checkpoint to the inference engine.", UserWarning, stacklevel=2)
            self.models = []
            self.checkpoints = []
        else:
            self.models = [self.__load_model(checkpoint) for checkpoint in checkpoints]

        self.score_thr = score_thr

    def which_models_are_available(self):
        """
        Returns the models which are currently available.
        :return: List of models as strings
        """
        # Find base repository path
        git_root = get_git_root(__file__)
        transformers_path = get_models_json(git_root)
        return read_json(transformers_path.joinpath("models.json"))['models']

    def __load_model(self, checkpoint):
        """
        Loads a model from the checkpoint file.
        :param checkpoint: Checkpoint file for Huggingface model
        :return:
        """
        print(f"Loading model from {checkpoint}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return pipeline("object-detection", model=checkpoint, device=device)

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

    def __call__(self, data):
        """
        Runs the inference on a batch of images for all models loaded.
        :param batches:
        :return:
        """

        results = []
        for model in self.models:
            print(f"Running inference on {len(data)} images with model {model.model.name_or_path}")
            result = model(data, threshold=self.score_thr)
            results.append(result)

        return results

