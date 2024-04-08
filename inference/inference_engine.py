import os
from typing import Union
from transformers import pipeline

class Inferencer(object):

    def __init__(self, checkpoints: Union[str, list], score_thr):
        """
        The InferenceEngine class is used to run the inference of a MMDetection Net on a folder of images.

        :param checkpoints: Checkpoint file for Huggingface model
        """

        if isinstance(checkpoints, str):
            checkpoints = [checkpoints]

        self.checkpoints = checkpoints
        self.models = [self.load_model(checkpoint) for checkpoint in checkpoints]
        self.score_thr = score_thr

    def load_model(self, checkpoint):
        """
        Loads a model from the checkpoint file.
        :param checkpoint: Checkpoint file for Huggingface model
        :return:
        """
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint file {checkpoint} not found.")
        else:
            print(f"Loading model from {checkpoint}")
        return pipeline("object-detection", model=checkpoint)

    def add_model(self, checkpoint):
        """
        Adds a model to the inference engine.
        :param network_folder:
        :param out_folder:
        :return:
        """
        self.checkpoints.append(checkpoint)
        self.models.append(self.load_model(checkpoint))

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

