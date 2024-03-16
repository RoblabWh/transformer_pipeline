"""FireDetDataset LOADER"""
import json
import os
from pathlib import Path

import datasets
#from https://huggingface.co/datasets/HuggingFaceM4/COCO/blob/main/COCO.py

_CITATION = """

"""

_DESCRIPTION = """
"""

_HOMEPAGE = ""

_LICENSE = ""

_IMAGES_URLS = {
    # TODO Maybe URLS to different versions of the dataset
}
_ANNOTATION_FILES_URL = {
    # TODO URLS to different annotation files of the dataset
}

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "filepath": datasets.Value("string"),
        "filename": datasets.Value("string"),
        "image_id": datasets.Value("int32"),
        "height": datasets.Value("int32"),
        "width": datasets.Value("int32"),
        "objects": datasets.Sequence({
            "area": datasets.Value("int32"),
            "bbox": datasets.Sequence(datasets.Value("float32")),
            "category": datasets.ClassLabel(num_classes=3, names=['fire', 'vehicle', 'human']),
            "id": datasets.Value("int32"),
        }),
    }
)


class FireDetDataset(datasets.GeneratorBasedBuilder):
    """FireDetDataset"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="GOLD",
            version=VERSION,
            description="Manually created dataset for detecting fires, vehicles and people from above."
        ),
        datasets.BuilderConfig(
            name="SILVER",
            version=VERSION,
            description="Automatically created dataset for detecting fires, vehicles and people from above.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "GOLD"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO make this downloadable from huggingface with dl_manager
        #
        train_ann = "/home/nex/Bilder/Datasets/golddataset/annotation/split/train.json"
        val_ann = "/home/nex/Bilder/Datasets/golddataset/annotation/split/val.json"
        test_ann = "/home/nex/Bilder/Datasets/golddataset/annotation/split/test.json"

        img_files_list = list(Path("/home/nex/Bilder/Datasets/golddataset/images").rglob("*.jpg"))
        image_folders = "/home/nex/Bilder/Datasets/golddataset/images"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": train_ann,
                    "image_folders": image_folders,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": val_ann,
                    "image_folders": image_folders,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file": test_ann,
                    "image_folders": image_folders,
                },
            ),
        ]

    def _generate_examples(self, annotation_file, image_folders):
        with open(annotation_file, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)
        # Initialize an empty dictionary to store the hashmap
        image_id_map = {}

        # Iterate through each entry in the COCO data
        for entry in annotations["annotations"]:
            # Get the image_id of the current entry
            image_id = entry['image_id']

            # If the image_id is not already a key in the hashmap,
            # initialize an empty list for this key
            if image_id not in image_id_map:
                image_id_map[image_id] = []

            # Append the current entry to the list associated with this image_id
            image_id_map[image_id].append(entry)
        for image_metadata in annotations["images"]:

            img_id = image_metadata['id']
            objects = image_id_map[img_id]
            # Create 'objects' key in the record
            objects_ = {'area': [], 'bbox': [], 'category': [], 'id': []}
            for object in objects:
                objects_['area'].append(object['area'])
                objects_['bbox'].append(object['bbox'])
                objects_['category'].append(object['category_id'] - 1)
                objects_['id'].append(object['id'])
            image_path = os.path.join(image_folders, image_metadata["file_name"])

            record = {
                "image": image_path,
                "filepath": image_metadata["file_name"],
                "filename": image_metadata["file_name"],
                "image_id": img_id,
                "height": image_metadata["height"],
                "width": image_metadata["width"],
                "objects": objects_,
            }

            yield record["image_id"], record

    # def _generate_examples(self, annotation_file, image_folders, split_key):
    #     with open(annotation_file, "r", encoding="utf-8") as fi:
    #         annotations = json.load(fi)
    #
    #         for image_metadata in annotations["images"]:
    #             if split_key == "train":
    #                 if image_metadata["split"] != "train" and image_metadata["split"] != "restval":
    #                     continue
    #             elif split_key == "validation":
    #                 if image_metadata["split"] != "val":
    #                     continue
    #             elif split_key == "test":
    #                 if image_metadata["split"] != "test":
    #                     continue
    #
    #             if "val2014" in image_metadata["filename"]:
    #                 image_path = image_folders["validation"] / _SPLIT_MAP["validation"]
    #             else:
    #                 image_path = image_folders["train"] / _SPLIT_MAP["train"]
    #
    #             image_path = image_path / image_metadata["filename"]
    #
    #             record = {
    #                 "image": str(image_path.absolute()),
    #                 "filepath": image_metadata["filename"],
    #                 "sentids": image_metadata["sentids"],
    #                 "filename": image_metadata["filename"],
    #                 "imgid": image_metadata["imgid"],
    #                 "split": image_metadata["split"],
    #                 "cocoid": image_metadata["cocoid"],
    #                 "sentences_tokens": [caption["tokens"] for caption in image_metadata["sentences"]],
    #                 "sentences_raw": [caption["raw"] for caption in image_metadata["sentences"]],
    #                 "sentences_sentid": [caption["sentid"] for caption in image_metadata["sentences"]],
    #             }
    #
    #             yield record["imgid"], record
