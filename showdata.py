from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names, load_dataset_builder
import cv2
import argparse
import numpy as np
import utils as u

from functools import partial
from typing import Any, List, Mapping, Tuple

import albumentations as A
import numpy as np
from datasets import load_dataset

from transformers import (
    AutoImageProcessor,
)
from transformers.image_processing_utils import BatchFeature

def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
    examples: Mapping[str, Any], transform: A.Compose, image_processor: AutoImageProcessor
) -> BatchFeature:
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    return result

def do_transforms(dataset, max_size = 1333):
    image_processor = AutoImageProcessor.from_pretrained(
        "SenseTime/deformable-detr",
        # At this moment we recommend using external transform to pad and resize images.
        # It`s faster and yields much better results for object-detection models.
        do_pad=False,
        do_resize=False,
        do_convert_annotations=True,
        # We will save image size parameter in config just for reference
        size={"longest_edge": max_size},
    )
    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------

    basic_transforms = [
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(max_size, max_size, border_mode=0, value=(128, 128, 128), position="top_left"),
    ]
    train_augment_and_transform = A.Compose(
        [
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=max_size, p=1.0),
                    A.RandomSizedBBoxSafeCrop(height=max_size, width=max_size, p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
            *basic_transforms,
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    validation_transform = A.Compose(
        basic_transforms,
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)
    dataset["combined"] = dataset["combined"].with_transform(validation_transform_batch)

def show_image(example, show_bboxes, show_classes=True):
    detr = not "image" in example.keys()
    if not detr:
        image = cv2.cvtColor(np.array(example['image']), cv2.COLOR_RGB2BGR)
        labels = example['objects']['category']
        bboxes = example['objects']['bbox']
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (((example['pixel_values'].permute(1, 2, 0).numpy() * std) + mean) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        labels = example['labels']['class_labels'].numpy()
        bboxes = example['labels']['boxes'].numpy() * image.shape[0]

    if show_bboxes:
        image = u.draw_bboxes(image, bboxes, labels, show_classes, detr=detr)
    return image

def show(args):
    datasetconfig = args.dataset
    splits = get_dataset_split_names(datasetconfig, trust_remote_code=True)
    if args.split not in splits:
        print(f"Split {args.split} not found in dataset {datasetconfig}\n"
              f"Available splits: {splits}")
        return
    if args.info:
        ds_builder = load_dataset_builder(datasetconfig, trust_remote_code=True)
        print(ds_builder.info.description)
        configs = get_dataset_config_names(datasetconfig, trust_remote_code=True)
        print(configs)
        print(splits)
    dataset = load_dataset(datasetconfig, args.datasetname, token=True, trust_remote_code=True)

    if args.transform:
        do_transforms(dataset)

    total_images = len(dataset[args.split])

    # sample images
    samples = total_images if args.samples is None else args.samples
    idxs = np.random.choice(np.arange(total_images), samples, replace=False).tolist() if args.randomize else np.arange(samples).tolist()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 600)

    print("Press 'd' for next image, 'a' for previous image, or 'q' to quit: ")

    current_index = 0
    while True:
        idx = idxs[current_index]
        image = dataset[args.split][idx]
        boxed_image = show_image(image, args.bboxes, args.classnames)
        cv2.imshow("Image", boxed_image)
        key = cv2.waitKey(0)

        if key == ord('d'):
            if current_index < samples - 1:
                current_index += 1
            else:
                #print("This was the last image, next image will be first in this dataset.")
                current_index = 0
        elif key == ord('a'):
            if current_index > 0:
                current_index -= 1
            else:
                #print("This was the first image, next image will be the last in this dataset.")
                current_index = samples - 1
        elif key == ord('q'):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lookup and show images from a dataset')

    # Use images on disk for inference
    parser.add_argument('--dataset', default="/home/nex/Bilder/Datasets/FireDetDataset_TestUp/", type=str, help='Dataset to load images from (Either Hub or Local)')
    parser.add_argument('--datasetname', default="MODIFIED", type=str, help='Dataset BuilderConfig Name to load, if "None" will use the default')
    parser.add_argument('--split', default="combined", type=str, help='Subset to load, either train, validation, test or combined (default: \'train\')')
    parser.add_argument('--samples', default=None, type=int, help='Number of samples to show, by default shows all images in the subset (default: None)')
    parser.add_argument('--info', default=True, type=bool, help='Print dataset info (default: False)')
    parser.add_argument('--bboxes', default=True, type=bool, help='Show bounding boxes on the images (default: False)')
    parser.add_argument('--randomize', default=False, type=bool, help='Randomizes the images in the dataset before displaying (default: False)')
    parser.add_argument('--classnames', default=False, type=bool, help='Applies transformation to the images, as it is done in training (default: False)')
    parser.add_argument('--transform', default=True, type=bool, help='Applies transformation to the images, as it is done in training (default: False)')

    args = parser.parse_args()

    show(args)
