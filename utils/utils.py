import json
import cv2
from pathlib import Path
import numpy as np


def empty_coco_ann():
    annotations = {'images': [], 'categories': [
        {
            "supercategory": "Defect",
            "id": 1,
            "name": "fire"
        },
        {
            "supercategory": "Defect",
            "id": 2,
            "name": "vehicle"
        },
        {
            "supercategory": "Defect",
            "id": 3,
            "name": "human"
        }
    ], 'annotations': []}
    return annotations


def read_json(path):
    """
    A function that reads a json file and returns the data.
    :param path:
    :return:
    """
    with open(path) as f:
        data = json.load(f)
    return data

def get_repository_root(current_path):
    # Search for a prominent marker of the project root
    current_path = Path(current_path)
    for parent in current_path.parents:
        if (parent / '.git').is_dir() or (parent / '.git').is_file() or (parent / '.detection').is_file():
            return parent
    raise Exception('Could not find project root directory')

def get_models_json(path):
    if path.stem == "transformer_pipeline":
        return path
    else:
        return path.joinpath("transformer_pipeline")

def draw_bboxes(img, bboxes, labels, show_classes):
    """
    A function that draws bounding boxes on an image.
    :param img:
    :param bboxes:
    :return:
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    categories = ["fire", "vehicle", "human"]

    for i, bbox in enumerate(bboxes):

        if isinstance(labels[i], str):
            label_idx = categories.index(labels[i]) # Inference
        else:
            label_idx = labels[i] # Normal Dataset

        label = categories[label_idx]

        if label_idx >= len(colors):
            color = (255, 255, 255)
        else:
            color = colors[label_idx]

        if isinstance(bbox, dict):
            x, y, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        else:
            x, y, w, h = bbox
            xmax = x + w
            ymax = y + h

        # Display BBOX
        cv2.rectangle(img, (int(x), int(y)), (int(xmax), int(ymax)), color, 2)

        # Display category ID
        if show_classes:
            text_position = (int(x)+2, int(y)+17)
            cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def compare_images(img1, img2, threshold=10):
    """
    A function that compares two images and returns if they are the same.
    :param img1:
    :param img2:
    :param threshold:
    :return:
    """
    diff = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    return not np.any(diff)


# TODO testen
def delete_from_json_images(json_file, img_names):
    """
    A function that deletes an image and its annotations from a json file.
    :param json_file: Annotation file
    :param img_names: Image basenames list
    :return:
    """
    with open(json_file) as f:
        data = json.load(f)
    new_images = []
    deleted_ids = []
    for img in data['images']:
        if img['file_name'] not in img_names:
            new_images.append(img)
        else:
            deleted_ids.append(img['id'])

    data['images'] = new_images
    with open(json_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    delete_from_json_annotation(json_file, deleted_ids)


def delete_images_from_json(json_file, images):
    """
    A function that deletes images from a json file.
    :param json_file:
    :param images:
    """
    with open(json_file) as f:
        data = json.load(f)
    new_images = []
    for img in data['images']:
        if img['file_name'] not in images:
            new_images.append(img)

    data['images'] = new_images
    with open(json_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def delete_from_json_annotation(json_file, annotation):
    """
    A function that deletes an annotation from a json file.
    :param json_file:
    :param annotation:
    :return:
    """
    with open(json_file) as f:
        data = json.load(f)
    new_annotations = []
    for ann in data['annotations']:
        if ann['image_id'] not in annotation:
            new_annotations.append(ann)
    data['annotations'] = new_annotations
    with open(json_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def calc_iou(bbox1, bbox2):
    """
    Calculates the IoU for two bounding boxes
    :param bbox1: bounding box 1 in COCO format [x, y, width, height]
    :param bbox2: bounding box 2 in COCO format [x, y, width, height]
    :return: IoU
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - inter
    if union == 0:
        iou = 0
    else:
        iou = inter / union

    return iou


def to_xywh(_bbox):
    return [
        _bbox['xmin'],
        _bbox['ymin'],
        _bbox['xmax'] - _bbox['xmin'],
        _bbox['ymax'] - _bbox['ymin'],
    ]


def to_xyxy(bbox):
    return {
        'xmin': bbox[0],
        'ymin': bbox[1],
        'xmax': bbox[0] + bbox[2],
        'ymax': bbox[1] + bbox[3]
    }


def to_dict(bbox, label, score):
    return {
        'box': to_xyxy(bbox),
        'label': label,
        'score': score
    }
