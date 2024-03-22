import os
import json
from utils import group_annotations_by_image
import glob
import cv2

# TODO MAYBE ONE OF THESE FUNCTIONS SHUFFLES THE CATEGORY IDS


def convert_labelstudio_to_coco_format(custom_format_data, root_dir):
    """
    Converts the label-studio custom JSON format back to COCO format.
    :param custom_format_data: The custom format data to convert. (.json)
    :param root_dir: The root directory of the images. (is needed because the image paths are relative)
    return: The converted data in COCO format.
    """
    images = []
    annotations = []
    ann_id = 1

    # Set category list
    categories_list = [{"id": 1, "name": 'fire'}, {"id": 2, "name": 'vehicle'}, {"id": 3, "name": 'human'}]

    # Set category_id in annotations
    category_name_to_id = {'fire': 1, 'vehicle': 2, 'human': 3}

    for custom_image in custom_format_data:
        # When there are no annotations for an image, load width and height from the image
        if len(custom_image["annotations"][0]["result"]) == 0:
            # TODO: This is a hack, because the image width and height are not stored in the custom format
            #       but we need them for the coco format. We can get them from the image file, but we need
            #       to know the path to the image file. We can get the path from the custom format, but we
            #       need the image width and height to get the path. So we assume that the image file is
            #       in the same folder as the custom format file and that the image file name is the same
            #       as the custom format file name.
            image_path = os.path.join(root_dir, custom_image["data"]["image"].split("/")[-1].split("=")[-1])
            img = cv2.imread(image_path)
            width, height = img.shape[1], img.shape[0]
            image = {
                "id": custom_image["id"],
                "width": width,
                "height": height,
                "file_name": custom_image["data"]["image"].split("/")[-1].split("=")[-1],
            }
        else:
            image = {
                "id": custom_image["id"],
                "width": custom_image["annotations"][0]["result"][0]["original_width"],
                "height": custom_image["annotations"][0]["result"][0]["original_height"],
                "file_name": custom_image["data"]["image"].split("/")[-1].split("=")[-1],
            }
        images.append(image)

        for ann in custom_image["annotations"][0]["result"]:
            category_name = ann["value"]["rectanglelabels"][0]

            bbox = [
                (ann["value"]["x"] / 100) * image["width"],
                (ann["value"]["y"] / 100) * image["height"],
                (ann["value"]["width"] / 100) * image["width"],
                (ann["value"]["height"] / 100) * image["height"],
            ]

            annotations.append({
                "id": ann_id,
                "image_id": image["id"],
                "category_id": category_name,
                "bbox": bbox,
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            })

            ann_id += 1

    for ann in annotations:
        ann["category_id"] = category_name_to_id[ann["category_id"]]

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
    }

    return coco_format


def convert_coco_to_labelstudio_format(coco_json, cloudstoragename):
    """
    Converts the coco format to the label-studio custom JSON format.
    :param coco_json: The coco format data to convert. (.json)
    :param cloudstoragename: The name of the cloud storage in label-studio. (e.g. "images")
    """
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_json["categories"]}
    grouped_annotations = group_annotations_by_image(coco_json["annotations"])

    custom_format_data = []
    for image in coco_json["images"]:
        if image["id"] not in grouped_annotations:
            image_annotations = []
        else:
            image_annotations = grouped_annotations[image["id"]]
        custom_format = {
            "id": image["id"],
            "annotations": [
                {
                    "result": [
                        {
                            "original_width": image["width"],
                            "original_height": image["height"],
                            "image_rotation": 0,
                            "value": {
                                "x": (ann["bbox"][0] / image["width"]) * 100,
                                "y": (ann["bbox"][1] / image["height"]) * 100,
                                "width": (ann["bbox"][2] / image["width"]) * 100,
                                "height": (ann["bbox"][3] / image["height"]) * 100,
                                "rotation": 0,
                                "rectanglelabels": [
                                    category_id_to_name[ann["category_id"]]
                                ],
                            },
                            "id": ann["id"],
                            "from_name": "label",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "origin": "manual",
                        }
                        for ann in image_annotations
                    ]
                }
            ],
            "data": {
                "image": f"/data/local-files?d={cloudstoragename}/{image['file_name']}"
            },
        }
        custom_format_data.append(custom_format)

    return custom_format_data


def convert_json(input_json, output_json, root_dir, mode_coco_to_custom=True):
    """
    Converts the input json file to the output json file.
    :param input_json: The input json file. (.json)
    :param output_json: The output json file. (.json)
    :param root_dir: The root directory of the images. (is needed because the image paths are relative)
    :param mode_coco_to_custom: If True, the input json file is in coco format and the output json file is in
                                label-studio custom format. If False, the input json file is in label-studio custom
                                format and the output json file is in coco format.
    """
    if mode_coco_to_custom:
        cloudstoragename = 'images'


    # Load the JSON file
    with open(input_json, "r") as custom_file:
        json_data = json.load(custom_file)

    if mode_coco_to_custom:
        converted_data = convert_coco_to_labelstudio_format(json_data, cloudstoragename)
    else:
        converted_data = convert_labelstudio_to_coco_format(json_data, root_dir)

    # Save the converted JSON to a new file
    with open(output_json, "w") as output_file:
        json.dump(converted_data, output_file, indent=2)


def read_n_dump(input_json, output_json):
    """
    Reads the input json file and dumps it to the output json file with indent=2.
    """

    with open(input_json, "r") as custom_file:
        json_data = json.load(custom_file)
    with open(output_json, "w") as output_file:
        json.dump(json_data, output_file, indent=2)


def split_json_into_batches(input_json, output_folder, batch_size = 1800):
    """
    Splits the input json file into multiple json files.
    :param input_json: The input json file in label-studio custom format. (.json)
    :param output_folder: The output folder where the json files will be saved.
    """

    with open(input_json, "r") as custom_file:
        data = json.load(custom_file)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        filename = f"batch_{i // batch_size + 1}.json"
        
        with open(os.path.join(output_folder, filename), "w") as outfile:
            json.dump(batch, outfile)


def merge_json(input_folder, output_json):
    """
    Merges all json files in the input folder into one json file.
    :param input_folder: The input folder where the json files are located.
    :param output_json: The output json file.
    """

    # Get all the JSON files in the input folder
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    # Sort JSON files by name
    json_files.sort(key=lambda x: int(x.split("_")[-2]))

    new_id = 1
    new_anno_id = 1
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fire"},
                       {"id": 2, "name": "vehicle"},
                       {"id": 3, "name": "human"}]}

    # Create a mapping for old to new image ids
    old_to_new_id = {}

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # Remap the image ids
        for image in data['images']:
            old_id = image['id']
            image['id'] = new_id
            old_to_new_id[old_id] = new_id
            new_id += 1
            merged_data['images'].append(image)

        # Remap the annotation ids and image ids in annotations
        for annotation in data['annotations']:
            annotation['id'] = new_anno_id
            new_anno_id += 1
            annotation['image_id'] = old_to_new_id[annotation['image_id']]
            merged_data['annotations'].append(annotation)

    # Write the merged data to a json file
    with open(output_json, 'w') as f:
        json.dump(merged_data, f, indent=2)


def main():
    input_json = "/home/windos/Downloads/project-1-at-2023-12-13-15-09-ef7173e7.json"
    output_json = "/home/windos/Desktop/datasets/Datensatz-Optimierung/ann_coco.json"
    convert_json(input_json, output_json, root_dir='/home/windos/Desktop/datasets/Datensatz-Optimierung/images', mode_coco_to_custom=False)


if __name__ == "__main__":
    main()
