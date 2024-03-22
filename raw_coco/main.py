import sys
sys.path.append("..")
from dataset import Dataset
import json
from different_formats.label_studio_json_converter import convert_json


def main():
    # Paths to relevant files    
    gt_folder = "/path/to/your/dataset/images/"
    gt_file = "path/to/your/annotationfile.json"

    gt_folder = "/home/nex/Bilder/Datasets/golddataset/images/"
    gt_file = "/home/nex/Bilder/Datasets/golddataset/annotation/ann.json"

    gt_data = Dataset(gt_folder, gt_file)

    gt_data.display_images(bboxes=True, show=True)


if __name__ == "__main__":
    main()
