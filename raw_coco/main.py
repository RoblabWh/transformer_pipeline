import sys
sys.path.append("..")
from dataset import Dataset
import json
# from different_formats.label_studio_json_converter import convert_json


def main():
    # Paths to relevant files    
    gt_folder = "/home/nex/Bilder/Datasets/archive/DRZ_rescuedataset/images"
    gt_file = "/home/nex/Bilder/Datasets/archive/DRZ_rescuedataset/ann.json"
    # train_file = "/home/nex/Bilder/FireDetDataset/data/split/viersen2024/train.json"
    # test_file = "/home/nex/Bilder/FireDetDataset/data/split/viersen2024/test.json"
    # val_file = "/home/nex/Bilder/FireDetDataset/data/split/viersen2024/val.json"
    # viersen_folder = "/home/nex/Bilder/Datasets/Viersen2024/images"
    # viersen_file = "/home/nex/Bilder/Datasets/Viersen2024/ann.json"

    gt_data = Dataset(gt_folder, gt_file)
    gt_data.split_dataset()
    # test_data = Dataset(gt_folder, test_file)
    # train_data = Dataset(gt_folder, train_file)
    # val_data = Dataset(gt_folder, val_file)

    # viersen_data = Dataset(viersen_folder, viersen_file)

    # gt_data.add_dataset(test_data)
    # gt_data.add_dataset(train_data)
    # gt_data.add_dataset(val_data)

    # gt_data.add_dataset(viersen_data)

    # idx = [3370 + i for i in range(50)]
    # test_data.display_images(img_ids=idx, bboxes=True, show=True, shuffle=False)
    #gt_data.display_images(bboxes=True, show=True, shuffle=True)


if __name__ == "__main__":
    main()
