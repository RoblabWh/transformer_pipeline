import argparse
from inference.datahandler import DataHandler
from inference.inference_engine import Inferencer


def main(args):
    datahandler = DataHandler(args)
    inferencer = Inferencer(args.checkpoints, args.score_thr)

    data = datahandler.get_data()
    results = inferencer(data)

    result = datahandler.postprocess(results)
    datahandler.save_annotation(result)
    if not args.nodisplay:
        datahandler.show(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Huggingface Transformers and COCO Datasets')

    # Use images on disk for inference
    parser.add_argument('--inputfolder', default=None, type=str, help='Input folder that is searched')
    parser.add_argument('--checkpoints', default=str, nargs='+', help='Checkpoints for Models to use for inference')

    # Use huggingface dataset for inference
    parser.add_argument('--dataset', default=None, type=str, help='Dataset to use for inference')
    parser.add_argument('--dataset_name', default=None, type=str, help='Name of the Dataset to use for inference')
    parser.add_argument('--subset', default='test', help='Subset of the Dataset to use for inference')
    parser.add_argument('--num_images', default=20, type=int, help='Number of images to use for inference')

    parser.add_argument('--outputfolder', type=str, nargs='+', default=None, help='Outputfolder everything get\'s saved in')
    parser.add_argument('--ann_path', type=str, default=None, help='Path to the annotation file.')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.jpg', '.png'], help='File extensions that are searched for')
    parser.add_argument('--pattern', default='.', help='Regex Pattern for input images')
    parser.add_argument('--include_subdirs', action=argparse.BooleanOptionalAction, help='Searches images additionally in all subdirs of input_folder (default: False)')
    parser.add_argument('--score_thr', default=0.5, help='Threshold for BBox Detection (default: 0.5)')
    parser.add_argument('--nodisplay', action=argparse.BooleanOptionalAction, help='Show the results (default: False)')
    parser.add_argument('--print_results', action=argparse.BooleanOptionalAction, help='Print results to console (default: False)')

    # Image manipulation
    parser.add_argument('--split', action=argparse.BooleanOptionalAction, help='Split images into tiles fore more precise detection (default: False)')
    parser.add_argument('--max_splitting_steps', default=1, help='Maximum number of splitting steps, splits into 4 images each time (default: 1)')

    # possibly unused
    parser.add_argument('--batch_size', default=5, help='Batch size for Model (default: 5)')
    parser.add_argument('--create_coco', action=argparse.BooleanOptionalAction, default=True, help='Create coco annotation file with the results from the inference')

    args = parser.parse_args()
    main(args)
