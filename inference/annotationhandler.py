from inference.customcoco import CustomCOCO
import cv2
import json


class AnnotationHandler:

    def __init__(self, args, datahandler_image_names, datahandler_images, keep_coco_format=True):
        self.custom_coco = None
        self.ann_path = None
        self.config = None

        self.CLASSES = ('fire', 'vehicle', 'human')
        self.PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142)]

        # Relevant config paths
        self.ann_path = "ann.json"
        self.score_thr = args.score_thr

        self.classes = [{"id": 1, "name": "fire"},
                        {"id": 2, "name": "vehicle"},
                        {"id": 3, "name": "human"}]

        self.keep_coco_format = keep_coco_format
        self.custom_coco = None
        self.update_ann_file(datahandler_image_names, datahandler_images)

    def update_ann_file(self, datahandler_image_names, datahandler_images):
        self.custom_coco = self.create_empty_ann(datahandler_image_names, datahandler_images)

    def create_empty_ann(self, img_names, imgs):
        """
        Create an empty annotation file with the same structure as COCO
        :return: empty annotation file
        """

        if img_names is None and imgs is None:
            raise ValueError("No images to create annotation file. This is most likely a bug.")

        # Get Hight and Width from images
        if imgs is None:
            imgs = [cv2.imread(str(img_name)) for img_name in img_names]
            img_sizes = [[img.shape[0], img.shape[1]] for img in imgs]  # (h, w)
        else:
            img_sizes = [[img.height, img.width] for img in imgs]  # (h, w)

        # Create image dictonary with id, name, height and width, load images from path if images is None
        images = [{"id": i, "file_name": str(img_name), "height": img_size[0], "width": img_size[1]} for
                  i, (img_name, img_size) in enumerate(zip(img_names, img_sizes))]

        ann = {"images": images, "annotations": [], "categories": self.classes}
        with open(self.ann_path, 'w') as json_ann_file:
            json.dump(ann, json_ann_file, ensure_ascii=False, indent=4)

        return CustomCOCO(ann_file=self.ann_path, score_thr=self.score_thr, keep_coco_format=self.keep_coco_format)

    def get_json(self, results):
        """
        Get the annotation in json format
        :param results: results from the inference
        :return: json annotation
        """
        annotations = self.custom_coco.det2ann(results)
        with open(self.ann_path) as json_ann_file:
            ann = json.load(json_ann_file)
            ann["annotations"] = annotations
        return ann

    def save_results_in_json(self, results):
        """
        Save the results in a json file
        :param out_folder: folder where to save the json file
        :param results: results to save
        :return:
        """
        annotations = self.custom_coco.det2ann(results)
        # open json file
        with open(self.ann_path) as json_ann_file:
            ann = json.load(json_ann_file)
            ann["annotations"] = annotations
        with open(self.ann_path, 'w') as json_ann_file:
            json.dump(ann, json_ann_file, ensure_ascii=False, indent=4)
