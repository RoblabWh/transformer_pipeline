from pycocotools.coco import COCO
from utils import to_xywh


class CustomCOCO(COCO):

    CLASSES = ('fire', 'vehicle', 'human')

    def __init__(self, ann_file=None, score_thr=0.3, keep_coco_format=True):
        super().__init__(annotation_file=ann_file)
        self.score_thr = score_thr
        self.keep_coco_format = keep_coco_format

    def __len__(self):
        return len(self.dataset['images'])

    def det2ann(self, results):

        if not isinstance(results, list):
            raise TypeError('invalid type of results')

        coco_correction = 0
        if self.keep_coco_format:
            coco_correction = 1

        annotations = []
        id = 1
        images = self.dataset['images']

        for image in images:
            img_id = image['id']
            image_result = results[img_id]
            if not len(image_result) > 0:
                continue
            assert (list(image_result[0]['box'].keys()) == ['xmin', 'ymin', 'xmax', 'ymax']), f"BBox are this Format: {image_result[0]['box'].keys()}, but need to be in this format: ['xmin', 'ymin', 'xmax', 'ymax']."
            for i, instance in enumerate(image_result):
                if instance['score'] > self.score_thr:
                    data = dict()
                    data['id'] = id
                    id += 1
                    data['image_id'] = img_id
                    data['bbox'] = to_xywh(instance['box'])
                    data['score'] = float(instance['score'])
                    data['category_id'] = [_['name'] for _ in list(self.cats.values())].index(instance['label']) + coco_correction # add 1 to stay in COCO format
                    data['segmentation'] = []
                    annotations.append(data)
        return annotations

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
