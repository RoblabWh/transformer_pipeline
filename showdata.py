import torch
from datasets import load_dataset
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

if __name__ == "__main__":
    ds = load_dataset("hugdataset.py", "GOLD")
    example = ds['train'][1948]
    categories = ds['train'].features['objects'].feature['category']

    boxes_xywh = torch.tensor(example['objects']['bbox'])
    boxes_xyxy = box_convert(boxes_xywh, 'xywh', 'xyxy')
    labels = [categories.int2str(x) for x in example['objects']['category']]
    img = to_pil_image(
        draw_bounding_boxes(
            pil_to_tensor(example['image']),
            boxes_xyxy,
            colors="red",
            labels=labels,
        )
    )
    img.show()
