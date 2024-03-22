from transformers import pipeline
from datasets import load_dataset
import utils as u
import numpy as np
import os
from pathlib import Path
import cv2


def get_image_paths(folder):
    image_paths = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    return image_paths


def main():
    datasetpath = Path.joinpath(Path(__file__).parent.parent, 'hugdataset.py')
    ds = load_dataset(path=str(datasetpath), name="GOLD")
    subset = 'test'
    obj_detector = pipeline("object-detection", model="new_test/checkpoint-107460")

    random_indices = np.random.choice(len(ds[subset]), 10, replace=False)
    images = ds[subset].select(random_indices)['image']
    results = obj_detector(images, threshold=0.3)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 600)

    for i, img in enumerate(images):
        if not results[i]:
            continue

        cv_img = np.array(img)
        image = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        for detection in results[i]:
            score = detection["score"]
            label = detection["label"]
            box = [round(i, 2) for i in detection["box"].values()]
            print(
                f"Detected {label} with confidence "
                f"{round(score, 3)} at location {box}"
            )

        boxxed_img = u.draw_bboxes(image, [detection["box"] for detection in results[i]], [detection["label"] for detection in results[i]], True)

        cv2.imshow("Image", boxxed_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
