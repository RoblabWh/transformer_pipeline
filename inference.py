from transformers import pipeline
import torch
import os
import requests
from PIL import Image, ImageDraw


def get_image_paths(folder):
    image_paths = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    return image_paths


def main():
    folder = "/home/nex/Bilder/Datasets/small-test/images"
    image_paths = get_image_paths(folder)
    obj_detector = pipeline("object-detection", model="checkpoint-22320")

    # manual detection possible
    # with torch.no_grad():
    #     inputs = obj_detector.image_processor(images=image, return_tensors="pt")
    #     outputs = obj_detector.model(**inputs)
    #     target_sizes = torch.tensor([image.size[::-1]])
    #     results = obj_detector.image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    results = obj_detector(image_paths, threshold=0.5)

    for i, image_path in enumerate(image_paths):
        if not results[i]:
            continue

        image = Image.open(image_path)

        for detection in results[i]:
            score = detection["score"]
            label = detection["label"]
            box = [round(i, 2) for i in detection["box"].values()]
            print(
                f"Detected {label} with confidence "
                f"{round(score, 3)} at location {box}"
            )

            draw = ImageDraw.Draw(image)

            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{label} {round(score, 3)}", fill="red")

        image.show()
        wait = input("Press Enter to continue.")


if __name__ == "__main__":
    main()
