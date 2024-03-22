from datasets import load_dataset
import cv2
import numpy as np
import random
import utils as u


def show_image_with_boxes(index, subset_, show_classes=True):
    example = ds[subset_][index]
    image = cv2.cvtColor(np.array(example['image']), cv2.COLOR_RGB2BGR)
    labels = example['objects']['category']
    bboxes = example['objects']['bbox']
    boxxed_img = u.draw_bboxes(image, bboxes, labels, show_classes)
    return boxxed_img


if __name__ == "__main__":
    ds = load_dataset("hugdataset.py", "GOLD")
    subset = 'validation'
    current_index = 0
    total_images = len(ds[subset])  # Assuming 'ds' has this attribute
    # sample images
    samples = 10
    idxs = np.random.choice(np.arange(total_images), samples, replace=False).tolist()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 600)
    print("Press 'n' for next image, 'p' for previous image, or 'q' to quit: ")

    while True:
        idx = idxs[current_index]
        boxed_image = show_image_with_boxes(idx, subset, True)
        cv2.imshow("Image", boxed_image)
        key = cv2.waitKey(0)

        if key == ord('n'):
            if current_index < samples - 1:
                current_index += 1
            else:
                print("This is the last image.")
        elif key == ord('p'):
            if current_index > 0:
                current_index -= 1
            else:
                print("This is the first image.")
        elif key == ord('q'):
            break
        else:
            print("Invalid input, please press 'n', 'p', or 'q'.")
