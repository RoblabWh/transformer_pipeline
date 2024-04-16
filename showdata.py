from datasets import load_dataset
import cv2
import numpy as np
import utils as u


def show_image_with_boxes(index, subset_, show_classes=True):
    example = ds[subset_][index]
    image = cv2.cvtColor(np.array(example['image']), cv2.COLOR_RGB2BGR)
    labels = example['objects']['category']
    bboxes = example['objects']['bbox']
    boxxed_img = u.draw_bboxes(image, bboxes, labels, show_classes)
    return boxxed_img


if __name__ == "__main__":
    ds = load_dataset("RoblabWhGe/FireDetDataset", token=True, trust_remote_code=True)
    subset = 'validation'
    current_index = 0
    total_images = len(ds[subset])

    # sample images
    samples = total_images
    idxs = np.random.choice(np.arange(total_images), samples, replace=False).tolist()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 600)
    print("Press '->' for next image, '<-' for previous image, or 'q' to quit: ")

    while True:
        idx = idxs[current_index]
        boxed_image = show_image_with_boxes(idx, subset, True)
        cv2.imshow("Image", boxed_image)
        key = cv2.waitKey(0)

        if key == 83:
            if current_index < samples - 1:
                current_index += 1
            else:
                #print("This was the last image, next image will be first in this dataset.")
                current_index = 0
        elif key == 81:
            if current_index > 0:
                current_index -= 1
            else:
                #print("This was the first image, next image will be the last in this dataset.")
                current_index = samples - 1
        elif key == ord('q'):
            break
