# Dataset Handling for COCO datasets

#### Helps to manage **existing** COCO datasets.

* Merge multiple COCO datasets into one.
* Splits COCO datasets into train, val and sets. 
* Updates annotations, image names and image ids.
* Shows data

# Dependencies
----

- tqdm
- sklearn

# Usage
----

Something like this:
    
    ```python
    from dataset import Dataset
    import random

    # create dataset object
    dataset = Dataset("/path/to/images/", "/path/to/ann.json")
    random_ids = random.sample(dataset.images.keys(), 50)
    dataset.display_images(random_ids, show=True, save=False)

    another_dataset = Dataset("/path/to/images/", "/path/to/ann.json")
    dataset.incorporate_dataset(another_dataset)
    ```

## Manual Labeling

For manual correction of the bounding boxes we use [label-studio](https://labelstud.io/). Install it over pip and set up the local storage according to the [docs](https://labelstud.io/guide/storage.html). 

To import the current annotations you need to convert the COCO format to the format label-studio is using. I've run into errors when trying to convert back to COCO from label-studio, if this happens to you, export and convert back to COCO manually. Scripts for both operations are provided in `label_studio_json_converter.py`.

If you want to use local storage with label-studio. Start like this: `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/dataset/root/folder label-studio`

