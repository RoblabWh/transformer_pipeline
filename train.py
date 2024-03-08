from transformers import AutoImageProcessor, AutoModelForObjectDetection
import albumentations
import numpy as np

from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset


class UniversalTrainer(object):

    def __init__(self, checkpoint, args, dataset):
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.args = args

        self.dataset = load_dataset(dataset["path"], dataset["version"])

        # TODO Transforms may differ from model to model, need to check on this and make it more flexible
        self.transform = albumentations.Compose([
            albumentations.Resize(224, 224),
            albumentations.HorizontalFlip(p=1.0),
            albumentations.RandomBrightnessContrast(p=1.0),
        ],
            bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]), )

        self.dataset["train"] = self.dataset["train"].with_transform(self.transform_aug_ann)
        # dataset = dataset.train_test_split(test_size=0.2)

        # Delete the bad bboxes from the dataset
        keep = [i for i in range(len(self.dataset["train"])) if i != 1948]  # TODO this is a dirty fix, better check this
        self.dataset["train"] = self.dataset["train"].select(keep)          # TODO on dataset creation

        # Create maps for the labels and ids
        labels = self.dataset["train"].features["objects"].feature["category"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        self.model = AutoModelForObjectDetection.from_pretrained(checkpoint,
                                                                 id2label=id2label,
                                                                 label2id=label2id,
                                                                 ignore_mismatched_sizes=True)

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.dataset["train"],
            tokenizer=self.image_processor,
            data_collator=self.collate_fn,
            # compute_metrics=compute_metrics,
        )

    def formatted_anns(self, image_id, category, area, bbox):
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id,
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)
        return annotations

    def transform_aug_ann(self, examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = self.transform(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": self.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return self.image_processor(images=images, annotations=targets, return_tensors="pt")

    def collate_fn(self, batch):
        pixel_values = [example["pixel_values"] for example in batch]
        encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
        labels = [example["labels"] for example in batch]
        batch = {"pixel_values": encoding["pixel_values"], "pixel_mask": encoding["pixel_mask"], "labels": labels}
        return batch

    def train(self):
        self.trainer.train()


if __name__ == "__main__":

    checkpoint = "SenseTime/deformable-detr"
    training_args = TrainingArguments(
        output_dir="test_detr",
        remove_unused_columns=False,
        fp16=True,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    dataset = {"path": "hugdataset.py", "version": "GOLD"}
    trainer = UniversalTrainer(checkpoint, training_args, dataset)
    trainer.train()


