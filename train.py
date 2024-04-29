from transformers import AutoImageProcessor, AutoModelForObjectDetection, EvalPrediction
import torch
import functools
import torchvision.transforms.v2 as transforms
from torchmetrics.detection import MeanAveragePrecision
import numpy as np

from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModel

from datasets import load_dataset

from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

_SUPP_MODELS = ["deformable-detr", "yolos-small"]
_MASK_MODELS = ["deformable-detr"]
_NOMASK_MODELS = ["yolos-small"]

# TODO must do: export CUDA_VISIBLE_DEVICES=1
class UniversalTrainer(object):

    def __init__(self, checkpoint, args, dataset):
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.args = args

        self.dataset = load_dataset(dataset["path"], dataset["version"])

        # TODO Transforms may differ from model to model, need to check on this and make it more flexible
        # self.transform = albumentations.Compose([
        #     albumentations.Resize(224, 224),
        #     albumentations.HorizontalFlip(p=1.0),
        #     albumentations.RandomBrightnessContrast(p=1.0),
        # ],
        #     bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]), )
        # TODO Inspect that BBox Error with Albumentations, it stemmed from the fact that the bbox was not in the
        #  correct bounds of the image, need to check on this
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),])

        self.dataset["train"] = self.dataset["train"].with_transform(self.transform_aug_ann)
        #self.dataset["validation"] = self.dataset["validation"].with_transform(self.transform_aug_ann)
        self.dataset["validation"] = self.dataset["validation"].select([_ for _ in range(10)]).with_transform(self.transform_aug_ann)
        # dataset = dataset.train_test_split(test_size=0.2)

        # Delete the bad bboxes from the dataset
        # keep = [i for i in range(len(self.dataset["train"])) if i != 1948]  # TODO this is a dirty fix, better check this
        # self.dataset["train"] = self.dataset["train"].select(keep)          # TODO on dataset creation

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

        self.modeltype = self.set_modeltype()

        self.mAP = MeanAveragePrecision(box_format="cxcywh", class_metrics=True)
        self.metrics = functools.partial(self.compute_metrics, map=self.mAP)

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.image_processor,
            data_collator=self.collate_fn,
            compute_metrics=self.metrics,
        )

    def set_modeltype(self):
        from pathlib import Path
        modelname = Path(self.model.name_or_path).name
        if modelname not in _SUPP_MODELS:
            from warnings import warn
            warn(f"Model {modelname} is not supported yet and may throw errors, please use one of the following: {_SUPP_MODELS}")

        return modelname

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
            out_image, out_bboxes, out_category = self.transform(image, objects["bbox"], objects["category"])

            area.append(objects["area"])
            images.append(out_image)
            bboxes.append(out_bboxes)
            categories.append(out_category)

        targets = [
            {"image_id": id_, "annotations": self.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return self.image_processor(images=images, annotations=targets, return_tensors="pt")

    def collate_with_mask(self, batch):
        pixel_values = [example["pixel_values"] for example in batch]
        encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
        labels = [example["labels"] for example in batch]
        batch = {"pixel_values": encoding["pixel_values"], "pixel_mask": encoding["pixel_mask"], "labels": labels}
        return batch

    def collate_without_mask(self, batch):
        pixel_values = [example["pixel_values"] for example in batch]
        encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
        labels = [example["labels"] for example in batch]
        batch = {"pixel_values": encoding["pixel_values"], "labels": labels}
        return batch

    def collate_fn(self, batch):
        if self.modeltype in _MASK_MODELS:
            return self.collate_with_mask(batch)
        elif self.modeltype in _NOMASK_MODELS:
            return self.collate_without_mask(batch)
        else:
            return self.collate_without_mask(batch)

    def compute_metrics(self, eval_pred: EvalPrediction, map: MeanAveragePrecision):
        print(eval_pred) #TODO HIER WEITERMACHEN
        (scores, pred_boxes, last_hidden_state, encoder_last_hidden_state), labels = eval_pred
        print(scores)
        # scores shape: (batch_size, number of detected anchors, num_classes + 1) last class is the no-object class
        # pred_boxes shape: (batch_size, number of detected anchors, 4)
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/detr-resnet50/README.md
        predictions = []
        for score, box in zip(scores, pred_boxes):
            # Extract the bounding boxes, labels, and scores from the model's output
            pred_scores = torch.from_numpy(score[:, :-1])  # Exclude the no-object class
            pred_boxes = torch.from_numpy(box)
            pred_labels = torch.argmax(pred_scores, dim=-1)

            # Get the scores corresponding to the predicted labels
            pred_scores_for_labels = torch.gather(pred_scores, 1, pred_labels.unsqueeze(-1)).squeeze(-1)
            predictions.append(
                {
                    "boxes": pred_boxes,
                    "scores": pred_scores_for_labels,
                    "labels": pred_labels,
                }
            )
        target = [
            {
                "boxes": torch.from_numpy(labels[i]["boxes"]),
                "labels": torch.from_numpy(labels[i]["class_labels"]),
            }
            for i in range(len(labels))
        ]
        map.update(preds=predictions, target=target)
        results = map.compute()
        # Convert tensors to scalars/lists, MLFlow doesn't really like tensors
        results = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in results.items()}
        return results

    def train(self):
        self.trainer.train()


if __name__ == "__main__":

    # Couldn't observe differences on either trainingspeed or memoryusage with:
    # fp16

    # Further look into:
    # https://huggingface.co/docs/peft/index

    # Accelerate
    # model = AutoModel.from_pretrained(checkpoint)

    # Multi GPU
    # https://huggingface.co/docs/transformers/perf_train_gpu_many

    # LR 2e-3 batchsize=4; gradient_acc= 16
    # {'train_runtime': 468540.382, 'train_samples_per_second': 6.905, 'train_steps_per_second': 0.108, 'train_loss': 2.926033954923115, 'epoch': 1196.44}
    # very bad - no detection

    checkpoint = "SenseTime/deformable-detr"
    #checkpoint = "hustvl/yolos-small"
    training_args = TrainingArguments(
        output_dir="yolosss",
        remove_unused_columns=False,
        #fp16=True,
        learning_rate=1e-5, # 1e-5 worked good
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1200,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_accumulation_steps=8,
        eval_steps=5,
        save_strategy="epoch",
        save_total_limit=4,
        load_best_model_at_end=False,
        push_to_hub=False,
        gradient_accumulation_steps=1,
        dataloader_num_workers=16,
        #torch_compile=True,
        optim="adamw_torch_fused",
        #torch_compile_backend=_dynamo.optimize("inductor"),
    )

    dataset = {"path": "RoblabWhGe/FireDetDataset", "version": "GOLD", "token": True, "trust_remote_code": True}
    #dataset = load_dataset("RoblabWhGe/FireDetDataset", token=True, trust_remote_code=True)
    dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size)

    trainer = UniversalTrainer(checkpoint, training_args, dataset)

    trainer.train()



