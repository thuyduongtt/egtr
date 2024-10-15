import argparse
from glob import glob

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from data.visual_genome import VGDataset
from lib.pytorch_misc import argsort_desc
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from train_egtr import collate_fn
from pathlib import Path

parser = argparse.ArgumentParser()
# Path
parser.add_argument("--images_path", type=str, default="images")
parser.add_argument("--artifact_path", type=str, required=True)
parser.add_argument("--object_threshold", type=float, default=0.3)
parser.add_argument("--relation_threshold", type=float, default=0.3)
parser.add_argument("--top_k", type=int, default=10)
args = parser.parse_args()

# config
architecture = "SenseTime/deformable-detr"
min_size = 800
max_size = 1333
artifact_path = args.artifact_path

# feature extractor
feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    architecture, size=min_size, max_size=max_size
)

test_dataset = VGDataset(
    data_folder='dataset/visual_genome',
    feature_extractor=feature_extractor,
    split='test',
    num_object_queries=200,
)
id2relation = test_dataset.rel_categories  # 50 relationships
id2label = {
    k - 1: v["name"] for k, v in test_dataset.coco.cats.items()
}  # 0 ~ 149

test_dataloader = DataLoader(
    test_dataset,
    collate_fn=lambda x: collate_fn(x, feature_extractor),
    batch_size=1,
    pin_memory=True,
    num_workers=1,
    persistent_workers=True,
)

# model
config = DeformableDetrConfig.from_pretrained(artifact_path)
model = DetrForSceneGraphGeneration.from_pretrained(
    architecture, config=config, ignore_mismatched_sizes=True
)
ckpt_path = sorted(
    glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
)[-1]
state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
for k in list(state_dict.keys()):
    state_dict[k[6:]] = state_dict.pop(k)  # "model."

model.load_state_dict(state_dict)
model.cuda()
model.eval()


@torch.no_grad()
def inference(output_dir):
    i = 0
    for batch in test_dataloader:
        # print(batch.keys())  # ['pixel_values', 'pixel_mask', 'labels']

        i += 1
        if i > 100:
            break

        # output
        outputs = model(
            pixel_values=batch['pixel_values'].cuda(),
            pixel_mask=batch['pixel_mask'].cuda(),
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )

        pred_logits = outputs['logits'][0]
        obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), -1)
        pred_classes_numpy = pred_classes.cpu().clone().numpy()
        sub_ob_scores = torch.outer(obj_scores, obj_scores)
        sub_ob_scores[
            torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
        ] = 0.0  # prevent self-connection

        # GET VALID OBJECTS
        valid_obj_indices = (obj_scores >= args.object_threshold).nonzero()[:, 0]
        valid_obj_classes = pred_classes[valid_obj_indices]  # [num_valid_objects]

        # GET OBJECT BOUNDING BOXES
        pred_boxes = outputs['pred_boxes'][0]
        valid_obj_boxes = pred_boxes[valid_obj_indices]  # [num_valid_objects, 4] (x_center, y_center, width, height)

        # GET RELATIONSHIPS
        pred_connectivity = outputs['pred_connectivity'][0]
        pred_rel = outputs['pred_rel'][0]
        pred_rel = torch.mul(pred_rel, pred_connectivity)

        triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)
        pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[:args.top_k, :]  # [pred_rels, 2(s,o)]
        rel_scores = (pred_rel.cpu().clone().numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1]])
        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1)))
        pred_boxes_numpy = pred_boxes.cpu().clone().numpy()
        triplets, triplets_box = get_triplets(pred_rels, pred_classes_numpy, pred_boxes_numpy)
        image_id = batch["labels"][0]["image_id"][0].cpu().numpy()

        print('==================== IMAGE_ID:', image_id)
        print(triplets)

        if len(triplets) > 0:
            image = Image.open(f'dataset/visual_genome/images/{image_id}.jpg')
            if not Path(output_dir).exists():
                Path(output_dir).mkdir(parents=True)
            visualization(image, valid_obj_boxes, valid_obj_classes, f'{output_dir}/{image_id}.jpg')


def get_triplets(pred_triplets, classes, boxes):
    triplets, triplet_boxes = [], []
    for sub, obj, rel in pred_triplets:
        triplet_boxes.append([boxes[sub], boxes[obj]])
        sub = id2label[classes[sub]]
        obj = id2label[classes[obj]]
        rel = id2relation[rel]
        triplets.append([sub, rel, obj])
    return triplets, triplet_boxes


def visualization(img, boxes, obj_classes, file_path):
    colors = ['red', 'blue', 'orange', 'green', 'yellow', 'purple']
    obj_list = obj_classes.cpu().numpy()
    bboxes = boxes.cpu().numpy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    i = 0
    for bbox, obj in zip(bboxes, obj_list):
        x_center, y_center, width, height = bbox
        x_max = int((x_center + width / 2) * img.width)
        y_max = int((y_center + height / 2) * img.height)
        x_min = int((x_center - width / 2) * img.width)
        y_min = int((y_center - height / 2) * img.height)
        obj = id2label[obj]
        text_width, text_height = 10, 10
        text_x = max(x_min + text_width + 5, 0)
        text_y = y_min + text_height + 5
        draw.text((text_x, text_y), str(obj), fill=colors[i], font=font)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=colors[i], fill=None, width=3)
        i = (i + 1) % 6
    img.save(file_path)


inference('output')
