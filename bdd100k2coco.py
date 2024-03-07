import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()

src_val_dir = os.path.join(cfg.bdd_dir, 'labels', 'det_20','det_val.json')
src_train_dir = os.path.join(cfg.bdd_dir, 'labels','det_20', 'det_train.json')

os.makedirs(os.path.join(cfg.bdd_dir, 'labels_coco'), exist_ok=True)

dst_val_dir = os.path.join(cfg.bdd_dir, 'labels_coco', 'bdd100k_labels_images_val_coco.json')
dst_train_dir = os.path.join(cfg.bdd_dir, 'labels_coco', 'bdd100k_labels_images_train_coco.json')


def bdd2coco_detection(labeled_images, save_dir):
    attr_dict = {
        "categories": [
            {"supercategory": "none", "id": 1, "name": "pedestrian"},
            {"supercategory": "none", "id": 2, "name": "car"},
            {"supercategory": "none", "id": 3, "name": "rider"},
            {"supercategory": "none", "id": 4, "name": "bus"},
            {"supercategory": "none", "id": 5, "name": "truck"},
            {"supercategory": "none", "id": 6, "name": "bicycle"},
            {"supercategory": "none", "id": 7, "name": "motorcycle"},
            {"supercategory": "none", "id": 8, "name": "traffic light"},
            {"supercategory": "none", "id": 9, "name": "traffic sign"},
            {"supercategory": "none", "id": 10, "name": "train"},
        ]
    }

    id_dict = {category["name"]: category["id"] for category in attr_dict["categories"]}

    images = []
    annotations = []
    ignore_categories = set()

    counter = 0
    for image_info in tqdm(labeled_images):
        counter += 1
        image = {
            "file_name": image_info["name"],
            "height": 720,
            "width": 1280,
            "id": counter,
        }

        empty_image = True
        tmp = 0
        if "labels" in image_info:
            for label_info in image_info["labels"]:
                if "category" in label_info:
                    category = label_info["category"]
                    if category in id_dict:
                        tmp = 1
                        empty_image = False
                        annotation = {
                            "iscrowd": 0,
                            "image_id": image["id"],
                            "bbox": [
                                label_info["box2d"]["x1"],
                                label_info["box2d"]["y1"],
                                label_info["box2d"]["x2"] - label_info["box2d"]["x1"],
                                label_info["box2d"]["y2"] - label_info["box2d"]["y1"],
                            ],
                            "area": float(
                                (label_info["box2d"]["x2"] - label_info["box2d"]["x1"])
                                * (label_info["box2d"]["y2"] - label_info["box2d"]["y1"])
                            ),
                            "category_id": id_dict[category],
                            "ignore": 0,
                            "id": label_info["id"],
                            "segmentation": [
                                [
                                    label_info["box2d"]["x1"],
                                    label_info["box2d"]["y1"],
                                    label_info["box2d"]["x1"],
                                    label_info["box2d"]["y2"],
                                    label_info["box2d"]["x2"],
                                    label_info["box2d"]["y2"],
                                    label_info["box2d"]["x2"],
                                    label_info["box2d"]["y1"],
                                ]
                            ],
                        }
                        annotations.append(annotation)
                    else:
                        ignore_categories.add(category)

        if empty_image:
            print("Empty image!")
            continue
        if tmp == 1:
            images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print("Ignored categories:", ignore_categories)
    print("Saving...")
    with open(save_dir, "w") as file:
        json.dump(attr_dict, file)
    print("Done.")


def main():
    # Create BDD training set detections in COCO format
    print("Loading training set...")
    with open(src_train_dir) as f:
        train_labels = json.load(f)
    print("Converting training set...")
    bdd2coco_detection(train_labels, dst_train_dir)

    # Create BDD validation set detections in COCO format
    print("Loading validation set...")
    with open(src_val_dir) as f:
        val_labels = json.load(f)
    print("Converting validation set...")
    bdd2coco_detection(val_labels, dst_val_dir)


if __name__ == "__main__":
    main()