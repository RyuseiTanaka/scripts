import json
from ubteacher.data.build import divide_label_unlabel
from detectron2.data.build import get_detection_dataset_dicts
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_bboxAreas(target_boxes):
    assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

    tgt_l = target_boxes[:, 0]
    tgt_r = target_boxes[:, 2]
    tgt_d = target_boxes[:, 1]
    tgt_u = target_boxes[:, 3]

    tgt_widths = abs(tgt_r - tgt_l) + 1.0
    tgt_heights = abs(tgt_d - tgt_u) + 1.0

    tgt_areas = tgt_widths * tgt_heights

    return tgt_areas

def get_bboxAreas_class(label_dicts, cls_num):
    areas = []
    for num in range(cls_num):
        bbox_dict = get_bboxdicts(
            label_dicts,
            category_id=num,
            flag=True
            )
        area = get_bboxAreas(torch.tensor(bbox_dict))
        areas.append(area)
    return areas

def get_bboxdicts(label_dicts, category_id=None, flag=False):
    bbox_dicts = []
    for label in label_dicts:
        annotations = label["annotations"]
        for annotation in annotations:
            if flag:
                if annotation["category_id"] == category_id:
                    bbox = annotation["bbox"]
                    bbox_dicts.append(bbox)
                else:
                    pass
            else:
                bbox = annotation["bbox"]
                bbox_dicts.append(bbox)
    return bbox_dicts

def get_reference_area_sort(areas, percent):
    areas_sort, _ = torch.sort(areas)
    cnt = len(areas)
    idx = int(cnt/100*percent)
    if idx < 1:
        idx = 1
    area = areas_sort[0:idx]
    reference_area = torch.mean(area)
    reference_area = reference_area.item()
    return reference_area

def get_reference_area_class(areas, percent):
    reference_area = []
    for area in areas:
        Ar = get_reference_area_sort(area, percent)
        reference_area.append(Ar)
    return torch.tensor(reference_area)

def get_cls_gamma(areas):
    sample = []
    for i in range(9):
        d = areas[i]
        sample.append(len(d))
    rmin = min(sample)
    gamma = []
    for i in range(9):
        r = rmin/sample[i]
        gamma.append(r)
    gamma.append(1.0)
    gamma = torch.tensor(gamma)
    return gamma

def get_cfg2reference_area(cfg):
    dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

    label_dicts, unlabel_dicts = divide_label_unlabel(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )
    class_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    areas = get_bboxAreas_class(label_dicts, class_num)

    reference_area = get_reference_area_class(areas, cfg.FOW.REFERENCE_PERCENT)
    gamma = get_cls_gamma(areas)
    
    return reference_area, gamma        
      