import torch
from scripts.area_process import get_bboxAreas

def FOW(bbox, classes, reference_area, gamma):
    area_weights = []
    areas = get_bboxAreas(bbox)
    reference_area = reference_area.to(areas.device)
    gamma = gamma.to(areas.device)
    for area, cls_idx in zip(areas, classes):
        if cls_idx == 10:
            Aw = 1.0
        elif area <= reference_area[cls_idx]:
            Aw = gamma[cls_idx]*torch.log(reference_area[cls_idx] / area) + 1.0
        else:
            Aw = 1.0
        area_weights.append(Aw)
    return torch.tensor(area_weights)

def fow_loss(loss, w):
    if loss.numel() != 0:
        if w.device != loss.device:
            w = w.to(loss.device)
        loss = w * loss
        return loss.sum()
    else:
        return loss.sum()