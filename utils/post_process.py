import cv2
import os
import torch

from config.yolov3 import cfg


def reorginalize_mask(mask, logit, image_size, cocoGt):
    img_id = logit[0]['image_id'].item()
    img_ann = cocoGt.loadImgs(ids=img_id)[0]  # 一次只读取一张图片，返回是一个列表，取列表的第一个元素
    img = cv2.imread(os.path.join(cfg.image_path, img_ann['file_name']))
    pad_x = max(img.shape[0] - img.shape[1], 0) * (image_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (image_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = image_size - pad_y
    unpad_w = image_size - pad_x

    P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y = mask
    P1_y = max(round(((P1_y - pad_y // 2) / unpad_h) * img.shape[0]), 0)
    P1_x = max(round(((P1_x - pad_x // 2) / unpad_w) * img.shape[1]), 0)
    P2_y = max(round(((P2_y - pad_y // 2) / unpad_h) * img.shape[0]), 0)
    P2_x = max(round(((P2_x - pad_x // 2) / unpad_w) * img.shape[1]), 0)
    P3_y = max(round(((P3_y - pad_y // 2) / unpad_h) * img.shape[0]), 0)
    P3_x = max(round(((P3_x - pad_x // 2) / unpad_w) * img.shape[1]), 0)
    P4_y = max(round(((P4_y - pad_y // 2) / unpad_h) * img.shape[0]), 0)
    P4_x = max(round(((P4_x - pad_x // 2) / unpad_w) * img.shape[1]), 0)

    new_mask = [P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y]
    return new_mask


def reorginalize_target(detections, logit, image_size, cocoGt):
    output = []
    for detection in detections:
        anns = detection.chunk(detection.shape[0], dim=0)
        assert len(anns) > 0
        for ann in anns:
            mask = ann[0, :8].tolist()
            mask = reorginalize_mask(mask, logit, image_size, cocoGt)
            scores = ann[0, 8].item()
            labels = torch.argmax(ann[0, 9:]).item()
            img_id = logit[0]['image_id'].item()
            output.append({'image_id': img_id, 'category_id': labels, 'segmentation': [mask], 'score': scores})
    return output
