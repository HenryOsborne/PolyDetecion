# Editor       : pycharm
# File name    : train_v3.py
# Author       : huangxinyu
# Created date : 2020-11-10
# Description  : 训练
import cv2
import torch
from torch.autograd import Variable
from models.yolov3 import yolov3
from config.yolov3 import cfg
import math
import json
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.loss import build_loss
from utils.scheduler import adjust_lr_by_wave
from utils.nms import non_max_suppression

from load_data import NewDataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from eval_utils.coco_utils import CocoEvaluator
from eval_utils.coco_utils import create_coco_dataset


class trainer(object):
    def __init__(self):
        self.device = torch.device(cfg.device)

        self.val_dataset = NewDataset(train_set=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=True,
                                         num_workers=cfg.num_worker,
                                         collate_fn=self.val_dataset.collate_fn)

        self.len_train_dataset = len(self.val_dataset)

        self.model = yolov3().to(self.device)
        weights_path = 'checkpoint/180.pt'
        checkpoint = torch.load(weights_path)
        self.model.load_state_dict(checkpoint)

        self.cocoGt = COCO(cfg.test_json)

    def reorginalize_mask(self, mask, logit, image_size):
        img_id = logit[0]['image_id'].item()
        img_ann = self.cocoGt.loadImgs(ids=img_id)[0]  # 一次只读取一张图片，返回是一个列表，取列表的第一个元素
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

    def reorginalize_target(self, detections, logit, image_size):
        output = []
        for detection in detections:
            anns = detection.chunk(detection.shape[0], dim=0)
            assert len(anns) > 0
            for ann in anns:
                mask = ann[0, :8].tolist()
                mask = self.reorginalize_mask(mask, logit, image_size)
                scores = ann[0, 8].item()
                labels = torch.argmax(ann[0, 9:]).item()
                img_id = logit[0]['image_id'].item()
                output.append({'image_id': img_id, 'category_id': labels, 'segmentation': [mask], 'score': scores})
        return output

    @torch.no_grad()
    def eval(self):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(n_threads)
        cpu_device = torch.device("cpu")
        self.model.eval()

        for ann_idx in self.cocoGt.anns:
            ann = self.cocoGt.anns[ann_idx]
            ann['area'] = maskUtils.area(self.cocoGt.annToRLE(ann))

        iou_types = 'segm'
        anns = []

        for val_data in self.val_dataloader:
            image, target, logit = val_data

            image = image.to(self.device)
            image_size = image.shape[3]  # image.shape[2]==image.shape[3]
            # resize之后图像的大小

            _, pred = self.model(image)
            # TODO:当前只支持batch_size=1
            pred = pred.unsqueeze(0)
            pred = pred[pred[:, :, 8] > cfg.conf_thresh]
            detections = non_max_suppression(pred.unsqueeze(0), cls_thres=cfg.cls_thresh, nms_thres=cfg.conf_thresh)

            anns.extend(self.reorginalize_target(detections, logit, image_size))

        for ann in anns:
            ann['segmentation'] = self.cocoGt.annToRLE(ann)  # 将polygon形式的segmentation转换RLE形式

        cocoDt = self.cocoGt.loadRes(anns)

        cocoEval = COCOeval(self.cocoGt, cocoDt, iou_types)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


if __name__ == '__main__':
    trainer().eval()
