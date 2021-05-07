import torch
from models.yolov3 import yolov3
from config.yolov3 import cfg
import math
import time
import cv2
import argparse
import os
import yaml
from tensorboardX import SummaryWriter
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from utils.loss import build_loss
from utils.scheduler import adjust_lr_by_wave
from load_data import NewDataset
from torch.utils.data import DataLoader
from utils.nms import non_max_suppression
from plot_curve import plot_map
from plot_curve import plot_loss_and_lr
from models.yolo import Model
from utils.torch_utils import intersect_dicts


def build_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='checkpoint/yolov5s.pt', help='initial weights path')  #
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')  #
    parser.add_argument('--hyp', type=str, default='config/hyp.scratch.yaml', help='hyperparameters path')  #
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    opt = parser.parse_args()

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    nc = cfg.num_classes
    device = torch.device(cfg.device)
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    return model


class _Trainer(object):
    def __init__(self):
        self.device = torch.device(cfg.device)
        self.max_epoch = cfg.max_epoch

        self.train_dataset = NewDataset(train_set=True)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.batch_size,
                                           shuffle=True,
                                           num_workers=cfg.num_worker,
                                           collate_fn=self.train_dataset.collate_fn)

        self.val_dataset = NewDataset(train_set=False)
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=0,
                                         collate_fn=self.val_dataset.collate_fn)

        self.len_train_dataset = len(self.train_dataset)

        self.model = build_model()

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=cfg.lr_start,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)
        self.scheduler = adjust_lr_by_wave(self.optimizer, self.max_epoch * self.len_train_dataset, cfg.lr_start,
                                           cfg.lr_end, cfg.warmup)
        # self.scheduler = adjust_lr_by_loss(self.optimizer,cfg.lr_start,cfg.warmup,self.train_dataloader.num_batches)
        self.writer = SummaryWriter(cfg.tensorboard_path)
        self.iter = 0
        self.cocoGt = COCO(cfg.test_json)

    def put_log(self, epoch_index, mean_loss, time_per_iter):
        print("[epoch:{}|{}] [iter:{}|{}] time:{}s loss:{} giou_loss:{} conf_loss:{} cls_loss:{} lr:{}".format(
            epoch_index + 1, self.max_epoch,
            self.iter + 1, math.ceil(self.len_train_dataset / cfg.batch_size), round(time_per_iter, 2),
            round(mean_loss[0], 4), round(mean_loss[1], 4)
            , round(mean_loss[2], 4), round(mean_loss[3], 4),
            self.optimizer.param_groups[0]['lr']))

        step = epoch_index * math.ceil(self.len_train_dataset / cfg.batch_size) + self.iter
        self.writer.add_scalar("loss", mean_loss[0], global_step=step)
        self.writer.add_scalar("giou loss", mean_loss[1], global_step=step)
        self.writer.add_scalar("conf loss", mean_loss[2], global_step=step)
        self.writer.add_scalar("cls loss", mean_loss[3], global_step=step)
        self.writer.add_scalar("learning rate", self.optimizer.param_groups[0]['lr'], global_step=step)

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

    def train_one_epoch(self, epoch_index, train_loss=None, train_lr=None):
        mean_loss = [0, 0, 0, 0]
        self.model.train()
        for self.iter, train_data in enumerate(self.train_dataloader):
            start_time = time.time()
            self.scheduler.step(epoch_index,
                                self.len_train_dataset * epoch_index + self.iter / cfg.batch_size)  # 调整学习率
            # self.scheduler.step(self.len_train_dataset * epoch_index + self.iter + 1,mean_loss[0])
            image, target, _ = train_data
            image = image.to(self.device)

            output, pred = self.model(image)

            # 计算loss
            loss, loss_giou, loss_conf, loss_cls = build_loss(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            end_time = time.time()
            time_per_iter = end_time - start_time  # 每次迭代所花时间

            loss_items = [loss.item(), loss_giou.item(), loss_conf.item(), loss_cls.item()]
            mean_loss = [(mean_loss[i] * self.iter + loss_items[i]) / (self.iter + 1) for i in range(4)]
            self.put_log(epoch_index, mean_loss, time_per_iter)

            # 记录训练损失
            loss_value = round(mean_loss[0], 4)
            if isinstance(train_loss, list):
                train_loss.append(loss_value)

            now_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(train_lr, list):
                train_lr.append(now_lr)

        if (epoch_index + 1) % cfg.save_step == 0:
            checkpoint = {'epoch': epoch_index,
                          'model': self.model.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
            torch.save(self.model.state_dict(), cfg.checkpoint_save_path + str(epoch_index + 1) + '.pth')

    @torch.no_grad()
    def eval(self, mAP_list=None):
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
            if pred.shape[0] == 0:
                pass
            else:
                detections = non_max_suppression(pred.unsqueeze(0), cls_thres=cfg.cls_thresh, nms_thres=cfg.conf_thresh)
                anns.extend(self.reorginalize_target(detections, logit, image_size))

        for ann in anns:
            ann['segmentation'] = self.cocoGt.annToRLE(ann)  # 将polygon形式的segmentation转换RLE形式

        cocoDt = self.cocoGt.loadRes(anns)

        cocoEval = COCOeval(self.cocoGt, cocoDt, iou_types)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print_txt = cocoEval.stats
        coco_mAP = print_txt[0]
        voc_mAP = print_txt[1]
        if isinstance(mAP_list, list):
            mAP_list.append(voc_mAP)


if __name__ == '__main__':
    train_loss = []
    learning_rate = []
    val_mAP = []
    trainer = _Trainer()
    for epoch_index in range(cfg.max_epoch):
        trainer.train_one_epoch(epoch_index, train_loss=train_loss, train_lr=learning_rate)
        trainer.eval(mAP_list=val_mAP)

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_mAP) != 0:
        plot_map(val_mAP)
