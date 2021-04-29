# Editor       : pycharm
# File name    : train.py
# Author       : huangxinyu
# Created date : 2020-11-10
# Description  : 训练

import torch
from torch.autograd import Variable
from models.yolov3 import yolov3
from config.yolov3 import cfg
import math
import time
from tensorboardX import SummaryWriter
from utils.loss import build_loss
from utils.scheduler import adjust_lr_by_wave

from load_data import NewDataset
from torch.utils.data import DataLoader
from eval_utils.coco_utils import CocoEvaluator
from eval_utils.coco_utils import create_coco_dataset
from eval_utils.coco_utils import reorginalize_target


class trainer(object):
    def __init__(self):
        self.device = torch.device(cfg.device)
        self.max_epoch = cfg.max_epoch

        self.train_dataset = NewDataset(train_set=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                           num_workers=cfg.num_worker,
                                           collate_fn=self.train_dataset.collate_fn)

        self.len_train_dataset = len(self.train_dataset)

        self.model = yolov3().to(self.device)

        weights_path = 'checkpoint/20.pth'
        checkpoint = torch.load(weights_path)

        self.model.load_state_dict(checkpoint)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr_start, momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)
        self.scheduler = adjust_lr_by_wave(self.optimizer, self.max_epoch * self.len_train_dataset, cfg.lr_start,
                                           cfg.lr_end, cfg.warmup)

    def eval(self):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(n_threads)
        cpu_device = torch.device("cpu")
        self.model.eval()

        coco = create_coco_dataset(self.train_dataset)
        iou_types = ["segm"]
        coco_evaluator = CocoEvaluator(coco, iou_types)
        mAP_list = []

        for i, train_data in enumerate(self.train_dataloader):
            image, target = train_data

            image = Variable(image).to(self.device)

            model_time = time.time()
            _, pred = self.model(image)

            pred, target = reorginalize_target(pred, target)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(target, outputs)}

            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time

        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)

        print_txt = coco_evaluator.coco_eval[iou_types[0]].stats
        coco_mAP = print_txt[0]
        voc_mAP = print_txt[1]
        if isinstance(mAP_list, list):
            mAP_list.append(voc_mAP)

        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate)

        # plot mAP curve
        if len(val_mAP) != 0:
            from plot_curve import plot_map
            plot_map(val_mAP)


if __name__ == '__main__':
    trainer().train()
