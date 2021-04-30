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
        self.model.load_darknet_weights("./checkpoint/darknet53_448.weights")

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr_start, momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)
        self.scheduler = adjust_lr_by_wave(self.optimizer, self.max_epoch * self.len_train_dataset, cfg.lr_start,
                                           cfg.lr_end, cfg.warmup)
        # self.scheduler = adjust_lr_by_loss(self.optimizer,cfg.lr_start,cfg.warmup,self.train_dataloader.num_batches)
        self.writer = SummaryWriter(cfg.tensorboard_path)
        self.iter = 0

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

    def train(self):
        for epoch_index in range(self.max_epoch):
            mean_loss = [0, 0, 0, 0]
            self.model.train()
            for self.iter, train_data in enumerate(self.train_dataloader):
                start_time = time.time()
                self.scheduler.step(epoch_index,
                                    self.len_train_dataset * epoch_index + self.iter / cfg.batch_size)  # 调整学习率
                # self.scheduler.step(self.len_train_dataset * epoch_index + self.iter + 1,mean_loss[0])
                image, target, _ = train_data

                image = Variable(image).to(self.device)
                # target = Variable(target).to(self.device)

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

            if (epoch_index + 1) % cfg.save_step == 0:
                checkpoint = {'epoch': epoch_index,
                              'model': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict()}
                torch.save(self.model.state_dict(), cfg.checkpoint_save_path + str(epoch_index + 1) + '.pth')

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
        train_loss = []
        learning_rate = []

        for i, train_data in enumerate(self.train_dataloader):
            image, target, logit = train_data

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
        if len(mAP_list) != 0:
            from plot_curve import plot_map
            plot_map(mAP_list)


if __name__ == '__main__':
    trainer().train()
