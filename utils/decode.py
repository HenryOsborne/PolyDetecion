# Editor       : pycharm
# File name    : utils/decode.py
# Author       : huangxinyu
# Created date : 2020-11-10
# Description  : 对yolov3的输出进行解码

import torch
from config.yolov3 import cfg


def transform(output):
    num_classes = cfg.num_classes
    batch_size, _, output_size, _ = output.shape
    num_anchors = len(cfg.anchors[0])
    output = output.permute(0, 2, 3, 1).contiguous()  # (batch size,3*(9+num classes),output size,output size)
    output = output.view(batch_size, output_size, output_size, num_anchors, num_classes + 9)

    return output


def build_decode(output):
    anchors = torch.FloatTensor(cfg.anchors)
    strides = torch.FloatTensor(cfg.strides)
    small_pred = decode(output[0], strides[0], anchors[0])
    middle_pred = decode(output[1], strides[1], anchors[1])
    big_pred = decode(output[2], strides[2], anchors[2])
    pred = torch.cat((small_pred.view(-1, 9 + cfg.num_classes), middle_pred.view(-1, 9 + cfg.num_classes),
                      big_pred.view(-1, 9 + cfg.num_classes)), dim=0)

    return pred


def decode(output, stride, anchors):
    batch_size, _, output_size = output.shape[0:3]
    output = output.view(batch_size, 3, _ // 3, output_size, output_size).permute(0, 1, 3, 4, 2).contiguous()
    grid_x = torch.arange(output_size).repeat(output_size, 1).view([1, 1, output_size, output_size]).float().cuda()
    grid_y = torch.arange(output_size).repeat(output_size, 1).t().view([1, 1, output_size, output_size]).float().cuda()
    P1_x = output[..., 0]  # Point1 x
    P1_y = output[..., 1]  # Point1 y
    P2_x = output[..., 2]  # Point2 x
    P2_y = output[..., 3]  # Point2 y
    P3_x = output[..., 4]  # Point3 x
    P3_y = output[..., 5]  # Point3 y
    P4_x = output[..., 6]  # Point4 x
    P4_y = output[..., 7]  # Point4 y

    pred_boxes = torch.FloatTensor(batch_size, 3, output_size, output_size, 8).cuda()
    pred_conf = output[..., 8]  # Conf
    pred_cls = output[..., 9:]  # Class
    pred_boxes[..., 0] = P1_x + grid_x
    pred_boxes[..., 1] = P1_y + grid_y
    pred_boxes[..., 2] = P2_x + grid_x
    pred_boxes[..., 3] = P2_y + grid_y
    pred_boxes[..., 4] = P3_x + grid_x
    pred_boxes[..., 5] = P3_y + grid_y
    pred_boxes[..., 6] = P4_x + grid_x
    pred_boxes[..., 7] = P4_y + grid_y

    pred = torch.cat((pred_boxes.view(batch_size, -1, 8) * stride,
                      torch.sigmoid(pred_conf.view(batch_size, -1, 1)), pred_cls.view(batch_size, -1, cfg.num_classes)),
                     -1)
    return pred
