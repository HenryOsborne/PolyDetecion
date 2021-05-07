import torch
import torch.nn as nn
from config.yolov3 import cfg
from utils.build_target import build_targets


def box_iou(box1, box2, giou=False):
    '''
    :param box1: shape(...,(x,y,w,h))
    :param box2: shape(...,(x,y,w,h))
    :param giou: 是否计算giou
    :return:
        iou or giou
    '''
    box1_area = box1[..., 2:3] * box1[..., 3:4]  # 计算box1的面积
    box2_area = box2[..., 2:3] * box2[..., 3:4]  # 计算box2的面积

    # 转为x1,y1,x2,y2
    box1 = torch.cat((box1[..., 0:2] - box1[..., 2:] * 0.5, box1[..., 0:2] + box1[..., 2:] * 0.5), dim=-1)
    box2 = torch.cat((box2[..., 0:2] - box2[..., 2:] * 0.5, box2[..., 0:2] + box2[..., 2:] * 0.5), dim=-1)

    left_up = torch.max(box1[..., :2], box2[..., :2])  # 求两个box的左上角顶点的最右下角点
    right_down = torch.min(box1[..., 2:], box2[..., 2:])  # 求两个box的右下角顶点的最左上角点

    inter = right_down - left_up
    zero = torch.zeros((inter.shape), dtype=torch.float32).to(torch.device(cfg.device))
    inter = torch.max(inter, zero)  # 计算横向和纵向的交集长度,如果没有交集则为0
    inter = inter[..., 0:1] * inter[..., 1:2]  # 计算交集面积

    union = box1_area + box2_area - inter  # 计算并集面积
    iou = 1.0 * inter / union  # iou = 交集/并集

    if giou:
        left_up = torch.min(box1[..., :2], box2[..., :2])  # 求两个box的左上角顶点的最左上角点
        right_down = torch.max(box1[..., 2:], box2[..., 2:])  # 求两个box的右下角顶点的最右下角点
        area_c = right_down - left_up
        area_c = area_c[..., 0:1] * area_c[..., 1:2]  # 计算两个box的最小外接矩形的面积
        giou = iou - (area_c - union) / area_c
        return giou
    else:
        return iou


def Focal_loss(input, target, gamma, alpha):
    BCE = nn.BCEWithLogitsLoss(reduction='none')
    loss = BCE(input, target)
    loss *= alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), gamma)
    return loss


def BCE_loss(input, target):
    BCE = nn.BCEWithLogitsLoss()  # 计算交叉熵之前会对input做sigmoid,所以不用提前经过sigmoid
    loss = BCE(input, target)
    return loss


def Cross_Entropy_Loss(input, target):
    Cross = nn.CrossEntropyLoss()
    loss = Cross(input, target)
    return loss


def Smooth_L1_loss(input, targrt):
    smooth = torch.nn.SmoothL1Loss()
    loss = smooth(input, targrt)
    return loss


def loss_layer(output, target, stride, anchors):
    '''
    :param output: yolo output(n,grid size,grid size,num anchors,9+num classes)
    :param pred: yolo output before decode(n,grid size,grid size,num anchors,9+num classes)
    :param label_mask: shape same as output and pred
    :param label_xywh: (max num of boxes every scale,(x,y,w,h))
    :param stride: input size//putput size
    :return:
        loss,giou_loss,conf_loss,cls_loss
    '''
    batch_size, _, output_size = output.shape[0:3]  # batch_size和yolo输出的大小
    num_anchors = len(anchors)

    device = torch.device(cfg.device)
    anchors = anchors.to(device)

    output = output.view(batch_size, num_anchors, cfg.num_classes + 9, output_size, output_size)
    output = output.permute(0, 1, 3, 4, 2).contiguous()  # (batch size,3,output size,output size,9+num classes)

    output_xy = output[..., 0:8]
    output_conf = output[..., 8]
    output_cls = output[..., 9:]

    t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls = \
        build_targets(target, anchors, num_anchors, cfg.num_classes, output_size)
    tcls = tcls[mask]
    t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls = \
        t1_x.to(device), t1_y.to(device), t2_x.to(device), t2_y.to(device), t3_x.to(device), t3_y.to(device), t4_x.to(
            device), t4_y.to(device), mask.to(device), tcls.to(device)
    num_pred = mask.sum().float()  # Number of anchors (assigned to targets)

    k = num_pred / batch_size

    if num_pred > 0:
        lx1 = (k) * Smooth_L1_loss(output_xy[..., 0][mask], t1_x[mask]) / 8
        ly1 = (k) * Smooth_L1_loss(output_xy[..., 1][mask], t1_y[mask]) / 8
        lx2 = (k) * Smooth_L1_loss(output_xy[..., 2][mask], t2_x[mask]) / 8
        ly2 = (k) * Smooth_L1_loss(output_xy[..., 3][mask], t2_y[mask]) / 8
        lx3 = (k) * Smooth_L1_loss(output_xy[..., 4][mask], t3_x[mask]) / 8
        ly3 = (k) * Smooth_L1_loss(output_xy[..., 5][mask], t3_y[mask]) / 8
        lx4 = (k) * Smooth_L1_loss(output_xy[..., 6][mask], t4_x[mask]) / 8
        ly4 = (k) * Smooth_L1_loss(output_xy[..., 7][mask], t4_y[mask]) / 8

        conf_loss = (k * 10) * BCE_loss(output_conf, mask.float())
        cls_loss = (k / cfg.num_classes) * Cross_Entropy_Loss(output_cls[mask], torch.argmax(tcls, 1))
    else:
        lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, conf_loss, cls_loss = \
            [torch.FloatTensor([0]).requires_grad_(True).to(device) for _ in range(10)]

    loc_loss = lx1 + ly1 + lx2 + ly2 + lx3 + ly3 + lx4 + ly4

    loss = loc_loss + conf_loss + cls_loss

    return loss, loc_loss, conf_loss, cls_loss


def build_loss(output, target):
    anchors = torch.FloatTensor(cfg.anchors)

    # 计算每种scale的loss
    loss_small = loss_layer(output[0], target, cfg.strides[0], anchors[0])
    loss_middle = loss_layer(output[1], target, cfg.strides[1], anchors[1])
    loss_big = loss_layer(output[2], target, cfg.strides[2], anchors[2])

    giou_loss = loss_small[1] + loss_middle[1] + loss_big[1]
    conf_loss = loss_small[2] + loss_middle[2] + loss_big[2]
    cls_loss = loss_small[3] + loss_middle[3] + loss_big[3]
    loss = loss_small[0] + loss_middle[0] + loss_big[0]

    return loss, giou_loss, conf_loss, cls_loss


if __name__ == '__main__':
    output = torch.randn((1, 13, 13, 3, 6)).to(torch.device(cfg.device))
    pred = torch.randn((1, 13, 13, 3, 6)).to(torch.device(cfg.device))
    mask = torch.randn((1, 13, 13, 3, 6)).to(torch.device(cfg.device))
    xywh = torch.randn((1, 150, 4)).to(torch.device(cfg.device))
    stride = 32
    loss, giou_loss, theta_loss, conf_loss, cls_loss = loss_layer(output, pred, mask, xywh, stride)
    print(loss, giou_loss, theta_loss, conf_loss, cls_loss)
