import numpy as np
import torch
from shapely.geometry import Polygon
import shapely
from config.yolov3 import cfg


def reorganize_targets(t, nTb):
    t = t.float()
    tc = t[:, 0]
    t = t[:, 1:9].view(nTb, 4, 2)

    x = t[..., 0]
    y = t[..., 1]
    y_sorted, y_indices = torch.sort(y)

    decice = torch.device(cfg.device)
    x_sorted = torch.zeros(nTb, 4).to(decice)
    for i in range(0, nTb):
        x_sorted[i] = x[i, y_indices[i]]

    x_sorted[:, :2], x_top_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_bottom_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    for i in range(0, nTb):
        y_sorted[i, :2] = y_sorted[i, :2][x_top_indices[i]]
        y_sorted[i, 2:4] = y_sorted[i, 2:4][x_bottom_indices[i]]

    t = torch.cat((x_sorted[:, 0].unsqueeze(1), y_sorted[:, 0].unsqueeze(1), x_sorted[:, 1].unsqueeze(1),
                   y_sorted[:, 1].unsqueeze(1),
                   x_sorted[:, 2].unsqueeze(1), y_sorted[:, 2].unsqueeze(1), x_sorted[:, 3].unsqueeze(1),
                   y_sorted[:, 3].unsqueeze(1)), 1)

    return torch.cat((tc.unsqueeze(1), t), 1)

def build_targets(target, anchor_wh, nA, nC, nG):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch
    nT = [len(x) for x in target]  # torch.argmin(target[:, :, 4], 1)  # targets per batch
    t1_x = torch.zeros(nB, nA, nG, nG)  # batch size (4), number of anchors (3), number of grid points (13)
    t1_y = torch.zeros(nB, nA, nG, nG)
    t2_x = torch.zeros(nB, nA, nG, nG)
    t2_y = torch.zeros(nB, nA, nG, nG)
    t3_x = torch.zeros(nB, nA, nG, nG)
    t3_y = torch.zeros(nB, nA, nG, nG)
    t4_x = torch.zeros(nB, nA, nG, nG)
    t4_y = torch.zeros(nB, nA, nG, nG)
    tconf = torch.BoolTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # nC = number of classes

    FN = torch.ByteTensor(nB, max(nT)).fill_(0)
    TC = torch.ShortTensor(nB, max(nT)).fill_(-1)  # target category
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    for b in range(nB):
        nTb = nT[b]  # number of targets per image
        if nTb == 0:
            continue
        t = target[b].to(device)
        FN[b, :nTb] = 1
        t = reorganize_targets(t, nTb)
        # Convert to position relative to box
        TC[b, :nTb] = t[:, 0].long()
        gp1_x, gp1_y = t[:, 1] * nG, t[:, 2] * nG
        gp2_x, gp2_y = t[:, 3] * nG, t[:, 4] * nG
        gp3_x, gp3_y = t[:, 5] * nG, t[:, 6] * nG
        gp4_x, gp4_y = t[:, 7] * nG, t[:, 8] * nG
        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        gp1_i, gp1_j = torch.clamp(torch.round(gp1_x).long(), min=0, max=nG - 1), torch.clamp(torch.round(gp1_y).long(), min=0, max=nG - 1)
        gp2_i, gp2_j = torch.clamp(torch.round(gp2_x).long(), min=0, max=nG - 1), torch.clamp(torch.round(gp2_y).long(), min=0, max=nG - 1)
        gp3_i, gp3_j = torch.clamp(torch.round(gp3_x).long(), min=0, max=nG - 1), torch.clamp(torch.round(gp3_y).long(), min=0, max=nG - 1)
        gp4_i, gp4_j = torch.clamp(torch.round(gp4_x).long(), min=0, max=nG - 1), torch.clamp(torch.round(gp4_y).long(), min=0, max=nG - 1)

        box1 = t[:, 1:9] * nG

        # Get each target center
        gp_x = torch.cat((gp1_x.unsqueeze(1), gp2_x.unsqueeze(1), gp3_x.unsqueeze(1), gp4_x.unsqueeze(1)), 1).view(-1, 4)
        gp_y = torch.cat((gp1_y.unsqueeze(1), gp2_y.unsqueeze(1), gp3_y.unsqueeze(1), gp4_y.unsqueeze(1)), 1).view(-1, 4)

        gp_x_min = torch.min(gp_x, 1)[0]
        gp_x_max = torch.max(gp_x, 1)[0]
        gp_y_min = torch.min(gp_y, 1)[0]
        gp_y_max = torch.max(gp_y, 1)[0]

        gp_x_center = torch.mean(torch.cat((gp_x_min.unsqueeze(1), gp_x_max.unsqueeze(1)), 1), 1)
        gp_y_center = torch.mean(torch.cat((gp_y_min.unsqueeze(1), gp_y_max.unsqueeze(1)), 1), 1)
        # Set target center in a certain cell
        gp_x_center = torch.round(gp_x_center)
        gp_y_center = torch.round(gp_y_center)

        x_min = torch.clamp \
            ((gp_x_center.unsqueeze(1).repeat(1, nA).view(-1, nA, 1) - anchor_wh[:, 0].view(-1, nA, 1) / 2), min=0, max=nG -1)
        x_max = torch.clamp \
            ((gp_x_center.unsqueeze(1).repeat(1, nA).view(-1, nA, 1) + anchor_wh[:, 0].view(-1, nA, 1) / 2), min=0, max=nG -1)
        y_min = torch.clamp \
            ((gp_y_center.unsqueeze(1).repeat(1, nA).view(-1, nA, 1) - anchor_wh[:, 1].view(-1, nA, 1) / 2), min=0, max=nG -1)
        y_max = torch.clamp \
            ((gp_y_center.unsqueeze(1).repeat(1, nA).view(-1, nA, 1) + anchor_wh[:, 1].view(-1, nA, 1) / 2), min=0, max=nG -1)

        top_left = torch.cat((x_min.view(-1, 1), y_min.view(-1, 1)), 1)
        top_right = torch.cat((x_max.view(-1, 1), y_min.view(-1, 1)), 1)
        bottom_right = torch.cat((x_max.view(-1, 1), y_max.view(-1, 1)), 1)
        bottom_left = torch.cat((x_min.view(-1, 1), y_max.view(-1, 1)), 1)
        # Get bounding boxes
        box2 = torch.cat((top_left, top_right, bottom_right, bottom_left), 1).view(-1, nA, 8)

        iou_anch = torch.zeros(nA, nTb, 1)
        for i in range(0, nTb):
            for j in range(0, nA):
                polygon1 = Polygon(box1[i ,:].view(4 ,2)).convex_hull
                polygon2 = Polygon(box2[i ,j ,:].view(4 ,2)).convex_hull
                if polygon1.intersects(polygon2):
                    try:
                        inter_area = polygon1.intersection(polygon2).area
                        union_area = polygon1.union(polygon2).area
                        iou_anch[j, i] =  inter_area / union_area
                    except shapely.geos.TopologicalError:
                        print('shapely.geos.TopologicalError occured, iou set to 0')

        iou_anch = iou_anch.squeeze(2)
        # Select best iou_pred and anchor
        iou_anch_best, a = iou_anch.max(0)  # best anchor [0-2] for each target
        a = a.to(device)

        # Select best unique target-anchor combinations
        if nTb > 1:
            iou_order = np.argsort(-iou_anch_best)  # best to worst

            # Unique anchor selection (slower but retains original order)

            u = torch.cat((gp1_i, gp1_j, gp2_i, gp2_j, gp3_i, gp3_j, gp4_i, gp4_j, a), 0).view(-1, nTb).cpu().numpy()
            _, first_unique = np.unique(u[:, iou_order], axis=1, return_index=True)  # first unique indices; each cell response to on target
            i = iou_order[first_unique]

            # best anchor must share significant commonality (iou) with target
            i = i[iou_anch_best[i] > 0.1]
            if len(i) == 0:
                continue

            a, gp1_i, gp1_j, gp2_i, gp2_j, gp3_i, gp3_j, gp4_i, gp4_j, t = \
                a[i], gp1_i[i], gp1_j[i], gp2_i[i], gp2_j[i], gp3_i[i], gp3_j[i], gp4_i[i], gp4_j[i], t[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_anch_best < 0.1:
                continue
            i = 0

        tc = t[:, 0].long()
        gp1_x, gp1_y = t[:, 1] * nG, t[:, 2] * nG
        gp2_x, gp2_y = t[:, 3] * nG, t[:, 4] * nG
        gp3_x, gp3_y = t[:, 5] * nG, t[:, 6] * nG
        gp4_x, gp4_y = t[:, 7] * nG, t[:, 8] * nG

        # Get target center
        gp_x = torch.cat((gp1_x.unsqueeze(1), gp2_x.unsqueeze(1), gp3_x.unsqueeze(1), gp4_x.unsqueeze(1)), 1).view(-1, 4)
        gp_y = torch.cat((gp1_y.unsqueeze(1), gp2_y.unsqueeze(1), gp3_y.unsqueeze(1), gp4_y.unsqueeze(1)), 1).view(-1, 4)
        gp_x_min = torch.min(gp_x, 1)[0]
        gp_x_max = torch.max(gp_x, 1)[0]
        gp_y_min = torch.min(gp_y, 1)[0]
        gp_y_max = torch.max(gp_y, 1)[0]
        gp_x_center = torch.mean(torch.cat((gp_x_min.unsqueeze(1), gp_x_max.unsqueeze(1)), 1), 1)
        gp_y_center = torch.mean(torch.cat((gp_y_min.unsqueeze(1), gp_y_max.unsqueeze(1)), 1), 1)
        # Set target center in a certain cell
        gp_x_center = torch.round(gp_x_center).long()
        gp_y_center = torch.round(gp_y_center).long()

        # Coordinates
        t1_x[b, a, gp_y_center, gp_x_center] = gp1_x.cpu() - gp_x_center.float().cpu()
        t1_y[b, a, gp_y_center, gp_x_center] = gp1_y.cpu() - gp_y_center.float().cpu()
        t2_x[b, a, gp_y_center, gp_x_center] = gp2_x.cpu() - gp_x_center.float().cpu()
        t2_y[b, a, gp_y_center, gp_x_center] = gp2_y.cpu() - gp_y_center.float().cpu()
        t3_x[b, a, gp_y_center, gp_x_center] = gp3_x.cpu() - gp_x_center.float().cpu()
        t3_y[b, a, gp_y_center, gp_x_center] = gp3_y.cpu() - gp_y_center.float().cpu()
        t4_x[b, a, gp_y_center, gp_x_center] = gp4_x.cpu() - gp_x_center.float().cpu()
        t4_y[b, a, gp_y_center, gp_x_center] = gp4_y.cpu() - gp_y_center.float().cpu()

        # One-hot encoding of label
        tcls[b, a, gp_y_center, gp_x_center, tc] = 1
        tconf[b, a, gp_y_center, gp_x_center] = 1

    return t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, tconf, tcls
