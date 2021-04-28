import numpy as np
import torch
from shapely.geometry import Polygon
import shapely
from config.yolov3 import cfg


def reorganize_targets(target, num_target):
    '''
    reorder the four coordinateds
    :param target:
    :param num_target:
    :return: sorted_target
    '''
    target = target.float()
    cls_of_target = target[:, 0]
    target = target[:, 1:9].view(num_target, 4, 2)

    x = target[..., 0]
    y = target[..., 1]
    y_sorted, y_indices = torch.sort(y)

    decice = torch.device(cfg.device)
    x_sorted = torch.zeros(num_target, 4).to(decice)
    for i in range(0, num_target):
        x_sorted[i] = x[i, y_indices[i]]

    x_sorted[:, :2], x_top_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_bottom_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    for i in range(0, num_target):
        y_sorted[i, :2] = y_sorted[i, :2][x_top_indices[i]]
        y_sorted[i, 2:4] = y_sorted[i, 2:4][x_bottom_indices[i]]

    sorted_target = torch.zeros_like(torch.cat((x_sorted, y_sorted), dim=1))
    sorted_target[:, 0::2] = x_sorted
    sorted_target[:, 1::2] = y_sorted

    return torch.cat((cls_of_target.unsqueeze(1), sorted_target), 1)


def build_targets(total_target, anchor_wh, num_anchor, num_class, output_size):
    """
    returns target_per_image, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    num_image = len(total_target)  # number of images in batch
    target_per_image = [len(x) for x in total_target]  # targets per batch
    # batch size (4), number of anchors (3), number of grid points (13)
    t = torch.zeros(num_image * 8, num_anchor, output_size, output_size)
    t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y = t.chunk(8)
    del t

    mask = torch.BoolTensor(num_image, num_anchor, output_size, output_size).fill_(0)
    tcls = torch.ByteTensor(num_image, num_anchor, output_size, output_size, num_class).fill_(0)
    target_category = torch.ShortTensor(num_image, max(target_per_image)).fill_(-1)  # target category
    device = torch.device(cfg.device)

    for img_idx in range(num_image):
        num_taget_cur_image = target_per_image[img_idx]  # number of targets per image
        if num_taget_cur_image == 0:
            continue
        target = total_target[img_idx].to(device)
        target = reorganize_targets(target, num_taget_cur_image)
        target_category[img_idx, :num_taget_cur_image] = target[:, 0].long()

        box1 = target[:, 1:9] * output_size

        # Convert to position relative to box
        gp1_x, gp1_y, gp2_x, gp2_y, gp3_x, gp3_y, gp4_x, gp4_y = box1.chunk(8, dim=1)

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        c_box = torch.clamp(torch.round(box1).long(), min=0, max=output_size - 1)
        gp1_i, gp1_j, gp2_i, gp2_j, gp3_i, gp3_j, gp4_i, gp4_j = c_box.chunk(8, dim=1)

        # Get each target center
        gp_x = torch.cat((gp1_x, gp2_x, gp3_x, gp4_x), 1).view(-1, 4)
        gp_y = torch.cat((gp1_y, gp2_y, gp3_y, gp4_y), 1).view(-1, 4)

        # min(x) max(x) min(y) max(y) in each instance
        gp_x_min = torch.min(gp_x, 1)[0]
        gp_x_max = torch.max(gp_x, 1)[0]
        gp_y_min = torch.min(gp_y, 1)[0]
        gp_y_max = torch.max(gp_y, 1)[0]

        # Set target center in a certain cell
        gp_x_center = torch.round(torch.mean((torch.stack((gp_x_min, gp_x_max), dim=1)), dim=1))
        gp_y_center = torch.round(torch.mean((torch.stack((gp_y_min, gp_y_max), dim=1)), dim=1))

        x_min = torch.clamp((gp_x_center.unsqueeze(1).repeat(1, num_anchor).view(-1, num_anchor, 1)
                             - anchor_wh[:, 0].view(-1, num_anchor, 1) / 2), min=0, max=output_size - 1)
        x_max = torch.clamp((gp_x_center.unsqueeze(1).repeat(1, num_anchor).view(-1, num_anchor, 1)
                             + anchor_wh[:, 0].view(-1, num_anchor, 1) / 2), min=0, max=output_size - 1)
        y_min = torch.clamp((gp_y_center.unsqueeze(1).repeat(1, num_anchor).view(-1, num_anchor, 1)
                             - anchor_wh[:, 1].view(-1, num_anchor, 1) / 2), min=0, max=output_size - 1)
        y_max = torch.clamp((gp_y_center.unsqueeze(1).repeat(1, num_anchor).view(-1, num_anchor, 1)
                             + anchor_wh[:, 1].view(-1, num_anchor, 1) / 2), min=0, max=output_size - 1)

        top_left = torch.cat((x_min.view(-1, 1), y_min.view(-1, 1)), 1)
        top_right = torch.cat((x_max.view(-1, 1), y_min.view(-1, 1)), 1)
        bottom_right = torch.cat((x_max.view(-1, 1), y_max.view(-1, 1)), 1)
        bottom_left = torch.cat((x_min.view(-1, 1), y_max.view(-1, 1)), 1)
        # Get bounding boxes
        box2 = torch.cat((top_left, top_right, bottom_right, bottom_left), 1).view(-1, num_anchor, 8)

        iou_anch = torch.zeros(num_anchor, num_taget_cur_image, 1)
        for i in range(0, num_taget_cur_image):
            for j in range(0, num_anchor):
                polygon1 = Polygon(box1[i, :].view(4, 2)).convex_hull
                polygon2 = Polygon(box2[i, j, :].view(4, 2)).convex_hull
                if polygon1.intersects(polygon2):
                    try:
                        inter_area = polygon1.intersection(polygon2).area
                        union_area = polygon1.union(polygon2).area
                        iou_anch[j, i] = inter_area / union_area
                    except shapely.geos.TopologicalError:
                        print('shapely.geos.TopologicalError occured, iou set to 0')

        iou_anch = iou_anch.squeeze(2)
        # Select best iou_pred and anchor
        iou_anch_best, matched_anchor_idx = iou_anch.max(0)  # best anchor [0-2] for each target
        matched_anchor_idx = matched_anchor_idx.to(device)

        # Select best unique target-anchor combinations
        if num_taget_cur_image > 1:
            iou_order = np.argsort(-iou_anch_best)  # best to worst
            # from largest iou to smallest iou
            # Unique anchor selection (slower but retains original order)

            u = torch.cat((gp1_i, gp1_j, gp2_i, gp2_j, gp3_i, gp3_j, gp4_i, gp4_j, matched_anchor_idx.unsqueeze(1)),
                          0).view(-1, num_taget_cur_image).cpu().numpy()

            _, first_unique = np.unique(u[:, iou_order], axis=1,
                                        return_index=True)  # first unique indices; each cell response to on target
            i = iou_order[first_unique]

            # best anchor must share significant commonality (iou) with target
            i = i[iou_anch_best[i] > 0.1]
            if len(i) == 0:
                continue

            matched_anchor_idx, target = matched_anchor_idx[i], target[i]
            if len(target.shape) == 1:
                target = target.view(1, 5)
        else:
            if iou_anch_best < 0.1:
                continue
            i = 0

        target_class = target[:, 0].long()
        gp1_x, gp1_y, gp2_x, gp2_y, gp3_x, gp3_y, gp4_x, gp4_y = (target[:, 1:9] * output_size).chunk(8, dim=1)

        # Get target center
        gp_x = torch.cat((gp1_x, gp2_x, gp3_x, gp4_x), 1).view(-1, 4)
        gp_y = torch.cat((gp1_y, gp2_y, gp3_y, gp4_y), 1).view(-1, 4)

        gp_x_min = torch.min(gp_x, 1)[0]
        gp_x_max = torch.max(gp_x, 1)[0]
        gp_y_min = torch.min(gp_y, 1)[0]
        gp_y_max = torch.max(gp_y, 1)[0]

        gp_x_center = torch.round(torch.mean((torch.stack((gp_x_min, gp_x_max), dim=1)), dim=1)).long()
        gp_y_center = torch.round(torch.mean((torch.stack((gp_y_min, gp_y_max), dim=1)), dim=1)).long()

        # Coordinates
        t1_x[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp1_x.squeeze(1).cpu() - gp_x_center.float().cpu()
        t1_y[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp1_y.squeeze(1).cpu() - gp_y_center.float().cpu()
        t2_x[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp2_x.squeeze(1).cpu() - gp_x_center.float().cpu()
        t2_y[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp2_y.squeeze(1).cpu() - gp_y_center.float().cpu()
        t3_x[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp3_x.squeeze(1).cpu() - gp_x_center.float().cpu()
        t3_y[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp3_y.squeeze(1).cpu() - gp_y_center.float().cpu()
        t4_x[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp4_x.squeeze(1).cpu() - gp_x_center.float().cpu()
        t4_y[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp4_y.squeeze(1).cpu() - gp_y_center.float().cpu()

        # One-hot encoding of label
        tcls[img_idx, matched_anchor_idx, gp_y_center, gp_x_center, target_class] = 1
        mask[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = 1

    return t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls
