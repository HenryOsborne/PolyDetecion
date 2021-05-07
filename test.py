import cv2
import torch
import os
import argparse
import random
import shutil

from models.yolov3 import yolov3
from config.yolov3 import cfg
from utils.nms import non_max_suppression
from load_data import NewDataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

parser = argparse.ArgumentParser()
parser.add_argument('-image_folder', type=str, default='./data/images', help='path to images')
parser.add_argument('-output_folder', type=str, default='result_1', help='path to outputs')
parser.add_argument('-plot_flag', type=bool, default=True)
parser.add_argument('-txt_out', type=bool, default=True)
parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-weights_path', type=str, default='checkpoint/180.pt', help='weight file path')
parser.add_argument('-class_path', type=str, default='data/data.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=608, help='size of each image dimension')
opt = parser.parse_args()


class Test(object):
    def __init__(self, opt=None):
        assert opt is not None
        self.opt = opt
        self.device = torch.device(cfg.device)

        self.val_dataset = NewDataset(train_set=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=True,
                                         num_workers=cfg.num_worker,
                                         collate_fn=self.val_dataset.collate_fn)

        self.len_train_dataset = len(self.val_dataset)

        self.model = yolov3().to(self.device)
        weights_path = self.opt.weights_path
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

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
        tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]

        cv2.line(img, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), color, tl)
        cv2.line(img, (int(x[2]), int(x[3])), (int(x[4]), int(x[5])), color, tl)
        cv2.line(img, (int(x[4]), int(x[5])), (int(x[6]), int(x[7])), color, tl)
        cv2.line(img, (int(x[6]), int(x[7])), (int(x[0]), int(x[1])), color, tl)
        cv2.putText(img, label, (int(x[0]), int(x[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    def drow_box(self, anns):
        image_id = [i['image_id'] for i in anns]
        assert all(x == image_id[0] for x in image_id)
        img_ann = self.cocoGt.loadImgs(ids=image_id[0])[0]
        img_name = img_ann['file_name']
        print('images:{}'.format(img_name))
        img_path = os.path.join(opt.image_folder, img_name)
        txt_path = os.path.join(opt.output_folder, img_name.replace('png', 'txt'))
        img = cv2.imread(img_path)
        for ann in anns:
            cat = self.cocoGt.loadCats(ids=ann['category_id'])[0]
            score = ann['score']
            label = '%s %.2f' % (cat['name'], score)
            color = (0, 0, 255)
            coord = ann['segmentation'][0]
            with open(txt_path, 'a') as f:
                f.write('%s %.2f %g %g %g %g %g %g %g %g  \n' %
                        (cat['name'], score,
                         coord[0], coord[1], coord[2], coord[3], coord[4], coord[5], coord[6], coord[7]))
            self.plot_one_box(coord, img, color, label)
        cv2.imwrite(os.path.join(opt.output_folder, img_name), img)

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

            new_ann = self.reorginalize_target(detections, logit, image_size)
            self.drow_box(new_ann)
            anns.extend(new_ann)

        for ann in anns:
            ann['segmentation'] = self.cocoGt.annToRLE(ann)  # 将polygon形式的segmentation转换RLE形式

        cocoDt = self.cocoGt.loadRes(anns)

        cocoEval = COCOeval(self.cocoGt, cocoDt, iou_types)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


if __name__ == '__main__':
    if os.path.isdir(opt.output_folder):
        shutil.rmtree(opt.output_folder)
    os.makedirs(opt.output_folder)
    test = Test(opt)
    test.eval()
