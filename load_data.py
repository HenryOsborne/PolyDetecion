import numpy as np
from torch.utils.data import Dataset
from config.yolov3 import cfg
import torch
import os
import glob
import cv2
import json
import random


class NewDataset(Dataset):
    def __init__(self, train_set=True):
        if train_set is True:
            self.annotation_json = json.load(open(cfg.trainval_json))
        else:
            self.annotation_json = json.load(open(cfg.test_json))
        self.image_path = cfg.image_path
        self.class_names = self.get_class_names()
        self.num_classes = len(self.class_names)
        self.bacth_size = cfg.batch_size
        self.anchors = np.array(cfg.anchors)
        self.input_sizes = cfg.input_sizes
        self.output_size = [52, 26, 13]
        self.strides = cfg.strides
        self.max_boxes_per_scale = cfg.max_boxes_per_scale
        self.if_show = False  # 是否可视化label读取效果
        # self.class_index = {self.class_names[i]: i for i in range(len(self.class_names))}

    def __len__(self):
        return len(self.annotation_json['images'])

    def __getitem__(self, idx):
        anno_info = [i for i in self.annotation_json['annotations'] if i['image_id'] == idx]
        image_info = [i for i in self.annotation_json['images'] if i['id'] == idx][0]

        # ndarray(num of instance,8)
        bboxes = self.get_bbox(anno_info)
        labels = self.get_label(anno_info)
        labels = torch.unsqueeze(torch.tensor(labels, dtype=torch.int64), 1)
        image = self.get_image(image_info['file_name'])
        iscrowd = [0]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["segmentation"] = bboxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.if_show:  # 可视化查看数据增强的正确性
            self.show_image_and_bboxes(np.copy(image), np.copy(bboxes))

        return image, bboxes, labels, target

    def get_class_names(self):
        class_names = []
        for cat in self.annotation_json['categories']:
            class_names.append(cat['name'])
        return class_names

    def get_label(self, anno_info):
        label_array = []
        for anno in anno_info:
            label = anno['category_id']
            label_array.append(label)
        return np.array(label_array, dtype=np.int64)

    def get_bbox(self, anno_info):
        bboxes_array = []
        for anno in anno_info:
            coord = anno['segmentation'][0]
            bboxes_array.append([int(float(x)) for x in coord])
        return np.array(bboxes_array, dtype=np.float64)

    def get_image(self, img_name):
        image = np.array(cv2.imread(os.path.join(self.image_path, img_name)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # BGR -> RGB
        return image

    def get_bbox_array(self, list_bboxes):
        bboxes_array = []
        for list_bbox in list_bboxes:
            bboxes_array.append([int(float(x)) for x in list_bbox[1:]])
        return np.array(bboxes_array, dtype=np.float64)

    def get_image_array(self, image_name):
        image = np.array(cv2.imread(self.image_path + image_name + '.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # BGR -> RGB
        return image

    # 对image进行归一化操作
    def normalization(self, image):
        image = image / 255.
        return image

    # 对图像resize以符合输入要求,可选择pad和no pad方式
    def resize_image(self, image, bboxes, input_size):
        h, w, _ = image.shape  # (h,w,c)
        if not cfg.if_pad:  # 直接resize,可能会导致图像变形
            new_image = cv2.resize(image, (input_size, input_size))
            bboxes[:, [0, 2, 4, 6]] = bboxes[:, [0, 2, 4, 6]] * input_size / w
            bboxes[:, [1, 3, 5, 7]] = bboxes[:, [1, 3, 5, 7]] * input_size / h
        else:  # 补空保证图像不变形
            scale = input_size / max(w, h)  # 得到input size/图像的宽和高较小的那一个scale
            w, h = int(scale * w), int(scale * h)  # 将原图像resize到这个大小,不改变原来的形状

            image = cv2.resize(image, (w, h))
            fill_value = 0  # 选择边缘补空的像素值
            new_image = np.ones((input_size, input_size, 3)) * fill_value  # 新的符合输入大小的图像
            dw, dh = (input_size - w) // 2, (input_size - h) // 2
            new_image[dh:dh + h, dw:dw + w, :] = image

            # 将bbox也映射到resize后的坐标
            bboxes[:, [0, 2, 4, 6]] = bboxes[:, [0, 2, 4, 6]] * scale + dw
            bboxes[:, [1, 3, 5, 7]] = bboxes[:, [1, 3, 5, 7]] * scale + dh

        return new_image, bboxes

    # TODO:随机水平翻转和随机裁剪代码有问题，需重写
    # 随机水平翻转
    def random_horizontal_flip(self, image, bboxes):
        pass

    # 随机裁剪
    def random_crop(self, image, bboxes):
        pass

    # 数据增强
    def data_augmentation(self, image, bboxes):
        if cfg.random_horizontal_flip:  # 随机水平翻转
            image, bboxes = self.random_horizontal_flip(image, bboxes)
        if cfg.random_crop:  # 随机裁剪
            image, bboxes = self.random_crop(image, bboxes)

        return image, bboxes

    def show_image_and_bboxes(self, image, bboxes):
        image = image.astype(np.uint8)
        pts = []
        for box in bboxes:
            box_pt = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
            box_pt = box_pt.reshape((-1, 1, 2))
            pts.append(box_pt)

        cv2.polylines(image, pts, True, (0, 255, 0), 2)
        cv2.imshow("show_image_and_bboxes", image)
        cv2.waitKey(0)

    def collate_fn(self, batch):
        image = [i[0] for i in batch]
        bboxes = [i[1] for i in batch]
        label = [i[2] for i in batch]
        target = tuple([i[3] for i in batch])
        assert len(image) == len(bboxes) == len(label) == len(target)

        input_size = random.choice(self.input_sizes)  # 每次随机选取输入图像的大小
        self.output_size = [input_size // stride for stride in self.strides]  # yolo输出大小

        logit = []
        for i in range(len(image)):
            # image[i], bboxes[i] = self.data_augmentation(image[i],bboxes[i])
            image[i], bboxes[i] = self.resize_image(image[i], bboxes[i], input_size)
            image[i] = self.normalization(image[i])
            image[i] = torch.from_numpy(image[i])
            image[i] = image[i].permute(2, 0, 1)
            bboxes[i] = torch.from_numpy(bboxes[i])
            logit.append(torch.cat((label[i], bboxes[i] / input_size), 1))

        image = torch.stack(image)
        image = image.type(torch.FloatTensor)

        return tuple((image, logit, target))


if __name__ == '__main__':
    mydataset = NewDataset()
    from torch.utils.data import DataLoader

    train_data_loader = DataLoader(mydataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_worker,
                                   collate_fn=mydataset.collate_fn)

    for epoch, data in enumerate(train_data_loader):
        print('1')
