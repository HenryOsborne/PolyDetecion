import json
import os
from glob import glob
import cv2

txt_path = 'ground-truth'
images_path = 'images'

mode = 'trainval'
if mode == 'trainval':
    tarinval_txt = 'trainval.txt'
elif mode == 'test':
    tarinval_txt = 'test.txt'

with open(tarinval_txt, 'r') as f:
    txt_list = f.read().splitlines()

txt_list = [i + '.txt' for i in txt_list]
images_list = [i.split('.')[0] + '.png' for i in txt_list]

images, annotations, categories = [], [], []

class_dict = {
    0: 'car',
    1: 'plane',
}

for cat_id in class_dict:
    categories.append({'id': cat_id, 'name': class_dict[cat_id]})

img_id = 0
for image in images_list:
    img = cv2.imread(os.path.join(images_path, image))
    height, weight = img.shape[0], img.shape[1]
    img_name = os.path.basename(image)
    images.append({"file_name": img_name, "height": height, "width": weight, "id": img_id})
    img_id += 1

idx = 0
for txt in txt_list:
    with open(os.path.join(txt_path, txt), 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            line = line.split(' ')
            coord = [float(i) for i in line[1:]]
            x1, y1, x2, y2, x3, y3, x4, y4 = coord
            catergory_id = 0 if line[0] == 'car' else 1
            segment = [x1, y1, x2, y2, x3, y3, x4, y4]
            txt_name = os.path.basename(txt)
            im_name = txt_name.split('.')[0] + '.png'
            for i in images:
                if i['file_name'] == im_name:
                    image_id = i['id']
                    break
            anno_info = {
                "id": idx, "category_id": catergory_id,
                "iscrowd": 0, "image_id": image_id,
                'segmentation': [segment]
            }
            annotations.append(anno_info)
            idx += 1

print('categories:{}'.format(len(categories)))
print('images:{}'.format(len(images)))
print('annotations:{}'.format(len(annotations)))

all_json = {"images": images, "annotations": annotations, "categories": categories}
with open(mode + '.json', 'w') as f:
    json.dump(all_json, f)
