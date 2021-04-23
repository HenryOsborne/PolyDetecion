import os
import shutil
import glob

txt_path = 'test.txt'
img_path = 'images'
out_path = 'test'
os.makedirs(out_path, exist_ok=True)

with open(txt_path) as f:
    lines = f.read().splitlines()
    for line in lines:
        if os.path.isfile(os.path.join(img_path, line + '.png')):
            shutil.copy(os.path.join(img_path, line + '.png'), out_path)
