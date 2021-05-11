import glob
import os
import os.path as osp
import math

txt_path = 'annotations'
output_path = 'ground-truth'
txt_list = glob.glob(txt_path + '/*.txt')


def fun(str_num):
    before_e = float(str_num.split('e')[0])
    if str_num.find('e') == -1:
        return float(str_num)
    sign = str_num.split('e')[1][:1]
    after_e = int(str_num.split('e')[1][1:])

    if sign == '+':
        float_num = before_e * math.pow(10, after_e)
    elif sign == '-':
        float_num = before_e * math.pow(10, -after_e)
    else:
        float_num = None
        print('error: unknown sign')
    return float_num


for txt in txt_list:
    with open(txt, 'r') as f:
        f2 = open(osp.join(output_path, osp.basename(txt)), 'w')
        lines = f.read().splitlines()
        if osp.basename(txt).startswith('C'):
            cls = 'car'
        elif osp.basename(txt).startswith('P'):
            cls = 'plane'
        else:
            raise ValueError
        for line in lines:
            line = line.split('\t')
            x1 = str(round(fun(line[0]), 3))
            y1 = str(round(fun(line[1]), 3))
            x2 = str(round(fun(line[2]), 3))
            y2 = str(round(fun(line[3]), 3))
            x3 = str(round(fun(line[4]), 3))
            y3 = str(round(fun(line[5]), 3))
            x4 = str(round(fun(line[6]), 3))
            y4 = str(round(fun(line[7]), 3))
            out_line = cls + ' ' + x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + ' ' + x3 + ' ' + y3 + ' ' + x4 + ' ' + y4
            f2.write(out_line + '\n')
        f2.close()
