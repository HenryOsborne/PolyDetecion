import os
import os.path as ops
import math


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


img_path = 'annotations'

txt_list = os.listdir(img_path)

txt_list = [i for i in txt_list if i.endswith('.txt')]

with open('annotations.txt', 'w') as f:
    for txt in txt_list:
        lines = open(ops.join(img_path, txt)).read().splitlines()
        out_line = txt.split('.')[0] + '.png '
        if txt.startswith('C'):
            cls = '0 '
        else:
            cls = '1 '
        for line in lines:
            line = line.split('\t')
            x1 = round(fun(line[0]), 3)
            y1 = round(fun(line[1]), 3)
            x2 = round(fun(line[2]), 3)
            y2 = round(fun(line[3]), 3)
            x3 = round(fun(line[4]), 3)
            y3 = round(fun(line[5]), 3)
            x4 = round(fun(line[6]), 3)
            y4 = round(fun(line[7]), 3)
            out_line += str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(x3) + ',' + str(
                y3) + ',' + str(x4) + ',' + str(y4) + ',' + cls
        f.write(out_line[:-1] + '\n')
