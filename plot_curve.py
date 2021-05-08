import os

import matplotlib.pyplot as plt
import numpy as np
import operator
from config.yolov3 import cfg


def plot_loss_and_lr(train_loss, learning_rate):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./loss_and_lr.png')
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAP')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('./mAP.png')
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)


def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def ap_per_category(cocoGt, cocoEval, epoch):
    precisions = cocoEval.eval['precision']

    # precision: (iou, recall, cls, area range, max dets)
    assert len(cocoGt.getCatIds()) == precisions.shape[2]

    results_per_category = {}
    for idx, catId in enumerate(cocoGt.getCatIds()):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        nm = cocoGt.loadCats(catId)[0]
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        if precision.size:
            ap = np.mean(precision)
        else:
            ap = float('nan')
        results_per_category.update({str(nm['name']): float(f'{float(ap):0.3f}')})

    print(f'\n{results_per_category}')

    AP = [i for i in results_per_category.values()]
    mAP = sum(AP) / len(AP)
    print('mAP:{}'.format(mAP))
    if epoch == cfg.max_epoch:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP * 100)
        x_label = "Average Precision"
        output_path = os.path.join(cfg.output_files_path, 'mAP.png')
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            results_per_category,
            cfg.num_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )


def draw_pr(cocoGt, cocoEval):
    # precision[t,:,k,a,m] 存储的是PR曲线在各个recall阈值的precision值
    # t：阈值，k：类别，a：面积 all、small、medium、large，m：maxdet 1、10、100
    for idx, catId in enumerate(cocoGt.getCatIds()):
        nm = cocoGt.loadCats(catId)[0]
        pr_array1 = cocoEval.eval['precision'][0, :, idx, 0, 2]
        pr_array2 = cocoEval.eval['precision'][2, :, idx, 0, 2]
        pr_array3 = cocoEval.eval['precision'][4, :, idx, 0, 2]
        # pr_array4 = cocoEval.eval['precision'][0, :, idx, 0, 2]

        x = np.arange(0.0, 1.01, 0.01)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)

        plt.plot(x, pr_array1, 'b-', label='IoU=0.5')
        plt.plot(x, pr_array2, 'c-', label='IoU=0.6')
        plt.plot(x, pr_array3, 'y-', label='IoU=0.7')
        # plt.plot(x, pr_array4, 'r-', label='large')
        # plt.xticks(x_1, x_1)
        plt.title("iou=0.5 catid=%s maxdet=100" % nm['name'])

        plt.legend(loc="lower left")
        os.makedirs(os.path.join(cfg.output_files_path, 'class'), exist_ok=True)
        plt.savefig(os.path.join(cfg.output_files_path, 'class', '%s.png' % nm['name']))
        plt.close()
        # precision[t, :, k, a, m]
        # t:阈值 t[0]=0.5,t[1]=0.55,t[2]=0.6,……,t[9]=0.95
        # k:类别 k[0]=person,k[1]=bycicle,.....COCO的80个类别
        # a:面积 a[0]=all,a[1]=small,a[2]=medium,a[3]=large
        # m:maxdet m[0]=1,m[1]=10,m[2]=100
