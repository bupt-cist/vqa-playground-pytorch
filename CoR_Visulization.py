import csv, sys, base64, copy, os
import scipy.misc
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import numpy as np
import utils
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def draw_item(item, output_filename):
    box_colors = ['green', 'red']
    line_color = 'orange'

    def color_map(l, name='jet', opacity=0.3):
        #
        # 产生attention效果的,彩色组合
        s = np.argsort(l)
        tmp = np.linspace(0, 1, len(l))
        t = [0] * len(l)
        for i, e in enumerate(s):
            t[e] = tmp[i]

        if name == 'custom':
            name = [[0, 0, 0.5],
                    [0.5, 1, 0.5],
                    [0.5, 0, 0]]
            name = ['blue', 'yellow', 'red']

        if isinstance(name, list):
            assert len(name) == len(l)
            colors = np.array(name)[s]
        else:
            colors = plt.get_cmap(name)(t)
            colors[:, 3] = opacity
        return colors

    def sub_figure(id, att, raw_img, boxes):
        index_sort = np.argsort(att)[::-1][:len(box_colors)]
        boxesN = boxes[index_sort]
        boxesN = [list(map(int, box)) for box in boxesN]
        attN = att[index_sort]

        fig = plt.figure()

        # img = copy.deepcopy(raw _img)
        img = copy.deepcopy(raw_img)
        ax = fig.add_subplot(111)

        # img = rgb2gray(img, boxesN)
        att_map_v = color_map(attN, name=box_colors, opacity=1)
        for i, e in enumerate(boxesN):
            rect = patches.Rectangle((e[0], e[1]), e[2] - e[0], e[3] - e[1], linewidth=4, edgecolor=att_map_v[i],
                                     facecolor=att_map_v[i], fill=False)
            ax.add_patch(rect)
        fout = plt.imshow(img)
        plt.axis('off')
        fout.axes.get_xaxis().set_visible(False)
        fout.axes.get_yaxis().set_visible(False)
        filename = "subimage_with_top3boxes_{}.jpg".format(id)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename, boxesN, attN

    def sub_region(target_img, sub, box, start_point, p):
        draw.ellipse((p[0] - cl, p[1] - cl, p[0] + cl, p[1] + cl), fill='black', outline='black')
        #  sub = filename
        sub_img = Image.open(sub).convert('RGB')
        sub_img = sub_img.resize((h, w), Image.ANTIALIAS)
        enhanced_img = ImageEnhance.Color(sub_img).enhance(5)
        target_img.paste(enhanced_img, start_point)

        sub_p = []
        for i in range(len(box)):
            box_img = sub_img.crop(box[i])
            box[i][0] += start_point[0]
            box[i][1] += start_point[1]
            box[i][2] += start_point[0]
            box[i][3] += start_point[1]
            x = (box[i][2] + box[i][0]) / 2
            #  x = box[i][0]
            y = (box[i][3] + box[i][1]) / 2
            #  y = box[i][1]
            sub_p.append((x, y))

            if i == 0:
                fill = 'red'
            elif i == 1:
                fill = 'yellow'
            elif i == 2:
                fill = 'blue'
            line_draw.line((p[0], p[1], sub_p[i][0], sub_p[i][1]), fill=line_color, width=linewidth)
            line_draw.ellipse((sub_p[i][0] - cl, sub_p[i][1] - cl, sub_p[i][0] + cl, sub_p[i][1] + cl), fill='black',
                              outline='black')

        return sub_p

    boxes = item['boxes']
    img = Image.open(item['img_filename']).convert('RGBA')
    enhancer = ImageEnhance.Contrast(img)
    raw_img = np.array(enhancer.enhance(0.5))
    w, h = raw_img.shape[:2]
    sub1, box1, att1 = sub_figure(1, item['alpha_dict']['alpha1'], raw_img, boxes)
    sub2, box2, att2 = sub_figure(2, item['alpha_dict']['alpha2'], raw_img, boxes)
    sub3, box3, att3 = sub_figure(3, item['alpha_dict']['alpha3'], raw_img, boxes)
    target_img = Image.new('RGBA', (
        h * 4, w * 4), 'white')
    draw = ImageDraw.Draw(target_img)
    line_draw = ImageDraw.Draw(target_img)

    p_s, i_s, mid = 100, 200, w * 2
    diff = h + 100
    linewidth = 4

    cl = 8

    start_point = (i_s, mid - w // 2)
    p1 = (p_s, mid)
    sub1_p = sub_region(target_img, sub1, box1, start_point, p1)

    start_point = (i_s + diff, mid - w // 2)
    p2 = (p_s + diff + 50, mid)
    line_draw.line((sub1_p[0][0], sub1_p[0][1], p2[0], p2[1]), fill=line_color, width=linewidth)
    sub2_p = sub_region(target_img, sub2, box2, start_point, p2)

    start_point = (i_s + diff * 2, mid - w // 2)
    p3 = (p_s + diff * 2 + 50, mid)
    line_draw.line((sub2_p[0][0], sub2_p[0][1], p3[0], p3[1]), fill=line_color, width=linewidth)
    sub3_p = sub_region(target_img, sub3, box3, start_point, p3)

    target_img = target_img.crop((p1[0] - 100, mid - w // 2 - 100, p_s + diff * 3 + 100, mid + w // 2 + 100))
    target_img.save(output_filename, quality=100)
    os.remove(sub1)
    os.remove(sub2)
    os.remove(sub3)


if __name__ == '__main__':
    output_dir = 'tmpdata'
    samples = utils.file2data('/root/data/VQA/analyze/ABRRCNNS3NormReluSigmoidLoss_VAL/samples,n_10000,s_10.h5')

    S3Good = [e for e in samples if e['q_id'] in [3403290]]

    for s in S3Good:
        output_img_filename = os.path.join(output_dir, '%s.jpg' % s['q_id'])
        draw_item(s, output_img_filename)
