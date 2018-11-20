# -*- coding:utf-8 -*-
print("[info] utils: Welcome to Deep Learning Utils, start loading...")
import collections
import csv
import functools
import gzip
import itertools
import json
import operator
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import timedelta
from timeit import default_timer
import deepdish
import h5py
import nltk
import numpy as np
import scipy.misc
from tqdm import tqdm
import yagmail
import time
import argparse
import copy
import munch
import platform
import imghdr
from PIL import Image
from itertools import groupby
import pickle
from datetime import datetime
from configobj import ConfigObj
from passlib.hash import sha512_crypt
import yaml
import torch
import glob
from collections import defaultdict
import random

# try:
#     import tensor_comprehensions
# except ImportError:
#     print("[info] utils: tensor_comprehensions hasn't installed")

print("[info] utils: Deep Learning Utils loaded successfully, enjoy it!")


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def str2bool(v):
    if v is None:
        return False
    elif isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
            return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Timer:
    def __init__(self):
        '''
        t = Timer()
        time.sleep(1)
        print(t.elapse())
        '''
        self.start = default_timer()

    def elapse(self, readable=False):
        seconds = default_timer() - self.start
        if readable:
            seconds = str(timedelta(seconds))
        return seconds


def timing(f):
    # 计时器
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def groupby(l, key=lambda x: x):
    d = collections.defaultdict(list)
    for item in l:
        d[key(item)].append(item)
    return dict(d.items())


def list_filenames(dirname, filter_fn=None, sort_fn=None, printable=True):
    dirname = os.path.abspath(dirname)
    filenames = os.listdir(dirname)
    if filter_fn:
        tmp = len(filenames)

        filenames = [e for e in filenames if filter_fn(e)]
        if printable: print('[info] utils.list_filenames: Detected %s files/dirs in %s, filtering to %s files.' % (
            tmp, dirname, len(filenames)))
    else:
        if printable: print(
            '[info] utils.list_filenames: Detected %s files/dirs in %s, No filtering.' % (len(filenames), dirname))
    if sort_fn:
        filenames = sorted(filenames, key=sort_fn)
    filenames = [os.path.join(dirname, filename) for filename in filenames]
    return filenames


def list_dirs(dirname):
    dirname = os.path.abspath(dirname)
    filenames = sorted(os.listdir(dirname))
    filenames = [os.path.join(dirname, filename) for filename in filenames]
    for e in filenames:
        if not os.path.isdir(e):
            raise ValueError("[error] utils.list_filenames: %s is not a dirname" % e)
    return filenames


def listdict2dict2list(listdict, printable=True):
    tmp_dict = collections.defaultdict(list)
    for example_dict in listdict:
        for k, v in example_dict.items():
            tmp_dict[k].append(v)
    if printable: print('[info] utils.listdict2dictlist: %s' % tmp_dict.keys())
    return dict(tmp_dict)


class AvgMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg


class SumMeter(object):
    def __init__(self):
        self.val = 0
        self.sum = 0

    def reset(self):
        self.val = 0
        self.sum = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n

    def value(self):
        return self.sum


class Experiment(object):
    def __init__(self, name, options=dict()):
        """ Create an experiment
        """

        self.name = name
        self.options = options
        self.date_and_time = time.strftime('%d-%m-%Y--%H-%M-%S')

        self.info = defaultdict(dict)
        self.logged = defaultdict(dict)
        self.meters = defaultdict(dict)

    def add_meters(self, tag, meters_dict):
        assert tag not in (self.meters.keys())
        for name, meter in meters_dict.items():
            self.add_meter(tag, name, meter)

    def add_meter(self, tag, name, meter):
        assert name not in list(self.meters[tag].keys()), \
            "meter with tag {} and name {} already exists".format(tag, name)
        self.meters[tag][name] = meter

    def update_options(self, options_dict):
        self.options.update(options_dict)

    def log_meter(self, tag, name, n=1):
        meter = self.get_meter(tag, name)
        if name not in self.logged[tag]:
            self.logged[tag][name] = {}
        self.logged[tag][name][n] = meter.value()

    def log_meters(self, tag, n=1):
        for name, meter in self.get_meters(tag).items():
            self.log_meter(tag, name, n=n)

    def reset_meters(self, tag):
        meters = self.get_meters(tag)
        for name, meter in meters.items():
            meter.reset()
        return meters

    def get_meters(self, tag):
        assert tag in list(self.meters.keys())
        return self.meters[tag]

    def get_meter(self, tag, name):
        assert tag in list(self.meters.keys())
        assert name in list(self.meters[tag].keys())
        return self.meters[tag][name]

    def to_json(self, filename):
        os.system('mkdir -p ' + os.path.dirname(filename))
        var_dict = copy.copy(vars(self))
        var_dict.pop('meters')
        for key in ('viz', 'viz_dict'):
            if key in list(var_dict.keys()):
                var_dict.pop(key)
        data2file(var_dict, filename, override=True)
        # with open(filename, 'w') as f:
        #     json.dump(var_dict, f)

    def from_json(filename):
        with open(filename, 'r') as f:
            var_dict = json.load(f)
        xp = Experiment('')
        xp.date_and_time = var_dict['date_and_time']
        xp.logged = var_dict['logged']
        # TODO: Remove
        if 'info' in var_dict:
            xp.info = var_dict['info']
        xp.options = var_dict['options']
        xp.name = var_dict['name']
        return xp


def split_filepath(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname


def load_npy(filename):
    return np.load(filename).item()


def data2file(data, filename, type=None, params=None, override=False):
    dirname, rootname, extname = split_filepath(filename)
    print_save_flag = True
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not os.path.exists(filename) or override:
        if extname == 'pkl':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        elif extname == 'h5':
            if params is None:
                params = {}
            split_num = params.get('split_num')

            if split_num:
                if not isinstance(data, list):
                    raise ValueError(
                        '[error] utils.data2file: data must have type of list when use split_num, but got %s' % (
                            type(data)))

                if not split_num <= len(data):
                    raise ValueError(
                        '[error] utils.data2file: split_num(%s) must <= data(%s)' % (len(split_num), len(data)))

                print_save_flag = False
                print_did_not_save_flag = False
                pre_define_filenames = ["%s_%d" % (filename, i) for i in range(split_num)]
                pre_search_filenames = glob.glob("%s*" % filename)

                strict_existed = (set(pre_define_filenames) == set(pre_search_filenames) and len(
                    set([os.path.exists(e) for e in pre_define_filenames])) == 1)
                common_existed = len(set([os.path.exists(e) for e in pre_search_filenames])) == 1

                def rewrite():
                    print('[info] utils.data2file: Spliting data to %s parts before saving...' % split_num)
                    data_splits = np.array_split(data, indices_or_sections=split_num)
                    for i, e in enumerate(data_splits):
                        deepdish.io.save("%s_%d" % (filename, i), list(e))
                    print('[info] utils.data2file: Saved data to %s_(0~%d)' % (
                        os.path.abspath(filename), len(data_splits) - 1))

                if strict_existed and not override:
                    print(
                        '[warning] utils.data2file: Did not save data to %s_(0~%d) because the files strictly exist and override is False' % (
                            os.path.abspath(filename), len(pre_search_filenames) - 1))
                elif common_existed:
                    print(
                        '[warning] utils.data2file: Old wrong files (maybe a differnt split) exist, auto delete them.')
                    for e in pre_search_filenames:
                        os.remove(e)
                    rewrite()
                else:
                    rewrite()
            else:
                deepdish.io.save(filename, data)
        elif extname == 'hy':
            # hy support 2 params: key and max_step
            # if key, then create group using key, else create group using index
            # if max_step, then the loop may early stopping, used for debug
            if params is None:
                params = {}
            key = params.get('key')
            max_step = params.get('max_step')
            if max_step is None:
                max_step = np.Infinity
            with h5py.File(filename, 'w') as f:
                for i, predict in enumerate(tqdm(data)):
                    if i < max_step:
                        if key:
                            grp = f.create_group(name=key(predict))
                        else:
                            grp = f.create_group(name=str(i))
                        for k in predict.keys():
                            grp[k] = predict[k]
                    else:
                        break
        elif extname == 'csv':
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif extname == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f)
        elif extname == 'npy':
            np.save(filename, data)
        # elif extname == 'ckpt':
        #     tf.train.Saver().save(data, filename)
        # elif extname == 'jpg' or extname == 'png':
        #     plt.imsave(filename, data)
        elif extname == 'pt':
            torch.save(data, filename)
        elif extname == 'txt':
            if params is None:
                params = {}
            max_step = params.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('type can only support h5, csv, json, sess')
        if print_save_flag: print('[info] utils.data2file: Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: print(
            '[warning] utils.data2file: Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))


def file2data(filename, type=None, params=None, printable=True):
    dirname, rootname, extname = split_filepath(filename)
    print_load_flag = True
    if type:
        extname = type
    if extname == 'pkl':
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    elif extname == 'h5':
        if params is None:
            params = {}
        split_num = params.get('split_num')
        if split_num:
            print_load_flag = False
            if isinstance(split_num, int):
                filenames = ["%s_%i" % (filename, i) for i in range(split_num)]
                if split_num != len(glob.glob("%s*" % filename)):
                    print(
                        '[warning] utils.file2data: Maybe you are giving a wrong split_num(%d) != seached num (%d)' % (
                            split_num, len(glob.glob("%s*" % filename))))

            elif split_num == 'auto':
                filenames = glob.glob("%s*" % filename)
                print('[warning] utils.file2data: Auto located %d splits linked to %s' % (len(filenames), filename))
            else:
                raise ValueError("params['split_num'] got unexpected value: %s, which is not supported." % split_num)
            data = []
            for e in filenames:
                data.extend(deepdish.io.load(e))
            print('[info] utils.file2data: Loaded data from %s_(%s)' % (
                os.path.abspath(filename), ','.join(sorted([e.split('_')[-1] for e in filenames]))))
        else:
            data = deepdish.io.load(filename)
    elif extname == 'hy':
        data = h5py.File(filename, 'r')
        # print('[info] utils.file2data: size: %d, keys: %s' % (len(f.keys()), list(f['0'].keys())))
    elif extname in ['npy', 'npz']:
        try:
            data = np.load(filename)
        except UnicodeError:
            print('[warning] utils.file2data: %s is python2 format, auto use latin1 encoding.' % os.path.abspath(
                filename))
            data = np.load(filename, encoding='latin1')
    elif extname == 'json':
        with open(filename) as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise ValueError('[error] utils.file2data: failed to load json file %s' % filename)
    elif extname == 'ini':
        data = ConfigObj(filename, encoding='utf-8')
    elif extname == 'pt':
        data = torch.load(filename)
    elif extname == 'txt':
        with open(filename, encoding='utf-8') as f:
            data = f.read().split('\n')[:-1]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            print('[info] utils.file2data: Loaded data from %s' % os.path.abspath(filename))
    return data


def get_type(value):
    if type(value) != list:
        return type(value).__name__
    else:
        if np.array(value).dtype == np.dtype('float64'):
            return "list_float"
        elif np.array(value).dtype == np.dtype('int32'):
            return "list_int"


def img2arr(image_filename, size=None, dim=4, divisor=255.0):
    # size = (299, 299, 3)
    # print("img2arr deprecated, use Class ImageLoader")
    img = scipy.misc.imread(image_filename)
    if len(img.shape) == 0 or len(img.shape) == 1:
        raise ValueError("{0} has a wrong format.".format(image_filename))
    if len(img.shape) == 2:
        print(image_filename)
        img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='float32')
        img_new[:, :, 0] = img
        img_new[:, :, 1] = img
        img_new[:, :, 2] = img
        img = img_new
    assert len(img.shape) == 3
    if size is None:
        return img
    elif len(size) == 3:
        w, h, d = size
        img_resized = scipy.misc.imresize(img, size)
        if dim == 1:
            return img_resized.reshape(w * h * d) / divisor
        elif dim == 2:
            return img_resized.reshape(1, w * h * d) / divisor
        elif dim == 3:
            return img_resized.reshape(w, h, d) / divisor
        elif dim == 4:
            return np.array(img_resized.reshape(1, w, h, d) / divisor, dtype=np.float32)
        else:
            raise ValueError("outputdim expect 1~4, but got {0}".format(dim))
    else:
        raise ValueError("size can only be 3 tuple or None")


class ImageLoader(object):
    def __init__(self, size=224, divisor=1, mean_file='/root/data/checkpoints/ilsvrc_2012_mean.npy'):
        self.size = size
        self.mean_file = mean_file
        if mean_file:
            self.mean = np.load(mean_file).mean(1).mean(1)
        self.divisor = divisor

    def load_img(self, img_filename):
        img = scipy.misc.imread(img_filename)
        if len(img.shape) == 0 or len(img.shape) == 1:
            raise ValueError("{0} has a wrong format.".format(img_filename))
        elif len(img.shape) == 2:
            img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='float32')
            img_new[:, :, 0] = img
            img_new[:, :, 1] = img
            img_new[:, :, 2] = img
            img = img_new
        else:
            assert len(img.shape) == 3
        img = scipy.misc.imresize(img, [self.size, self.size])
        if self.mean_file:
            img = img - self.mean
        img = img / self.divisor
        img = np.array(img, dtype=np.float32)
        return img

    def load_imgs(self, img_filenames):
        return np.array([self.load_img(e) for e in img_filenames], dtype=np.float32)


# def preprocess_imgs(img_filenames, image_size=224, is_training=True):
#     # img_filenames: list<str> [batch_size,]
#     # return: tensor [batch_size, image_size, image_size, 3]
#     preprocessed_imgs = []
#     for img_filename in tf.unstack(img_filenames):
#         img_string = tf.read_file(img_filename)
#         try:
#             img = tf.image.decode_jpeg(img_string, channels=3)
#         except Exception:
#             tmp = tf.image.decode_png(img_string, channels=3)
#             tmp = tf.image.encode_jpeg(tmp, format='rgb', quality=100)
#             img = tf.image.decode_jpeg(tmp, channels=3)
#
#         processed_img = vgg_preprocessing.preprocess_image(img, image_size, image_size, is_training=False)
#         preprocessed_imgs.append(processed_img)
#     return tf.stack(preprocessed_imgs)


def download_file(fileurl, filedir=None, progress_bar=True, override=False, fast=False, printable=True):
    if filedir:
        ensure_dirname(filedir)
        assert os.path.isdir(filedir)
    else:
        filedir = ''
    filename = os.path.abspath(os.path.join(filedir, fileurl.split('/')[-1]))
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("[info] utils.download_file: %s not exist, automatic makedir." % dirname)
    if not os.path.exists(filename) or override:
        if fast:
            p = subprocess.Popen('axel -n 10 -o {0} {1}'.format(filename, fileurl), shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(p.stdout.readline, ''):
                if line:
                    print(line.decode('utf-8').replace('\n', ''))
                else:
                    p.kill()
                    break
        else:
            if progress_bar:
                def my_hook(t):
                    last_b = [0]

                    def inner(b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            t.total = tsize
                        t.update((b - last_b[0]) * bsize)
                        last_b[0] = b

                    return inner

                with tqdm(unit='B', unit_scale=True, miniters=1,
                          desc=fileurl.split('/')[-1]) as t:
                    urllib.request.urlretrieve(fileurl, filename=filename,
                                               reporthook=my_hook(t), data=None)
            else:
                urllib.request.urlretrieve(fileurl, filename=filename)
        if printable: print("[info] utils.download_file: %s downloaded sucessfully." % filename)
    else:
        if printable: print("[info] utils.download_file: %s already existed" % filename)
    return filename


def extract_file(filename, targetname="HERE", override=False, printable=True):
    assert os.path.exists(filename)
    dirname, rootname, extname = split_filepath(filename)

    if targetname == 'HERE':
        targetname = os.path.abspath(dirname)
    elif targetname == 'NEW':
        targetname = os.path.join(dirname, rootname)
    else:
        targetname = os.path.abspath(targetname)

    if targetname == os.path.abspath(dirname) or override or not os.path.exists(targetname):
        if extname == 'tar' or extname == 'tar.gz':
            with tarfile.open(filename) as f:
                for e in f.getnames():
                    f.extract(e, path=targetname)
        elif extname == 'zip':
            with zipfile.ZipFile(filename) as f:
                f.extractall(path=targetname)
        elif extname == 'gz':
            with gzip.GzipFile(filename) as f, open(os.path.join(targetname, rootname), "wb") as t:
                t.write(f.read())
        else:
            raise ValueError("Only support tar, tar.gz, zip, gz")
        if printable: print("[info] utils.extract_file: extracted sucessfully to %s " % targetname)
    else:
        if printable: print("[info] utils.extract_file: %s already existed" % targetname)


def copy_file(filename, targetname, override=False, printable=True):
    filename = os.path.abspath(filename)
    targetname = os.path.abspath(targetname)
    if not os.path.exists(targetname) or override:
        with open(filename, 'r') as f1, open(targetname, 'w') as f2:
            shutil.copyfileobj(f1, f2)
        if printable:
            print('[info] utils.copy_file: Copied %s to %s.' % (filename, targetname))
    else:
        if printable:
            print('[warning] utils.copy_file: Did not copy because %s exists.' % targetname)


def compress_file(filename, targetname=None, type='zip', override=False, printable=True):
    if targetname is None:
        targetname = os.path.abspath("%s.%s" % (filename, type))
    filename = os.path.abspath(filename)
    if not os.path.exists(targetname) or override:
        if type == 'zip':
            with zipfile.ZipFile(targetname, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(filename, arcname=os.path.basename(filename))
            if printable:
                print('[info] utils.compress_file: Compressed %s to %s.' % (filename, targetname))
        else:
            raise ValueError('Only support type zip now, but got %s' % type)
    else:
        if printable:
            print('[warning] utils.compress_file: Did not compress because %s exists.' % targetname)
    return targetname


def clean_path(path):
    while os.path.exists(path):
        shutil.rmtree(path)
    while not os.path.exists(path):
        os.makedirs(path)


def ensure_path(path, override=False):
    raise ValueError('[error] utils.ensure_path: this function has been deprecated.')
    # dirname, rootname, extname = split_filepath(path)
    if not os.path.exists(path) or override:
        while os.path.exists(path):
            shutil.rmtree(path)
        while not os.path.exists(path):
            os.makedirs(path)


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        print('[info] utils.ensure_dirname: removing dirname: %s' % os.path.abspath(dirname))
        shutil.rmtree(dirname)
    if not os.path.exists(dirname):
        print('[info] utils.ensure_dirname: making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname)

        # if override:
        #     shutil.rmtree(dirname)
        # if not os.path.exists(dirname):
        #     print('[info] utils.ensure_dirname: making dirname: %s' % os.path.abspath(dirname))
        #     os.makedirs(dirname)


def ensure_filename(filename, override=False):
    dirname, rootname, extname = split_filepath(filename)
    ensure_dirname(dirname, override=override)


# 数据处理

def sentencelist2wordlist(sentencelist):
    return list(itertools.chain(*[e.split() for e in sentencelist]))


def flattenlist(nestedlist):
    return list(itertools.chain(*nestedlist))


def length2sublist(length, num_sublist):
    spacing = np.linspace(0, length, num_sublist + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    return ranges


def split_length(length, step=None, num=None):
    if step:
        assert not num
        assert step <= length
    else:
        assert num
        assert num <= length

    assert (not step and num) or (not num and step)
    if num:
        step = int(np.ceil(length / num))

    spacing = list(np.arange(0, length, step)) + [length]
    if num and len(spacing) - 1 < num:
        x = length - num
        spacing = spacing[0:x] + [i for i in range(spacing[x], length + 1)]

    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append(list(range(spacing[i], spacing[i + 1])))
    return ranges


def sentence2wordlist(sentence, start=None, end=None):
    s = sentence.split()
    tmp = []
    if start:
        tmp.append(start)
    tmp.extend(s)
    if end:
        tmp.append(end)
    return tmp


def preprocess_text2(s):
    t_str = s.lower()

    for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\.', r'\;']:
        t_str = re.sub(i, '', t_str)
    for i in [r'\-', r'\/']:
        t_str = re.sub(i, ' ', t_str)
    q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list


def preprocess_text(s):
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                    "couldnt": "couldn't", \
                    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                    "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                    "hed": "he'd", "hed've": "he'd've", \
                    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                    "Id've": "I'd've", "I'dve": "I'd've", \
                    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                    "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                    "mightn'tve": "mightn't've", "mightve": "might've", \
                    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                    "oclock": "o'clock", "oughtnt": "oughtn't", \
                    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                    "shed've": "she'd've", "she'dve": "she'd've", \
                    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                    "someone'dve": "someone'd've", \
                    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                    "somethingd've": "something'd've", \
                    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                    "thered": "there'd", "thered've": "there'd've", \
                    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                    "theyd've": "they'd've", \
                    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                    "twas": "'twas", "wasnt": "wasn't", \
                    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                    "whatll": "what'll", "whatre": "what're", \
                    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                    "wheres": "where's", "whereve": "where've", \
                    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                    "whos": "who's", "whove": "who've", "whyll": "why'll", \
                    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                    "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                    "yall'd've": "y'all'd've", \
                    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                    "youd've": "you'd've", "you'dve": "you'd've", \
                    "youll": "you'll", "youre": "you're", "youve": "you've"}
    manualMap = {'none': '0',
                 'zero': '0',
                 'one': '1',
                 'two': '2',
                 'three': '3',
                 'four': '4',
                 'five': '5',
                 'six': '6',
                 'seven': '7',
                 'eight': '8',
                 'nine': '9',
                 'ten': '10'
                 }
    articles = ['a', 'an', 'the']
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    s = s.strip().lower().split()
    tmp = []
    for e in s:
        if e in articles:
            continue
        elif e in contractions:
            tmp.append(contractions[e])
        elif e in manualMap:
            tmp.append(manualMap[e])
        else:
            tmp.append(e)
    s = wordlist2sentence(nltk.tokenize.word_tokenize(wordlist2sentence(tmp)))
    return s


def wordlist2sentence(wordlist):
    s = ''
    for i in range(len(wordlist)):
        s += wordlist[i]
        if i != len(wordlist) - 1:
            s += ' '
    return s


# def prob2synsetword(prob, synset=np.array(data.imagenet_classes.class_names), top=3):
#     prob = np.array(prob)
#     assert prob.ndim == 1
#     sort_id = np.argsort(prob, axis=-1)[::-1][0:top]
#     return [(synset[sort_id][i], prob[sort_id][i]) for i in range(len(sort_id))]

class ImageNet():
    def __init__(self):
        self.class_names = np.array(
            "tench, Tinca tinca\ngoldfish, Carassius auratus\ngreat white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias\ntiger shark, Galeocerdo cuvieri\nhammerhead, hammerhead shark\nelectric ray, crampfish, numbfish, torpedo\nstingray\ncock\nhen\nostrich, Struthio camelus\nbrambling, Fringilla montifringilla\ngoldfinch, Carduelis carduelis\nhouse finch, linnet, Carpodacus mexicanus\njunco, snowbird\nindigo bunting, indigo finch, indigo bird, Passerina cyanea\nrobin, American robin, Turdus migratorius\nbulbul\njay\nmagpie\nchickadee\nwater ouzel, dipper\nkite\nbald eagle, American eagle, Haliaeetus leucocephalus\nvulture\ngreat grey owl, great gray owl, Strix nebulosa\nEuropean fire salamander, Salamandra salamandra\ncommon newt, Triturus vulgaris\neft\nspotted salamander, Ambystoma maculatum\naxolotl, mud puppy, Ambystoma mexicanum\nbullfrog, Rana catesbeiana\ntree frog, tree-frog\ntailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui\nloggerhead, loggerhead turtle, Caretta caretta\nleatherback turtle, leatherback, leathery turtle, Dermochelys coriacea\nmud turtle\nterrapin\nbox turtle, box tortoise\nbanded gecko\ncommon iguana, iguana, Iguana iguana\nAmerican chameleon, anole, Anolis carolinensis\nwhiptail, whiptail lizard\nagama\nfrilled lizard, Chlamydosaurus kingi\nalligator lizard\nGila monster, Heloderma suspectum\ngreen lizard, Lacerta viridis\nAfrican chameleon, Chamaeleo chamaeleon\nKomodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis\nAfrican crocodile, Nile crocodile, Crocodylus niloticus\nAmerican alligator, Alligator mississipiensis\ntriceratops\nthunder snake, worm snake, Carphophis amoenus\nringneck snake, ring-necked snake, ring snake\nhognose snake, puff adder, sand viper\ngreen snake, grass snake\nking snake, kingsnake\ngarter snake, grass snake\nwater snake\nvine snake\nnight snake, Hypsiglena torquata\nboa constrictor, Constrictor constrictor\nrock python, rock snake, Python sebae\nIndian cobra, Naja naja\ngreen mamba\nsea snake\nhorned viper, cerastes, sand viper, horned asp, Cerastes cornutus\ndiamondback, diamondback rattlesnake, Crotalus adamanteus\nsidewinder, horned rattlesnake, Crotalus cerastes\ntrilobite\nharvestman, daddy longlegs, Phalangium opilio\nscorpion\nblack and gold garden spider, Argiope aurantia\nbarn spider, Araneus cavaticus\ngarden spider, Aranea diademata\nblack widow, Latrodectus mactans\ntarantula\nwolf spider, hunting spider\ntick\ncentipede\nblack grouse\nptarmigan\nruffed grouse, partridge, Bonasa umbellus\nprairie chicken, prairie grouse, prairie fowl\npeacock\nquail\npartridge\nAfrican grey, African gray, Psittacus erithacus\nmacaw\nsulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita\nlorikeet\ncoucal\nbee eater\nhornbill\nhummingbird\njacamar\ntoucan\ndrake\nred-breasted merganser, Mergus serrator\ngoose\nblack swan, Cygnus atratus\ntusker\nechidna, spiny anteater, anteater\nplatypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus\nwallaby, brush kangaroo\nkoala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus\nwombat\njellyfish\nsea anemone, anemone\nbrain coral\nflatworm, platyhelminth\nnematode, nematode worm, roundworm\nconch\nsnail\nslug\nsea slug, nudibranch\nchiton, coat-of-mail shell, sea cradle, polyplacophore\nchambered nautilus, pearly nautilus, nautilus\nDungeness crab, Cancer magister\nrock crab, Cancer irroratus\nfiddler crab\nking crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica\nAmerican lobster, Northern lobster, Maine lobster, Homarus americanus\nspiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish\ncrayfish, crawfish, crawdad, crawdaddy\nhermit crab\nisopod\nwhite stork, Ciconia ciconia\nblack stork, Ciconia nigra\nspoonbill\nflamingo\nlittle blue heron, Egretta caerulea\nAmerican egret, great white heron, Egretta albus\nbittern\ncrane\nlimpkin, Aramus pictus\nEuropean gallinule, Porphyrio porphyrio\nAmerican coot, marsh hen, mud hen, water hen, Fulica americana\nbustard\nruddy turnstone, Arenaria interpres\nred-backed sandpiper, dunlin, Erolia alpina\nredshank, Tringa totanus\ndowitcher\noystercatcher, oyster catcher\npelican\nking penguin, Aptenodytes patagonica\nalbatross, mollymawk\ngrey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus\nkiller whale, killer, orca, grampus, sea wolf, Orcinus orca\ndugong, Dugong dugon\nsea lion\nChihuahua\nJapanese spaniel\nMaltese dog, Maltese terrier, Maltese\nPekinese, Pekingese, Peke\nShih-Tzu\nBlenheim spaniel\npapillon\ntoy terrier\nRhodesian ridgeback\nAfghan hound, Afghan\nbasset, basset hound\nbeagle\nbloodhound, sleuthhound\nbluetick\nblack-and-tan coonhound\nWalker hound, Walker foxhound\nEnglish foxhound\nredbone\nborzoi, Russian wolfhound\nIrish wolfhound\nItalian greyhound\nwhippet\nIbizan hound, Ibizan Podenco\nNorwegian elkhound, elkhound\notterhound, otter hound\nSaluki, gazelle hound\nScottish deerhound, deerhound\nWeimaraner\nStaffordshire bullterrier, Staffordshire bull terrier\nAmerican Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier\nBedlington terrier\nBorder terrier\nKerry blue terrier\nIrish terrier\nNorfolk terrier\nNorwich terrier\nYorkshire terrier\nwire-haired fox terrier\nLakeland terrier\nSealyham terrier, Sealyham\nAiredale, Airedale terrier\ncairn, cairn terrier\nAustralian terrier\nDandie Dinmont, Dandie Dinmont terrier\nBoston bull, Boston terrier\nminiature schnauzer\ngiant schnauzer\nstandard schnauzer\nScotch terrier, Scottish terrier, Scottie\nTibetan terrier, chrysanthemum dog\nsilky terrier, Sydney silky\nsoft-coated wheaten terrier\nWest Highland white terrier\nLhasa, Lhasa apso\nflat-coated retriever\ncurly-coated retriever\ngolden retriever\nLabrador retriever\nChesapeake Bay retriever\nGerman short-haired pointer\nvizsla, Hungarian pointer\nEnglish setter\nIrish setter, red setter\nGordon setter\nBrittany spaniel\nclumber, clumber spaniel\nEnglish springer, English springer spaniel\nWelsh springer spaniel\ncocker spaniel, English cocker spaniel, cocker\nSussex spaniel\nIrish water spaniel\nkuvasz\nschipperke\ngroenendael\nmalinois\nbriard\nkelpie\nkomondor\nOld English sheepdog, bobtail\nShetland sheepdog, Shetland sheep dog, Shetland\ncollie\nBorder collie\nBouvier des Flandres, Bouviers des Flandres\nRottweiler\nGerman shepherd, German shepherd dog, German police dog, alsatian\nDoberman, Doberman pinscher\nminiature pinscher\nGreater Swiss Mountain dog\nBernese mountain dog\nAppenzeller\nEntleBucher\nboxer\nbull mastiff\nTibetan mastiff\nFrench bulldog\nGreat Dane\nSaint Bernard, St Bernard\nEskimo dog, husky\nmalamute, malemute, Alaskan malamute\nSiberian husky\ndalmatian, coach dog, carriage dog\naffenpinscher, monkey pinscher, monkey dog\nbasenji\npug, pug-dog\nLeonberg\nNewfoundland, Newfoundland dog\nGreat Pyrenees\nSamoyed, Samoyede\nPomeranian\nchow, chow chow\nkeeshond\nBrabancon griffon\nPembroke, Pembroke Welsh corgi\nCardigan, Cardigan Welsh corgi\ntoy poodle\nminiature poodle\nstandard poodle\nMexican hairless\ntimber wolf, grey wolf, gray wolf, Canis lupus\nwhite wolf, Arctic wolf, Canis lupus tundrarum\nred wolf, maned wolf, Canis rufus, Canis niger\ncoyote, prairie wolf, brush wolf, Canis latrans\ndingo, warrigal, warragal, Canis dingo\ndhole, Cuon alpinus\nAfrican hunting dog, hyena dog, Cape hunting dog, Lycaon pictus\nhyena, hyaena\nred fox, Vulpes vulpes\nkit fox, Vulpes macrotis\nArctic fox, white fox, Alopex lagopus\ngrey fox, gray fox, Urocyon cinereoargenteus\ntabby, tabby cat\ntiger cat\nPersian cat\nSiamese cat, Siamese\nEgyptian cat\ncougar, puma, catamount, mountain lion, painter, panther, Felis concolor\nlynx, catamount\nleopard, Panthera pardus\nsnow leopard, ounce, Panthera uncia\njaguar, panther, Panthera onca, Felis onca\nlion, king of beasts, Panthera leo\ntiger, Panthera tigris\ncheetah, chetah, Acinonyx jubatus\nbrown bear, bruin, Ursus arctos\nAmerican black bear, black bear, Ursus americanus, Euarctos americanus\nice bear, polar bear, Ursus Maritimus, Thalarctos maritimus\nsloth bear, Melursus ursinus, Ursus ursinus\nmongoose\nmeerkat, mierkat\ntiger beetle\nladybug, ladybeetle, lady beetle, ladybird, ladybird beetle\nground beetle, carabid beetle\nlong-horned beetle, longicorn, longicorn beetle\nleaf beetle, chrysomelid\ndung beetle\nrhinoceros beetle\nweevil\nfly\nbee\nant, emmet, pismire\ngrasshopper, hopper\ncricket\nwalking stick, walkingstick, stick insect\ncockroach, roach\nmantis, mantid\ncicada, cicala\nleafhopper\nlacewing, lacewing fly\ndragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk\ndamselfly\nadmiral\nringlet, ringlet butterfly\nmonarch, monarch butterfly, milkweed butterfly, Danaus plexippus\ncabbage butterfly\nsulphur butterfly, sulfur butterfly\nlycaenid, lycaenid butterfly\nstarfish, sea star\nsea urchin\nsea cucumber, holothurian\nwood rabbit, cottontail, cottontail rabbit\nhare\nAngora, Angora rabbit\nhamster\nporcupine, hedgehog\nfox squirrel, eastern fox squirrel, Sciurus niger\nmarmot\nbeaver\nguinea pig, Cavia cobaya\nsorrel\nzebra\nhog, pig, grunter, squealer, Sus scrofa\nwild boar, boar, Sus scrofa\nwarthog\nhippopotamus, hippo, river horse, Hippopotamus amphibius\nox\nwater buffalo, water ox, Asiatic buffalo, Bubalus bubalis\nbison\nram, tup\nbighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis\nibex, Capra ibex\nhartebeest\nimpala, Aepyceros melampus\ngazelle\nArabian camel, dromedary, Camelus dromedarius\nllama\nweasel\nmink\npolecat, fitch, foulmart, foumart, Mustela putorius\nblack-footed ferret, ferret, Mustela nigripes\notter\nskunk, polecat, wood pussy\nbadger\narmadillo\nthree-toed sloth, ai, Bradypus tridactylus\norangutan, orang, orangutang, Pongo pygmaeus\ngorilla, Gorilla gorilla\nchimpanzee, chimp, Pan troglodytes\ngibbon, Hylobates lar\nsiamang, Hylobates syndactylus, Symphalangus syndactylus\nguenon, guenon monkey\npatas, hussar monkey, Erythrocebus patas\nbaboon\nmacaque\nlangur\ncolobus, colobus monkey\nproboscis monkey, Nasalis larvatus\nmarmoset\ncapuchin, ringtail, Cebus capucinus\nhowler monkey, howler\ntiti, titi monkey\nspider monkey, Ateles geoffroyi\nsquirrel monkey, Saimiri sciureus\nMadagascar cat, ring-tailed lemur, Lemur catta\nindri, indris, Indri indri, Indri brevicaudatus\nIndian elephant, Elephas maximus\nAfrican elephant, Loxodonta africana\nlesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens\ngiant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca\nbarracouta, snoek\neel\ncoho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch\nrock beauty, Holocanthus tricolor\nanemone fish\nsturgeon\ngar, garfish, garpike, billfish, Lepisosteus osseus\nlionfish\npuffer, pufferfish, blowfish, globefish\nabacus\nabaya\nacademic gown, academic robe, judge's robe\naccordion, piano accordion, squeeze box\nacoustic guitar\naircraft carrier, carrier, flattop, attack aircraft carrier\nairliner\nairship, dirigible\naltar\nambulance\namphibian, amphibious vehicle\nanalog clock\napiary, bee house\napron\nashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin\nassault rifle, assault gun\nbackpack, back pack, knapsack, packsack, rucksack, haversack\nbakery, bakeshop, bakehouse\nbalance beam, beam\nballoon\nballpoint, ballpoint pen, ballpen, Biro\nBand Aid\nbanjo\nbannister, banister, balustrade, balusters, handrail\nbarbell\nbarber chair\nbarbershop\nbarn\nbarometer\nbarrel, cask\nbarrow, garden cart, lawn cart, wheelbarrow\nbaseball\nbasketball\nbassinet\nbassoon\nbathing cap, swimming cap\nbath towel\nbathtub, bathing tub, bath, tub\nbeach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon\nbeacon, lighthouse, beacon light, pharos\nbeaker\nbearskin, busby, shako\nbeer bottle\nbeer glass\nbell cote, bell cot\nbib\nbicycle-built-for-two, tandem bicycle, tandem\nbikini, two-piece\nbinder, ring-binder\nbinoculars, field glasses, opera glasses\nbirdhouse\nboathouse\nbobsled, bobsleigh, bob\nbolo tie, bolo, bola tie, bola\nbonnet, poke bonnet\nbookcase\nbookshop, bookstore, bookstall\nbottlecap\nbow\nbow tie, bow-tie, bowtie\nbrass, memorial tablet, plaque\nbrassiere, bra, bandeau\nbreakwater, groin, groyne, mole, bulwark, seawall, jetty\nbreastplate, aegis, egis\nbroom\nbucket, pail\nbuckle\nbulletproof vest\nbullet train, bullet\nbutcher shop, meat market\ncab, hack, taxi, taxicab\ncaldron, cauldron\ncandle, taper, wax light\ncannon\ncanoe\ncan opener, tin opener\ncardigan\ncar mirror\ncarousel, carrousel, merry-go-round, roundabout, whirligig\ncarpenter's kit, tool kit\ncarton\ncar wheel\ncash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM\ncassette\ncassette player\ncastle\ncatamaran\nCD player\ncello, violoncello\ncellular telephone, cellular phone, cellphone, cell, mobile phone\nchain\nchainlink fence\nchain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour\nchain saw, chainsaw\nchest\nchiffonier, commode\nchime, bell, gong\nchina cabinet, china closet\nChristmas stocking\nchurch, church building\ncinema, movie theater, movie theatre, movie house, picture palace\ncleaver, meat cleaver, chopper\ncliff dwelling\ncloak\nclog, geta, patten, sabot\ncocktail shaker\ncoffee mug\ncoffeepot\ncoil, spiral, volute, whorl, helix\ncombination lock\ncomputer keyboard, keypad\nconfectionery, confectionary, candy store\ncontainer ship, containership, container vessel\nconvertible\ncorkscrew, bottle screw\ncornet, horn, trumpet, trump\ncowboy boot\ncowboy hat, ten-gallon hat\ncradle\ncrane\ncrash helmet\ncrate\ncrib, cot\nCrock Pot\ncroquet ball\ncrutch\ncuirass\ndam, dike, dyke\ndesk\ndesktop computer\ndial telephone, dial phone\ndiaper, nappy, napkin\ndigital clock\ndigital watch\ndining table, board\ndishrag, dishcloth\ndishwasher, dish washer, dishwashing machine\ndisk brake, disc brake\ndock, dockage, docking facility\ndogsled, dog sled, dog sleigh\ndome\ndoormat, welcome mat\ndrilling platform, offshore rig\ndrum, membranophone, tympan\ndrumstick\ndumbbell\nDutch oven\nelectric fan, blower\nelectric guitar\nelectric locomotive\nentertainment center\nenvelope\nespresso maker\nface powder\nfeather boa, boa\nfile, file cabinet, filing cabinet\nfireboat\nfire engine, fire truck\nfire screen, fireguard\nflagpole, flagstaff\nflute, transverse flute\nfolding chair\nfootball helmet\nforklift\nfountain\nfountain pen\nfour-poster\nfreight car\nFrench horn, horn\nfrying pan, frypan, skillet\nfur coat\ngarbage truck, dustcart\ngasmask, respirator, gas helmet\ngas pump, gasoline pump, petrol pump, island dispenser\ngoblet\ngo-kart\ngolf ball\ngolfcart, golf cart\ngondola\ngong, tam-tam\ngown\ngrand piano, grand\ngreenhouse, nursery, glasshouse\ngrille, radiator grille\ngrocery store, grocery, food market, market\nguillotine\nhair slide\nhair spray\nhalf track\nhammer\nhamper\nhand blower, blow dryer, blow drier, hair dryer, hair drier\nhand-held computer, hand-held microcomputer\nhandkerchief, hankie, hanky, hankey\nhard disc, hard disk, fixed disk\nharmonica, mouth organ, harp, mouth harp\nharp\nharvester, reaper\nhatchet\nholster\nhome theater, home theatre\nhoneycomb\nhook, claw\nhoopskirt, crinoline\nhorizontal bar, high bar\nhorse cart, horse-cart\nhourglass\niPod\niron, smoothing iron\njack-o'-lantern\njean, blue jean, denim\njeep, landrover\njersey, T-shirt, tee shirt\njigsaw puzzle\njinrikisha, ricksha, rickshaw\njoystick\nkimono\nknee pad\nknot\nlab coat, laboratory coat\nladle\nlampshade, lamp shade\nlaptop, laptop computer\nlawn mower, mower\nlens cap, lens cover\nletter opener, paper knife, paperknife\nlibrary\nlifeboat\nlighter, light, igniter, ignitor\nlimousine, limo\nliner, ocean liner\nlipstick, lip rouge\nLoafer\nlotion\nloudspeaker, speaker, speaker unit, loudspeaker system, speaker system\nloupe, jeweler's loupe\nlumbermill, sawmill\nmagnetic compass\nmailbag, postbag\nmailbox, letter box\nmaillot\nmaillot, tank suit\nmanhole cover\nmaraca\nmarimba, xylophone\nmask\nmatchstick\nmaypole\nmaze, labyrinth\nmeasuring cup\nmedicine chest, medicine cabinet\nmegalith, megalithic structure\nmicrophone, mike\nmicrowave, microwave oven\nmilitary uniform\nmilk can\nminibus\nminiskirt, mini\nminivan\nmissile\nmitten\nmixing bowl\nmobile home, manufactured home\nModel T\nmodem\nmonastery\nmonitor\nmoped\nmortar\nmortarboard\nmosque\nmosquito net\nmotor scooter, scooter\nmountain bike, all-terrain bike, off-roader\nmountain tent\nmouse, computer mouse\nmousetrap\nmoving van\nmuzzle\nnail\nneck brace\nnecklace\nnipple\nnotebook, notebook computer\nobelisk\noboe, hautboy, hautbois\nocarina, sweet potato\nodometer, hodometer, mileometer, milometer\noil filter\norgan, pipe organ\noscilloscope, scope, cathode-ray oscilloscope, CRO\noverskirt\noxcart\noxygen mask\npacket\npaddle, boat paddle\npaddlewheel, paddle wheel\npadlock\npaintbrush\npajama, pyjama, pj's, jammies\npalace\npanpipe, pandean pipe, syrinx\npaper towel\nparachute, chute\nparallel bars, bars\npark bench\nparking meter\npassenger car, coach, carriage\npatio, terrace\npay-phone, pay-station\npedestal, plinth, footstall\npencil box, pencil case\npencil sharpener\nperfume, essence\nPetri dish\nphotocopier\npick, plectrum, plectron\npickelhaube\npicket fence, paling\npickup, pickup truck\npier\npiggy bank, penny bank\npill bottle\npillow\nping-pong ball\npinwheel\npirate, pirate ship\npitcher, ewer\nplane, carpenter's plane, woodworking plane\nplanetarium\nplastic bag\nplate rack\nplow, plough\nplunger, plumber's helper\nPolaroid camera, Polaroid Land camera\npole\npolice van, police wagon, paddy wagon, patrol wagon, wagon, black Maria\nponcho\npool table, billiard table, snooker table\npop bottle, soda bottle\npot, flowerpot\npotter's wheel\npower drill\nprayer rug, prayer mat\nprinter\nprison, prison house\nprojectile, missile\nprojector\npuck, hockey puck\npunching bag, punch bag, punching ball, punchball\npurse\nquill, quill pen\nquilt, comforter, comfort, puff\nracer, race car, racing car\nracket, racquet\nradiator\nradio, wireless\nradio telescope, radio reflector\nrain barrel\nrecreational vehicle, RV, R.V.\nreel\nreflex camera\nrefrigerator, icebox\nremote control, remote\nrestaurant, eating house, eating place, eatery\nrevolver, six-gun, six-shooter\nrifle\nrocking chair, rocker\nrotisserie\nrubber eraser, rubber, pencil eraser\nrugby ball\nrule, ruler\nrunning shoe\nsafe\nsafety pin\nsaltshaker, salt shaker\nsandal\nsarong\nsax, saxophone\nscabbard\nscale, weighing machine\nschool bus\nschooner\nscoreboard\nscreen, CRT screen\nscrew\nscrewdriver\nseat belt, seatbelt\nsewing machine\nshield, buckler\nshoe shop, shoe-shop, shoe store\nshoji\nshopping basket\nshopping cart\nshovel\nshower cap\nshower curtain\nski\nski mask\nsleeping bag\nslide rule, slipstick\nsliding door\nslot, one-armed bandit\nsnorkel\nsnowmobile\nsnowplow, snowplough\nsoap dispenser\nsoccer ball\nsock\nsolar dish, solar collector, solar furnace\nsombrero\nsoup bowl\nspace bar\nspace heater\nspace shuttle\nspatula\nspeedboat\nspider web, spider's web\nspindle\nsports car, sport car\nspotlight, spot\nstage\nsteam locomotive\nsteel arch bridge\nsteel drum\nstethoscope\nstole\nstone wall\nstopwatch, stop watch\nstove\nstrainer\nstreetcar, tram, tramcar, trolley, trolley car\nstretcher\nstudio couch, day bed\nstupa, tope\nsubmarine, pigboat, sub, U-boat\nsuit, suit of clothes\nsundial\nsunglass\nsunglasses, dark glasses, shades\nsunscreen, sunblock, sun blocker\nsuspension bridge\nswab, swob, mop\nsweatshirt\nswimming trunks, bathing trunks\nswing\nswitch, electric switch, electrical switch\nsyringe\ntable lamp\ntank, army tank, armored combat vehicle, armoured combat vehicle\ntape player\nteapot\nteddy, teddy bear\ntelevision, television system\ntennis ball\nthatch, thatched roof\ntheater curtain, theatre curtain\nthimble\nthresher, thrasher, threshing machine\nthrone\ntile roof\ntoaster\ntobacco shop, tobacconist shop, tobacconist\ntoilet seat\ntorch\ntotem pole\ntow truck, tow car, wrecker\ntoyshop\ntractor\ntrailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi\ntray\ntrench coat\ntricycle, trike, velocipede\ntrimaran\ntripod\ntriumphal arch\ntrolleybus, trolley coach, trackless trolley\ntrombone\ntub, vat\nturnstile\ntypewriter keyboard\numbrella\nunicycle, monocycle\nupright, upright piano\nvacuum, vacuum cleaner\nvase\nvault\nvelvet\nvending machine\nvestment\nviaduct\nviolin, fiddle\nvolleyball\nwaffle iron\nwall clock\nwallet, billfold, notecase, pocketbook\nwardrobe, closet, press\nwarplane, military plane\nwashbasin, handbasin, washbowl, lavabo, wash-hand basin\nwasher, automatic washer, washing machine\nwater bottle\nwater jug\nwater tower\nwhiskey jug\nwhistle\nwig\nwindow screen\nwindow shade\nWindsor tie\nwine bottle\nwing\nwok\nwooden spoon\nwool, woolen, woollen\nworm fence, snake fence, snake-rail fence, Virginia fence\nwreck\nyawl\nyurt\nweb site, website, internet site, site\ncomic book\ncrossword puzzle, crossword\nstreet sign\ntraffic light, traffic signal, stoplight\nbook jacket, dust cover, dust jacket, dust wrapper\nmenu\nplate\nguacamole\nconsomme\nhot pot, hotpot\ntrifle\nice cream, icecream\nice lolly, lolly, lollipop, popsicle\nFrench loaf\nbagel, beigel\npretzel\ncheeseburger\nhotdog, hot dog, red hot\nmashed potato\nhead cabbage\nbroccoli\ncauliflower\nzucchini, courgette\nspaghetti squash\nacorn squash\nbutternut squash\ncucumber, cuke\nartichoke, globe artichoke\nbell pepper\ncardoon\nmushroom\nGranny Smith\nstrawberry\norange\nlemon\nfig\npineapple, ananas\nbanana\njackfruit, jak, jack\ncustard apple\npomegranate\nhay\ncarbonara\nchocolate sauce, chocolate syrup\ndough\nmeat loaf, meatloaf\npizza, pizza pie\npotpie\nburrito\nred wine\nespresso\ncup\neggnog\nalp\nbubble\ncliff, drop, drop-off\ncoral reef\ngeyser\nlakeside, lakeshore\npromontory, headland, head, foreland\nsandbar, sand bar\nseashore, coast, seacoast, sea-coast\nvalley, vale\nvolcano\nballplayer, baseball player\ngroom, bridegroom\nscuba diver\nrapeseed\ndaisy\nyellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum\ncorn\nacorn\nhip, rose hip, rosehip\nbuckeye, horse chestnut, conker\ncoral fungus\nagaric\ngyromitra\nstinkhorn, carrion fungus\nearthstar\nhen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa\nbolete\near, spike, capitulum\ntoilet tissue, toilet paper, bathroom tissue".split(
                '\n'))

    def prob2classnames(self, prob, top=3):
        prob = np.array(prob)
        assert prob.ndim == 1
        sort_id = np.argsort(prob, axis=-1)[::-1][0:top]
        return [(self.class_names[sort_id][i], prob[sort_id][i]) for i in range(len(sort_id))]


def xnor(x, y):
    if (x and y) or (not x and not y):
        return True
    else:
        return False


def print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def expand_list(l, length, fill=0, direction='right'):
    tmp = [fill] * length
    if direction == 'left':
        tmp[0:len(l)] = l[0: length]
        return tmp
    elif direction == 'right':
        tmp[-len(l):] = l[0: length]
        return tmp
    else:
        raise ValueError("diretion can only be left or right")


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_dense = np.array(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def mul(l):
    return functools.reduce(operator.mul, l)


# 模型常用
class Vocabulary(object):
    def __init__(self, original_wordlist, special_wordlist=None, vocabulary_size=None, min_word_count=None,
                 name='',
                 printable=True):
        # min_word_count 将会覆盖vocabulary_size
        # special_wordlist 第一个值必须表示padding word, 第二个值必须表示unkown word
        # 一般地，PAD index 0, UNK index 1

        self._original_wordlist = original_wordlist
        self._special_wordlist = special_wordlist
        self._vocabulary_size = vocabulary_size
        self._min_word_count = min_word_count
        self._name = name
        self._printable = printable

        self._try = None
        self._word2idx = None
        self._idx2word = None
        self._size = None
        self._vocabulary_wordlist = None
        self._build_vocabulary()

    def _build_vocabulary(self):
        if self._printable:
            print("[info] utils.build_vocabulary: ==Start building vocabulary %s ==" % self._name)

        counter = collections.Counter(self._original_wordlist)
        sorted_count = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        # 若指定special_wordlist, 则special_wordlist中的字符不参加排序（从排序中去除）
        if self._special_wordlist:
            tmp = []
            for k, v in sorted_count:
                if k not in self._special_wordlist:
                    tmp.append((k, v))
                else:
                    if self._printable:
                        print(
                            "[warning] utils.build_vocabulary: special_wordlist element %s is in original_wordlist" % k)
            sorted_count = tmp
        self._try_size = []
        for tmpi in range(1, 100):
            sorted_count_part = [e for e in sorted_count if e[1] >= tmpi]
            self._try_size.append(
                (tmpi, len(sorted_count_part),
                 "%.3f%%" % (sum([e[1] for e in sorted_count_part]) / len(self._original_wordlist) * 100)))
        if self._printable:
            tmp = "[info] utils.build_vocabulary: trying_min_word_count1~5: %s" % self._try_size[0:5]
            if self._special_wordlist:
                tmp += " + %d" % len(self._special_wordlist)
            print(tmp)
        # print(self._try_size)

        # 按照要求截取字典
        if self._min_word_count:
            self._vocabulary_size = len([e for e in sorted_count if e[1] >= self._min_word_count])
            if self._special_wordlist:
                self._vocabulary_size += len(self._special_wordlist)
        elif not self._vocabulary_size:
            self._vocabulary_size = len(sorted_count)
            if self._special_wordlist:
                self._vocabulary_size += len(self._special_wordlist)

        sorted_final = [k for k, v in sorted_count]
        if self._special_wordlist:
            sorted_final = self._special_wordlist + sorted_final

        sorted_final = sorted_final[:self._vocabulary_size]
        self._word2idx = {e: i for i, e in enumerate(sorted_final)}
        self._idx2word = {v: k for k, v in self._word2idx.items()}
        self._vocabulary_wordlist = sorted_final

        if self._printable:
            self.printinfo()
            print("[info] utils.build_vocabulary: ===Ended building vocabulary %s ==" % self._name)

    def printinfo(self):
        print("[info] utils.build_vocabulary: original_wordlist_size: %d" % len(self._original_wordlist))
        print("[info] utils.build_vocabulary: original_wordset_size: %d" % len(set(self._original_wordlist)))
        print("[info] utils.build_vocabulary: target vocabulary_size: %d" % self._vocabulary_size)

    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            if self._special_wordlist:
                return 1
            else:
                raise ValueError(
                    '[error] utils.Vocabulary.word2idx: Word %s is not in Vocabulary %s and you did not set UNK' % (
                        word, self._name))

    def idx2word(self, idx):
        if idx in self._idx2word:
            return self._idx2word[idx]
        else:
            raise ValueError('[error] tools.Dictionary: %s is not a vocab index.' % idx)

    def wordlist2idxlist(self, wordlist):
        return [self.word2idx(e) for e in wordlist]

    def idxlist2wordlist(self, idxlist):
        return [self.idx2word(e) for e in idxlist]

    def wordlist2sparselist(self, wordlist):
        sparselist = np.zeros([self._vocabulary_size], dtype=np.uint8)
        for e in wordlist:
            sparselist[self.word2idx(e)] = 1
        return sparselist

    def get_idlist_length(self, idlist):
        assert self._special_wordlist
        return idlist.index(0)

    def get_original_idxlist(self, one_hot=False):
        idxlist = [self.word2idx(e) for e in self._original_wordlist]
        if one_hot:
            idxlist = np.array([np.identity(self._vocabulary_size)[idx] for idx in idxlist])
        return idxlist

    def __len__(self):
        return self._vocabulary_size


class DataSetNew():
    def __init__(self, inputs, batch_size=1, shuffle=False, return_last=False):
        '''

        :param inputs:
        :param batch_size:
        :param shuffle:
        :param return_last:
        when len(inputs) % batch_size != 0, return_last is useful. we choose if return the last minor batch.
        when len(inputs) % batch_size == 0, return_last is not useful, because we always return the last batch.

        when training, set shuffle=True and return_last=False
        when testing, set shuffle=False and return_last=True

        you can use the following loop:
        for epoch in range(num_epoch):
            for batch_idx in range(dataset.num_batches):
            batch = dataset.next_batch()

        '''
        self.inputs = np.array(inputs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_last = return_last
        self.count = len(inputs)
        self.indices = np.arange(self.count)
        self.current_index = 0
        if return_last:
            self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        else:
            self.num_batches = int(np.floor(self.count * 1.0 / self.batch_size))
        self.reset()

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        if not self.has_next_batch():
            self.reset()
        excerpt = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        batch_dict = collections.defaultdict(list)
        for example_dict in self.inputs[excerpt]:
            for k, v in example_dict.items():
                batch_dict[k].append(v)
        return dict(batch_dict)

    def has_next_batch(self):
        if self.return_last:
            return self.current_index < self.count
        else:
            return self.current_index + self.batch_size <= self.count


def func_name():
    return sys._getframe(1).f_code.co_name


def class_name(self):
    return self.__class__.__name__


def update_python(fileordirname=None):
    if not fileordirname:
        fileordirname = os.getcwd()
    os.system('2to3 {0} -w'.format(fileordirname))


def update_tensorflow(fileordirname=None):
    if not fileordirname:
        fileordirname = os.getcwd()
    tool_filename = 'D:/programs/tensorflow/tensorflow/tools/compatibility/tf_upgrade.py'
    if os.path.isfile(fileordirname):
        os.system('python {0} --infile {1} --outfile {2}'.format(tool_filename, fileordirname, fileordirname))
    else:
        for parent, dirnames, filenames in os.walk(fileordirname):
            for filename in filenames:
                d, b = os.path.split(filename)
                try:
                    f, a = b.split('.', maxsplit=1)
                    if a == 'py':
                        target = os.path.join(parent, filename)
                        os.system('python3 {0} --infile {1} --outfile {2}'.format(tool_filename, target, target))
                        shutil.move('report.txt', os.path.join(parent, '{0}.report'.format(filename)))
                except Exception as e:
                    pass


def execute(cmd):
    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # if wait:
    #     for line in iter(p.stdout.readline, ''):
    #         if line:
    #             print(line.decode('utf-8').replace('\n', ''))
    #         else:
    #             p.kill()
    #             break

    output = p.communicate()[0].decode('utf-8')
    return output


def get_passwd():
    while True:
        passwd = input("Enter cqwuchenfei@163.com passwd: ")
        hash = '$6$rounds=656000$Vudi1qMpjSjN0pRt$LgwzawG/RAyBWFC5a2zc8FUPoi9ewWc3F2tcmFlKPjC' + \
               '.wLITyFAdMpEst16YLIyYm6RAguuDpkI0IIQNZ9rmf.'
        if sha512_crypt.verify(passwd, hash):
            return passwd
        else:
            print("[utils.warning] Wrong passwd.")


def email(subject, contents, passwd):
    yag = yagmail.SMTP(user='cqwuchenfei@163.com', password=passwd, host='smtp.163.com')
    yag.send('2533816498@qq.com', subject=subject, contents=contents)


def merge_tuple_list(tuple_list, fn=None):
    # 整合形如[('grape', 100), ('grape', 3), ('apple', 15), ('apple', 10), ('apple', 4), ('banana', 3)]
    # 为[('grape', [100, 3]), ('apple', [15, 10, 4]), ('banana', [3])]
    tuple_list = sorted(tuple_list, key=lambda x: x[0], reverse=True)
    if fn:
        return [(key, fn([num for _, num in value])) for key, value in
                itertools.groupby(tuple_list, lambda x: x[0])]
    else:
        return [(key, [num for _, num in value]) for key, value in
                itertools.groupby(tuple_list, lambda x: x[0])]


# format

def format_img(fileordirname, type='jpg'):
    fileordirname = os.path.abspath(fileordirname)

    def format_a_img(filename):
        dirname, rootname, extname = split_filepath(filename)
        if extname in ['jpg', 'png', 'jpeg']:
            rawtype = imghdr.what(filename)
            if rawtype == 'jpeg':
                rawtype = 'jpg'
            if rawtype != type:
                dirname, rootname, extname = split_filepath(filename)
                img = Image.open(filename)
                img.save(os.path.join(dirname, rootname + '.' + type))
                print('[info] utils.format_img: formatting image file %s from %s to %s' % (
                    os.path.abspath(filename), rawtype, type))

    if os.path.isfile(fileordirname):
        format_a_img(fileordirname)
    elif os.path.isdir(fileordirname):
        for parent, dirnames, filenames in tqdm(os.walk(fileordirname)):
            for filename in filenames:
                target = os.path.join(parent, filename)
                format_a_img(target)


def get_log():
    import logging
    from colorlog import ColoredFormatter
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(message)s",
        #    datefmt='%H:%M:%S.%f',
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white,bold',
            'INFOV': 'cyan,bold',
            'WARNING': 'yellow',
            'ERROR': 'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger('rn')
    log.setLevel(logging.DEBUG)
    log.handlers = []  # No duplicated handlers
    log.propagate = False  # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')

    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    logging.Logger.infov = _infov

    return log


# 排列
def get_permutation(height, width):
    """ Get the permutation corresponding to a snake-like walk as decribed by the paper. Used to flatten the convolutional feats. """
    permutation = np.zeros(height * width, np.int32)
    for i in range(height):
        for j in range(width):
            permutation[i * width + j] = i * width + j if i % 2 == 0 else (i + 1) * width - j - 1
    return permutation


# 判别
def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


if __name__ == '__main__':
    pass
    # sf = Stanford()
    # t = sf.question_to_parsertree(["What color is the man's shirt"]*10000)
    # ts = t.split('\n\n')
    # print("Done!")
