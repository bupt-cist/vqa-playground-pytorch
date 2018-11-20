import gc
import lda
import torch.utils.data as data
from nltk.parse.stanford import StanfordDependencyParser
from putils import *
import redis
import subprocess


class VQA(object):
    def __init__(self, data_dir, process_dir, version, samplingans=True, vgenome=False, tdiuc=False, clevr=False,
                 version1_multiple_choices=False, use_pos=False, box100=False):
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.version = version
        self.split = None
        self.samplingans = samplingans
        self.vgenome = vgenome
        self.tdiuc = tdiuc
        self.clevr = clevr
        self.box100 = box100
        self.version1_multiple_choices = version1_multiple_choices
        self.use_pos = use_pos

        if self.version not in [1, 2]:
            raise ValueError('version can only be 1 or 2, but got %s' % self.version)

        self.data = dict()
        # TODO you must add next line!!!
        # self.download()
        self.preprocess_vgenome()
        self.preprocess_clevr()
        self.preprocess()

    def transform_clevr(self, raw_que, split):
        train_que = {}
        train_que['license'] = raw_que['info']['license']
        train_que['info'] = raw_que['info']
        train_que['data_type'] = 'mscoco'
        train_que['data_subtype'] = 'train2014'
        train_que['task_type'] = 'Open-Ended'
        train_que['questions'] = []
        for e in raw_que['questions']:
            que = {}
            que['question'] = e['question']
            que['question_id'] = e['question_index']
            que['image_id'] = e['image_index']
            train_que['questions'].append(que)
        if split != 'test':
            train_ann = {}
            train_ann['license'] = raw_que['info']['license']
            train_ann['info'] = raw_que['info']
            train_ann['data_type'] = 'mscoco'
            train_ann['data_subtype'] = 'train2014'
            train_ann['task_type'] = 'Open-Ended'
            train_ann['annotations'] = []
            for e in raw_que['questions']:
                ann = {}
                ann['answer_type'] = None
                ann['question_type'] = None
                ann['question_id'] = e['question_index']
                ann['image_id'] = e['image_index']
                ann['multiple_choice_answer'] = e['answer']
                ann['answers'] = [{'answer_id': 1, 'answer_confidence': 'yes', 'answer': e['answer']}]
                ann['image_filename'] = e['image_filename']
                ann['program'] = e['program']
                ann['question_family_id'] = e['question_family_index']
                ann['split'] = e['split']
                train_ann['annotations'].append(ann)
        else:
            train_ann = None
        return train_que, train_ann

    def preprocess_clevr(self):
        """
            rewrite the vgenome question-answer dataset by coco format
            vgenome_question_answers.json => vgenome_annotations.json + vgenome_questions.json
        """
        if not self.clevr:
            return
        train_ann_filename = os.path.join(self.data_dir, 'clevr_mscoco_train2014_annotations.json')
        val_ann_filename = os.path.join(self.data_dir, 'clevr_mscoco_val2014_annotations.json')
        train_que_filename = os.path.join(self.data_dir, 'clevr_mscoco_train2014_questions.json')
        val_que_filename = os.path.join(self.data_dir, 'clevr_mscoco_val2014_questions.json')
        test_que_filename = os.path.join(self.data_dir, 'clevr_mscoco_test2015_questions.json')
        testdev_que_filename = os.path.join(self.data_dir, 'clevr_mscoco_test-dev2015_questions.json')
        if not os.path.exists(train_ann_filename) or not os.path.exists(train_que_filename):
            raw_train = file2data(os.path.join(self.data_dir, 'clevr_question/CLEVR_train_questions.json'))
            raw_val = file2data(os.path.join(self.data_dir, 'clevr_question/CLEVR_val_questions.json'))
            raw_test = file2data(os.path.join(self.data_dir, 'clevr_question/CLEVR_test_questions.json'))

            train_que, train_ann = self.transform_clevr(raw_train, 'train')
            val_que, val_ann = self.transform_clevr(raw_val, 'val')
            test_que, _ = self.transform_clevr(raw_test, 'test')

            data2file(train_ann, train_ann_filename)
            data2file(val_ann, val_ann_filename)
            data2file(train_que, train_que_filename)
            data2file(val_que, val_que_filename)
            data2file(test_que, test_que_filename)
            data2file(test_que, testdev_que_filename)

    def preprocess_vgenome(self):
        """
            rewrite the vgenome question-answer dataset by coco format 
            vgenome_question_answers.json => vgenome_annotations.json + vgenome_questions.json
        """
        if not self.vgenome:
            return
        annotations_filename = os.path.join(self.data_dir, 'vgenome_annotations.json')
        questions_filename = os.path.join(self.data_dir, 'vgenome_questions.json')
        if os.path.exists(annotations_filename) and os.path.exists(questions_filename):
            return

        vgenome_filenames = list_filenames(os.path.join(self.data_dir, 'vgenome'))
        vgenome_imageids = set(map(lambda x: int(x.split("/")[-1].split(".")[0]), vgenome_filenames))
        vgenome = file2data(os.path.join(self.data_dir, 'vgenome_question_answers.json'))
        annotations = {'annotations': list()}
        questions = {'questions': list()}
        # TODO: this way is tricky
        coco = file2data(os.path.join(self.process_dir, 'version_1,nans_2000,splitnum_3,mwc_0,mql_16.h5'))
        coco_answer_vocab = coco['a_vocab']
        coco_answer_words = set(coco_answer_vocab._word2idx.keys())
        for image in vgenome:
            image_id = image['id']
            if image_id not in vgenome_imageids:
                continue
            qas = image['qas']
            for qa in qas:
                #  assert(image_id == qa['image_id'], "image_id=%s not equal to qa_image_id=%s" % (image_id, qa['image_id']))
                anno = {'image_id': image_id, 'question_id': qa['qa_id']}
                ques = {'image_id': image_id, 'question_id': qa['qa_id'], 'question': qa['question']}
                anno['answer_type'] = 'unknown'  # just set this value 'other'
                anno[
                    'question_type'] = 'unknown'  # qa['question'].split()[0] # just set this value with the first value of question
                ans = qa['answer']
                ans = ans.strip().strip('.').lower()
                anno['multiple_choice_answer'] = ans
                if ans not in coco_answer_words:
                    continue
                answer = {'answer': ans, 'answer_confidence': 'yes', 'answer_id': 1}
                anno['answers'] = list()
                for i in range(10):
                    answer['answer_id'] = i + 1
                    anno['answers'].append(answer)
                annotations['annotations'].append(anno)
                questions['questions'].append(ques)
        print("<datasets.py> vgenome has %s question answer pairs", len(questions['questions']))
        data2file(annotations, annotations_filename)
        data2file(questions, questions_filename)

    def download(self):
        # 下载VQA数据集中的图片

        img_url_to_targets = {'http://msvocds.blob.core.windows.net/coco2014/train2014.zip': ['train2014'],
                              'http://msvocds.blob.core.windows.net/coco2014/val2014.zip': ['val2014'],
                              'http://msvocds.blob.core.windows.net/coco2015/test2015.zip': ['test2015']}

        # 下载VQA数据集中的问题和答案
        if self.version == 1:
            qa_url_to_targets = {'http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip':
                                     ['OpenEnded_mscoco_train2014_questions.json',
                                      'MultipleChoice_mscoco_train2014_questions.json'],
                                 'http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip':
                                     ['OpenEnded_mscoco_val2014_questions.json',
                                      'MultipleChoice_mscoco_val2014_questions.json'],
                                 'http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip':
                                     ['OpenEnded_mscoco_test-dev2015_questions.json',
                                      'OpenEnded_mscoco_test2015_questions.json',
                                      'MultipleChoice_mscoco_test-dev2015_questions.json',
                                      'MultipleChoice_mscoco_test2015_questions.json'],
                                 'http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip':
                                     ['mscoco_train2014_annotations.json'],
                                 'http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip':
                                     ['mscoco_val2014_annotations.json']}
        else:
            qa_url_to_targets = {'http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip':
                                     ['v2_OpenEnded_mscoco_train2014_questions.json'],
                                 'http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip':
                                     ['v2_OpenEnded_mscoco_val2014_questions.json'],
                                 'http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip':
                                     ['v2_OpenEnded_mscoco_test-dev2015_questions.json',
                                      'v2_OpenEnded_mscoco_test2015_questions.json'],
                                 'http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip':
                                     ['v2_mscoco_train2014_annotations.json'],
                                 'http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip':
                                     ['v2_mscoco_val2014_annotations.json']}
        url_to_targets = {**img_url_to_targets, **qa_url_to_targets}

        for url, targets in url_to_targets.items():
            for target in targets:
                if not os.path.exists(os.path.join(self.data_dir, target)):
                    print(
                        '<datasets.py> VQA.download: Did not find expected file %s, downloading now.' % os.path.abspath(
                            os.path.join(
                                self.data_dir, target)))
                    filename = download_file(url, self.data_dir)
                    extract_file(filename)
        print('<datasets.py> VQA.download: All file prepared.')

    def preprocess(self, override=False):
        if self.version1_multiple_choices:
            # force to override while evaluation multiple choice answer
            override = True
        if self.vgenome:
            filename = os.path.join(self.process_dir, 'vgenome_version_%s.json' % self.version)
        elif self.tdiuc:
            filename = os.path.join(self.process_dir, 'tdiuc_version_%s.json' % self.version)
        elif self.clevr:
            filename = os.path.join(self.process_dir, 'clevr_version_%s.json' % self.version)
        else:
            filename = os.path.join(self.process_dir, 'version_%s.json' % self.version)
        if not os.path.exists(filename) or override:
            if self.tdiuc:
                flag = "tdiuc_"
                train_a = file2data(
                    os.path.join(self.data_dir, '%smscoco_train2014_annotations.json' % flag))
                val_a = file2data(
                    os.path.join(self.data_dir, '%smscoco_val2014_annotations.json' % flag))
                if self.version1_multiple_choices:
                    eval_metric = 'MultipleChoice'
                else:
                    eval_metric = 'OpenEnded'
                train_q = file2data(
                    os.path.join(self.data_dir, '%s%s_mscoco_train2014_questions.json' % (flag, eval_metric)))
                val_q = file2data(
                    os.path.join(self.data_dir, '%s%s_mscoco_val2014_questions.json' % (flag, eval_metric)))

                raw_data = {'train': (train_q['questions'], train_a['annotations'], 'train2014'),
                            'val': (val_q['questions'], val_a['annotations'], 'val2014'),
                            }

                self.data['raw'] = {'train': [],
                                    'val': [], }

                for split, (q_list, a_list, flag) in raw_data.items():
                    for i in range(len(q_list)):
                        if split == 'train_vgenome':
                            split = 'train'
                        img_filename = os.path.join("/root/data/VQA/download/TDIUC_Images", '%s/COCO_%s_%012d.jpg'
                                                    % (flag, flag, q_list[i]['image_id']))
                        if a_list:
                            self.data['raw'][split].append({
                                'q_id': q_list[i]['question_id'],
                                'q': q_list[i]['question'],
                                'img_filename': img_filename,
                                'a_10': sorted(
                                    collections.Counter([e['answer'] for e in a_list[i]['answers']]).items(),
                                    key=lambda x: (-x[1], x[0])),
                                'a_word': a_list[i]['answers'][0]['answer'],
                                # TODO: Don't have multiple chioce answer
                            })
                        else:
                            self.data['raw'][split].append({
                                'q_id': q_list[i]['question_id'],
                                'q': q_list[i]['question'],
                                'img_filename': img_filename,
                            })
                        if self.version1_multiple_choices:
                            self.data['raw'][split][-1].update({'a_mc': q_list[i]['multiple_choices']})
                data2file(self.data['raw'], filename, override=override)
            else:
                if self.version1_multiple_choices:
                    eval_metric = 'MultipleChoice_'
                else:
                    eval_metric = 'OpenEnded_'
                if self.clevr:
                    flag = "clevr_"
                    eval_metric = ""
                else:
                    flag = "" if self.version == 1 else "v2_"
                train_a = file2data(
                    os.path.join(self.data_dir, '%smscoco_train2014_annotations.json' % flag))
                val_a = file2data(
                    os.path.join(self.data_dir, '%smscoco_val2014_annotations.json' % flag))

                train_q = file2data(
                    os.path.join(self.data_dir, '%s%smscoco_train2014_questions.json' % (flag, eval_metric)))
                val_q = file2data(
                    os.path.join(self.data_dir, '%s%smscoco_val2014_questions.json' % (flag, eval_metric)))
                test_q = file2data(
                    os.path.join(self.data_dir, '%s%smscoco_test2015_questions.json' % (flag, eval_metric)))
                test_dev_q = file2data(
                    os.path.join(self.data_dir, '%s%smscoco_test-dev2015_questions.json' % (flag, eval_metric)))
                if self.vgenome:
                    vgenome_a = file2data(os.path.join(self.data_dir, 'vgenome_annotations.json'))
                    vgenome_q = file2data(os.path.join(self.data_dir, 'vgenome_questions.json'))

                raw_data = {'train': (train_q['questions'], train_a['annotations'], 'train2014'),
                            'val': (val_q['questions'], val_a['annotations'], 'val2014'),
                            'test': (test_q['questions'], None, 'test2015'),
                            'test_dev': (test_dev_q['questions'], None, 'test2015'),
                            }
                if self.vgenome:
                    raw_data.update({'train_vgenome': (vgenome_q['questions'], vgenome_a['annotations'], 'vgenome')})

                self.data['raw'] = {'train': [],
                                    'val': [],
                                    'test': [],
                                    'test_dev': []}

                for split, (q_list, a_list, flag) in raw_data.items():
                    for i in range(len(q_list)):
                        if split == 'train_vgenome':
                            split = 'train'
                        if flag == 'vgenome':
                            img_filename = os.path.join("/root/data/VQA/download", '%s/%d.jpg'
                                                        % (flag, q_list[i]['image_id']))
                        if self.clevr:
                            img_filename = "CLEVR_v1.0/images/%s/CLEVR_%s_%06d.png" % (
                                split, split, q_list[i]['image_id'])
                        else:
                            img_filename = os.path.join("/root/data/VQA/download", '%s/COCO_%s_%012d.jpg'
                                                        % (flag, flag, q_list[i]['image_id']))
                        if a_list:
                            self.data['raw'][split].append({
                                'q_id': q_list[i]['question_id'],
                                'q': q_list[i]['question'],
                                'img_filename': img_filename,
                                'a_word': a_list[i]['multiple_choice_answer'],
                                'a_10': sorted(
                                    collections.Counter([e['answer'] for e in a_list[i]['answers']]).items(),
                                    key=lambda x: (-x[1], x[0])),
                            })
                        else:
                            self.data['raw'][split].append({
                                'q_id': q_list[i]['question_id'],
                                'q': q_list[i]['question'],
                                'img_filename': img_filename,
                            })
                        if self.version1_multiple_choices:
                            self.data['raw'][split][-1].update({'a_mc': q_list[i]['multiple_choices']})
                data2file(self.data['raw'], filename, override=override)
        else:
            self.data['raw'] = file2data(filename)

    def process_img(self, arch='fbresnet152', size=224, override=False, load_mem=None, load_splits=None,
                    pos_version='wh'):
        '''

        :param arch:
        :param size:
        :param override:
        :param load_mem_split: should be a list. for example, ['train', 'val']
        :return:
        '''
        if self.vgenome:
            hy_filename = 'cocovg,size,{}_arch,{}.hy'.format(arch, size)
            txt_filename = 'cocovg,size,{}_arch,{}.txt'.format(arch, size)
            pos_filename = 'cocovg,size,{}_arch,{},pos,{}.hy'.format(arch, size, pos_version)
            load_splits += ['vgenome']
        elif self.box100:
            hy_filename = 'box100,size,{}_arch,{}.hy'.format(arch, size)
            pos_filename = 'box100,size,{}_arch,{},pos,{}.hy'.format(arch, size, pos_version)
            txt_filename = 'box100,size,{}_arch,{}.txt'.format(arch, size)
        elif self.clevr:
            if arch == 'fbresnet152':
                arch = 'resnet101'
            hy_filename = 'clevr,size,{}_arch,{}.hy'.format(arch, size)
            pos_filename = 'clevr,size,{}_arch,{},pos,{}.hy'.format(arch, size, pos_version)
            txt_filename = 'clevr,size,{}_arch,{}.txt'.format(arch, size)
        elif self.tdiuc:
            hy_filename = 'tdiuc,size,{}_arch,{}.hy'.format(arch, size)
            pos_filename = 'tdiuc,size,{}_arch,{},pos,{}.hy'.format(arch, size, pos_version)
            txt_filename = 'tdiuc,size,{}_arch,{}.txt'.format(arch, size)
        else:
            hy_filename = 'size,{}_arch,{}.hy'.format(arch, size)
            pos_filename = 'size,{}_arch,{},pos,{}.hy'.format(arch, size, pos_version)
            txt_filename = 'size,{}_arch,{}.txt'.format(arch, size)

        if os.path.exists("/root/data/VQA/preprocess/"):
            extract_hy_filename = os.path.join("/root/data/VQA/preprocess/", hy_filename)
        else:
            extract_hy_filename = os.path.join(self.process_dir, hy_filename)
        extract_pos_filename = os.path.join(self.process_dir, pos_filename)
        extract_txt_filename = os.path.join(self.process_dir, txt_filename)

        if arch != 'fbresnet152' and self.data.get('img'):
            del self.data['img']
        gc.collect()

        if not os.path.exists(extract_hy_filename) or not os.path.exists(extract_txt_filename) or override:
            print('[warning] mydatasets.py: Did not find extracted feature in  : %s, auto extract now.'
                  % extract_hy_filename)
            filenames = list_filenames(os.path.join(self.data_dir, 'train2014')) + \
                        list_filenames(os.path.join(self.data_dir, 'val2014')) + \
                        list_filenames(os.path.join(self.data_dir, 'test2015'))

            class ImgData(data.Dataset):
                def __getitem__(self, index):
                    visual = Image.open(filenames[index]).convert('RGB')
                    return default_transform(size)(visual)

                def __len__(self):
                    return len(filenames)

            data_loader = DataLoader(ImgData(), batch_size=80, shuffle=False, num_workers=4,
                                     pin_memory=True)
            model = pmodels.image_feature_factory({'arch': arch}, cuda=True, data_parallel=True)
            model.eval()

            output_test = model(
                Variable(torch.ones(1, 3, size, size), volatile=True))
            output_size = (len(data_loader.dataset), output_test.size(1), output_test.size(2), output_test.size(3))
            with h5py.File(extract_hy_filename, 'w') as hy:
                att = hy.create_dataset('att', output_size, dtype='f')
                for i, e in enumerate(tqdm(data_loader)):
                    input_var = Variable(e, volatile=True)
                    output_att = model(input_var)
                    batch_size = output_att.size(0)
                    att[i * batch_size: (i + 1) * batch_size] = output_att.data.cpu().numpy()
                    torch.cuda.synchronize()
            data2file(filenames, extract_txt_filename, override=override)

        txt_data = file2data(extract_txt_filename)
        feature = file2data(extract_hy_filename, type='hy')['att']

        # add postion and box feature
        pos_feature, box_feature = [], []
        if arch in ['rcnn'] and self.use_pos:
            h5_filename = 'size,{}_arch,{}.h5'.format(arch, size)
            extract_h5_filename = os.path.join(self.process_dir, h5_filename)
            boxes_info = file2data(extract_h5_filename, type='h5')
            box_feature = [e['boxes'] for e in boxes_info]
            if not os.path.exists(extract_pos_filename):
                # process the box feature

                if pos_version == 'wh':
                    print("<dataset.py> processing wh boxes")
                    output_size = (len(boxes_info), 36, 36 * 4)
                    with h5py.File(extract_pos_filename, 'w') as hy:
                        pos = hy.create_dataset('pos', output_size, dtype='f')
                        for boxes_item in tqdm(boxes_info):
                            boxes = boxes_item['boxes']  # 36 * 4
                            boxes_tmp = []  # 36 * 4 (x_mid, y_mid, w, h)
                            for box in boxes:
                                w = box[2] - box[0]
                                h = box[3] - box[1]
                                boxes_tmp.append((box[0] + w / 2, box[1] + h / 2, w, h))
                            box_pos = []
                            for i, boxi in enumerate(boxes_tmp):
                                for j, boxj in enumerate(boxes_tmp):
                                    box_pos.append([(boxi[k] - boxj[k]) / boxi[2 + k % 2] for k in range(4)])
                        pos[i] = np.array(box_pos).reshape(36, 36 * 4)
                    print("<dataset.py> processing wh boxes done")

                elif pos_version == 'area':
                    print("<dataset.py> processing area boxes")
                    # Next line is for debug
                    # boxes_info = boxes_info[0:10]
                    output_size = (len(boxes_info), 36, 36)

                    def calculate_overlapped_area(a, b):
                        # a, b无先后顺序之分
                        SI = max(0, min(a[2], b[2]) - max(a[0], b[0])) \
                             * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                        return SI

                    def calculate_S(a, b):
                        # a, b有先后顺序之分
                        S = 1 - calculate_overlapped_area(a, b) / ((b[3] - b[1]) * (b[2] - b[0]))
                        return S

                    with h5py.File(extract_pos_filename, 'w') as hy:
                        pos = hy.create_dataset('pos', output_size, dtype='f')
                        for idx, boxes_item in tqdm(enumerate(boxes_info)):
                            boxes = boxes_item['boxes']  # 36 * 4
                            box_pos = []
                            for i, boxi in enumerate(boxes):
                                for j, boxj in enumerate(boxes):
                                    box_pos.append(calculate_S(boxi, boxj))
                            pos[idx] = np.array(box_pos).reshape(36, 36)
                    print("<dataset.py> processing area boxes done")

                elif pos_version == 'neigh':
                    print("<dataset.py> processing neigh boxes")
                    # Next line is for debug
                    # boxes_info = boxes_info[0:10]
                    output_size = (len(boxes_info), 36, 36)

                    def calculate_overlapped_area(a, b):
                        # a, b无先后顺序之分
                        SI = max(0, min(a[2], b[2]) - max(a[0], b[0])) \
                             * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                        return SI

                    def calculate_S(a, b):
                        # a, b有先后顺序之分
                        S = 1 - calculate_overlapped_area(a, b) / ((b[3] - b[1]) * (b[2] - b[0]))
                        return S

                    with h5py.File(extract_pos_filename, 'w') as hy:
                        pos = hy.create_dataset('pos', output_size, dtype='f')
                        for idx, boxes_item in tqdm(enumerate(boxes_info)):
                            boxes = boxes_item['boxes']  # 36 * 4
                            box_pos = []
                            for i, boxi in enumerate(boxes):
                                for j, boxj in enumerate(boxes):
                                    box_pos.append(calculate_S(boxi, boxj))
                            pos[idx] = np.array(box_pos).reshape(36, 36)
                    print("<dataset.py> processing neigh boxes done")

                else:
                    raise ValueError('Not Supported Yet.')

            pos_feature = file2data(extract_pos_filename, type='h5')['pos']

        if load_mem == 'MEM':
            target = []
            for i, img_filename in enumerate(tqdm(txt_data)):
                for split in load_splits:
                    if split in img_filename:
                        target.append(feature[i])
                if len(target) == i:
                    target.append(None)
            feature = target
        elif load_mem == 'DB':
            class Redis():
                def __init__(self, key_prefix, shape=None, dtype=np.float32, host="localhost", port=6379, db=0):
                    self.key_prefix = key_prefix
                    self.dtype = dtype
                    self.shape = shape

                    p = subprocess.Popen("pgrep redis-server", shell=True, stdout=subprocess.PIPE)
                    out = p.stdout.read()
                    if out == b'':
                        os.system("nohup redis-server ./redis/redis.conf > ./redis/redis.out 2> ./redis/redis.err &")
                        time.sleep(3)
                    self.db = redis.StrictRedis(host=host, port=port, db=db)

                def set(self, k, v, override=False):
                    k = key_prefix + str(k)
                    v_str = v.tostring()
                    if override or not self.db.exists(k):
                        self.db.set(k, v_str)

                def get(self, k):
                    v_str = self.db.get(k)
                    v = np.fromstring(v_str, dtype=self.dtype)
                    if self.shape is not None:
                        v = v.reshape(self.shape)
                    return v

                def exists(self, k):
                    return self.db.exists(k)

                def __getitem__(self, index):
                    k = self.key_prefix + str(index)
                    return self.get(k)

            key_prefix = 'arch={},size={},idx='.format(arch, size)
            feature_shape = feature[0].shape
            special_k = key_prefix + str(0)
            end_k = key_prefix + str(len(txt_data) - 1)
            target = Redis(key_prefix, shape=feature_shape, dtype=np.float32)

            if not target.exists(special_k) or (self.vgenome and not target.exists(end_k)):
                for i, img_filename in enumerate(tqdm(txt_data)):
                    for split in load_splits:
                        if split in img_filename:
                            target.set(i, feature[i])
            else:
                print("Already loading {}_{} feature into memory".format(arch, size))
            feature = target
        elif load_mem is None:
            pass
        else:
            raise ValueError('Only support DB, MEM, None.')

        idx_to_name = {i: e for i, e in enumerate(txt_data)}
        name_to_idx = {v: k for k, v in idx_to_name.items()}

        arch_img = 'img' if arch in ['rcnn', 'rcnnsort'] else 'resnet_img'
        data_new = {arch_img:
                        {'feature': feature,
                         'pos_feature': pos_feature,
                         'box_feature': box_feature,
                         'idx_to_name': idx_to_name,
                         'name_to_idx': name_to_idx}}

        self.data.update(data_new)

    def process_qa(self, nans=2000, splitnum=2, mwc=0, mql=26, override=False):
        '''
        注意：本函数将覆盖self.data['raw'], self.data['q_vocab'], self.data['a_vocab']
        :param override:
        :param nans:
        :param splitnum:
        :param mwc:
        :param mql:
        :return:
        '''

        if self.version1_multiple_choices:
            # force to override while evaluation multiple choice answer
            override = True
        if splitnum not in [2, 3]:
            raise ValueError('split_num can only be 2 or 3')
        if self.vgenome:
            filename = os.path.join(self.process_dir, 'vgenome_version_%s,nans_%s,splitnum_%s,mwc_%s,mql_%s.h5' %
                                    (self.version, nans, splitnum, mwc, mql))
        elif self.tdiuc:
            filename = os.path.join(self.process_dir, 'tdiuc_version_%s,nans_%s,splitnum_%s,mwc_%s,mql_%s.h5' %
                                    (self.version, nans, splitnum, mwc, mql))
        elif self.clevr:
            filename = os.path.join(self.process_dir, 'clevr_version_%s,nans_%s,splitnum_%s,mwc_%s,mql_%s.h5' %
                                    (self.version, nans, splitnum, mwc, mql))
        else:
            filename = os.path.join(self.process_dir, 'version_%s,nans_%s,splitnum_%s,mwc_%s,mql_%s.h5' %
                                    (self.version, nans, splitnum, mwc, mql))

        if not os.path.exists(filename) or override:
            raw = copy.deepcopy(self.data['raw'])

            if splitnum == 2:
                splits = [raw['train'], raw['val']]
            elif splitnum == 3:
                splits = [raw['train'] + raw['val'], raw['test'],
                          raw['test_dev']]
            else:
                raise ValueError

            # 通过trainset构建答案词典
            a_vocab = Vocabulary([e['a_word'] for e in splits[0]], vocabulary_size=nans)

            # 依据答案词典筛选trainset
            splits[0] = [e for e in splits[0] if e['a_word'] in a_vocab._vocabulary_wordlist]

            # 编码trainset答案词典

            for e in splits[0]:
                e['a_idx'] = a_vocab.word2idx(e['a_word'])
                e['a_10_idx'] = []
                for e_word, e_count in e['a_10']:
                    try:
                        a_vocab.word2idx(e_word)
                    except ValueError:
                        continue
                    e['a_10_idx'].append([a_vocab.word2idx(e_word), e_count])
                count_sum = sum([v for k, v in e['a_10_idx']])
                e['a_10_idx'] = [(k, v / count_sum) for k, v in e['a_10_idx']]

            # # 特别地，编码valset答案词典
            # if split_num == 2:
            #     for e in splits[1]:
            #         e['a_idx'] = a_vocab.word2idx(e['a_word'])


            # 一　首先，处理直接编码问题
            # 对所有set的问题进行分词
            for split in splits:
                for e in tqdm(split):
                    e['q_words'] = preprocess_text2(e['q'])
                    if self.version1_multiple_choices:
                        e['a_mc_idx'] = []
                        for w in e['a_mc']:
                            try:
                                a_vocab.word2idx(w)
                            except ValueError:
                                continue
                            e['a_mc_idx'].append(a_vocab.word2idx(w))

            # 通过trainset构建问题词典
            q_vocab = Vocabulary([e1 for e0 in splits[0] for e1 in e0['q_words']],
                                 special_wordlist=['PAD', 'UNK'],
                                 min_word_count=mwc, name='q_vocab')

            # 依据问题词典编码allset的问题
            for s in splits:
                for e in s:
                    e['q_idxes'] = expand_list(q_vocab.wordlist2idxlist(e['q_words']), mql,
                                               direction='left')
                    e['q_len'] = min(mql, len(e['q_words']))

            # 二　然后，处理编码parser
            # 对所有set的问题进行分词
            """

            for splitset in splits:
                for e in tqdm(splitset):
                    e['q_five_words'] = e['q_five'].split('\t')

            # 通过trainset构建问题词典
            q_five_vocab = Vocabulary([e1 for e0 in splits[0] for e1 in e0['q_five_words']],
                                      special_wordlist=['PAD', 'UNK'],
                                      min_word_count=mwc, name='q_five_vocab')

            # 依据问题词典编码allset的问题
            for s in splits:
                for e in s:
                    e['q_five_idxes'] = expand_list(q_five_vocab.wordlist2idxlist(e['q_five_words']), 50,
                                                    direction='left')
                    e['q_five_len'] = min(mql, len(e['q_five_words']))
            """

            if splitnum == 2:
                data_new = {'qa':
                                {'train': splits[0],
                                 'val': splits[1]},
                            'q_vocab': q_vocab,
                            #  'q_five_vocab': q_five_vocab,
                            'a_vocab': a_vocab}

            elif splitnum == 3:
                data_new = {'qa':
                                {'trainval': splits[0],
                                 'test': splits[1],
                                 'test_dev': splits[2]},
                            'q_vocab': q_vocab,
                            #  'q_five_vocab': q_five_vocab,
                            'a_vocab': a_vocab}
            else:
                raise ValueError
            data2file(data_new, filename, override=override)

        else:
            data_new = file2data(filename)
        self.data.update(data_new)

    def process_five(self, override=False):
        return
        # This function will add q_five to self.data['raw']
        if self.vgenome:
            filename = os.path.join(self.process_dir, 'vgenome_version_%s_five.json' % self.version)
        else:
            filename = os.path.join(self.process_dir, 'version_%s_five.json' % self.version)
        if not os.path.exists(filename) or override:
            if platform.node() == 'DELL':
                sdp = StanfordDependencyParser(
                    path_to_jar='/home/liuqiang/tools/stanford-parser-full-2016-10-31/stanford-parser.jar',
                    path_to_models_jar='/home/liuqiang/tools/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar')
            elif platform.node() == 'chenfei-PC':
                sdp = StanfordDependencyParser(
                    path_to_jar='/root/data/stanford/stanford-parser-full-2016-10-31/stanford-parser.jar',
                    path_to_models_jar='/root/data/stanford/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar')
            else:
                raise ValueError('Not supported yet, your platform is %s' % platform.node())

            final = dict()
            for split, split_data in self.data['raw'].items():
                # split_data = split_data[0:4]
                sentences = [[e['q'].lower().replace('?', '')] for e in split_data]
                fives = [list(list(e)[0].triples()) for e in list(sdp.parse_sents(sentences, verbose=True))]
                fives_str = ['\t'.join(['\t'.join(['\t'.join(e1[0]), e1[1], '\t'.join(e1[2])]) for e1 in e0]) for e0 in
                             fives]
                tmp = []
                for i, e in enumerate(split_data):
                    tmp.append({
                        'q_id': e['q_id'],
                        'q_five': fives_str[i]
                    })
                final[split] = tmp
            data2file(final, filename, override=override)
        else:
            final = file2data(filename)
        for split, split_data in final.items():
            for i, e in enumerate(split_data):
                self.data['raw'][split][i]['q_five'] = e['q_five']

    def process_topic(self, min_word_count=7, n_topics=20, n_iter=10000, override=False):
        return
        if self.vgenome:
            filename = os.path.join(self.process_dir, 'vgenome_version_%d,mwc_%d,topic_%d,iter_%d.pkl' %
                                    (self.version, min_word_count, n_topics, n_iter))
        else:
            filename = os.path.join(self.process_dir, 'version_%d,mwc_%d,topic_%d,iter_%d.pkl' %
                                    (self.version, min_word_count, n_topics, n_iter))

        if not os.path.exists(filename) or override:
            questions = [preprocess_text2(e['q']) for e in self.data['raw']['train'] + \
                         self.data['raw']['val'] + \
                         self.data['raw']['test']]
            ids = [e['q_id'] for e in self.data['raw']['train'] + \
                   self.data['raw']['val'] + \
                   self.data['raw']['test']]

            question_words = flattenlist([e for e in questions])
            vocab = Vocabulary(question_words, special_wordlist=['UNK'], min_word_count=min_word_count)
            x = np.array([vocab.wordlist2sparselist(e) for e in questions])[:, 1:]
            model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1)
            model.fit(x)
            doc_topic = {ids[i]: e for i, e in enumerate(model.doc_topic_)}
            data2file(doc_topic, filename)
        else:
            doc_topic = file2data(filename)
        self.data['doc_topic'] = doc_topic

    def add_qt(self):
        # 处理问题类别
        t1_to_t2 = {'action': ['what sport is'],
                    'cause': ['why', 'why is the', 'how'],
                    'color': ['what color is the',
                              'what is the color of the',
                              'what color',
                              'what color is',
                              'what color are the'],
                    'count': ['how many people are',
                              'how many',
                              'how many people are in',
                              'what number is'],
                    'object': ['who is',
                               'what animal is',
                               'what brand',
                               'what room is',
                               'what is',
                               'what is the woman',
                               'where is the',
                               'what is the person',
                               'what is on the',
                               'what is in the',
                               'what is the',
                               'what is this',
                               'what are',
                               'what are the',
                               'what',
                               'what is the man',
                               'what does the'],
                    'other': ['none of the above'],
                    'position': ['where are the'],
                    'time': ['what is the name', 'what time'],
                    'type': ['what type of', 'what kind of', 'which'],
                    'yesno': ['can you',
                              'could',
                              'is it',
                              'does the',
                              'are these',
                              'are',
                              'are the',
                              'is that a',
                              'is this an',
                              'is the person',
                              'is the woman',
                              'is this person',
                              'is he',
                              'do you',
                              'has',
                              'is the man',
                              'is',
                              'is there',
                              'is there a',
                              'is this',
                              'is this a',
                              'was',
                              'are there',
                              'are they',
                              'is the',
                              'do',
                              'does this',
                              'are there any']}
        t2_to_t1 = {}
        for k, v in t1_to_t2.items():
            for e in v:
                t2_to_t1[e] = k
        self.data['t1_to_t2'] = t1_to_t2
        self.data['t2_to_t1'] = t2_to_t1

        for split in self.data['qa'].keys():
            for e in tqdm(self.data['qa'][split]):
                if e.get('q_type'):
                    e['q_type_1'] = self.data['t2_to_t1'][e['q_type']]
                else:
                    e['q_type_1'], e['q_type'] = self.get_qtype(e['q'])
        print('<datasets.py> VQA.add_qt: Added question types.')

    def get_qtype(self, question):
        t2 = [e for e in sorted(self.data['t2_to_t1'].keys(), reverse=True) if
              re.match("%s(?![a-zA-Z'-])" % e, re.sub('\s+', ' ', question).lower().strip())]
        if t2:
            t2 = t2[0]
        else:
            t2 = 'none of the above'
        t1 = self.data['t2_to_t1'][t2]
        return t1, t2

    def get_qt_acc_and_num(self, acc):
        acc_group = groupby([(k, v) for k, v in acc['perQuestionType_2'].items()],
                            lambda x: self.data['t2_to_t1'][x[0]])
        acc_type_1 = {k: [np.divide(*np.sum([e[1] for e in v], axis=0)), int(np.sum([e[1][1] for e in v], axis=0))] for
                      k, v
                      in acc_group.items()}
        total = [acc['overall'] / 100, sum([v[1] for k, v in acc['perQuestionType_2'].items()])]
        final = {'class': acc_type_1, 'total': total}

        return final

    def generate_question_txt(self, base_dir):
        keys = self.data['qa'].keys()

        filenames = [os.path.join(base_dir, "splitnum_%s" % len(keys), "%s.txt" % e) for e in keys]
        for key, filename in zip(keys, filenames):
            data2file([e['q'] for e in self.data['qa'][key]], filename=filename)

    def data_loader(self, split, target_list=None, batch_size=10, num_workers=0, shuffle=False, reverse=False,
                    debug=False):

        if split not in self.data['qa'].keys():
            raise ValueError('You need to specify self.split in %s, but got %s' %
                             (self.data['qa'].keys(), split))

        outer = self

        if debug:
            outer.data['qa'][split] = outer.data['qa'][split][0:10]

        class Inner(data.Dataset):
            def __getitem__(self, index):
                item = {}
                item_vqa = outer.data['qa'][split][index]

                # 图片
                if 'v' in target_list:
                    visual_index = outer.data['img']['name_to_idx'][item_vqa['img_filename']]
                    item['v'] = torch.Tensor(outer.data['img']['feature'][visual_index])

                if 'v_resnet' in target_list:
                    visual_index = outer.data['resnet_img']['name_to_idx'][item_vqa['img_filename']]
                    item['v_resnet'] = torch.Tensor(outer.data['resnet_img']['feature'][visual_index])

                if 'v_box' in target_list:
                    visual_index = outer.data['img']['name_to_idx'][item_vqa['img_filename']]
                    item['v_box'] = torch.Tensor(outer.data['img']['box_feature'][visual_index])

                if 'v_pos' in target_list:
                    visual_index = outer.data['img']['name_to_idx'][item_vqa['img_filename']]
                    item['v_pos'] = torch.Tensor(outer.data['img']['pos_feature'][visual_index])

                # 样本id
                if 'q_id' in target_list:
                    item['q_id'] = item_vqa['q_id']

                # 句子idx
                if 'q_idxes' in target_list:
                    if reverse:
                        item_vqa['q_idxes'][:item_vqa['q_len']] = item_vqa['q_idxes'][:item_vqa['q_len']][::-1]
                    item['q_idxes'] = torch.LongTensor(item_vqa['q_idxes'])

                # 句子parser_idx
                if 'q_five_idxes' in target_list:
                    item['q_five_idxes'] = torch.LongTensor(item_vqa['q_five_idxes'])

                # 句子topic
                if 'q_t' in target_list:
                    item['q_t'] = torch.FloatTensor(outer.data['doc_topic'][item['q_id']])

                # We need multiple_choices answers in test(_dev)
                if outer.version1_multiple_choices and split in ['test_dev', 'test']:
                    # PADDING to same length
                    item['a_mc_idx'] = torch.LongTensor(item_vqa['a_mc_idx'] + [-1] * (50 - len(item_vqa['a_mc_idx'])))

                # 对于train, trainval,还需要返回答案
                if split in ['trainval', 'train']:
                    if outer.samplingans:
                        assert item_vqa['a_10_idx']
                        choice_id, choice_prob = tuple(zip(*item_vqa['a_10_idx']))
                        # try:
                        #     choice_id, choice_prob = tuple(zip(*item_vqa['a_10_idx']))
                        # except:
                        #     raise ValueError(item_vqa['a_10_idx'])
                        item['a'] = int(np.random.choice(choice_id, p=choice_prob))
                    else:
                        # import ipdb; ipdb.set_trace()
                        assert item_vqa['a_10_idx']
                        item['a'] = torch.FloatTensor(len(outer.data['a_vocab'])).zero_()
                        for c_id, c_prob in item_vqa['a_10_idx']:
                            # TODO
                            # SigmoidLoss use c_prob
                            # SigmoidLossOne use 1
                            # item['a'][c_id] = 1  # TODO: replace c_prob with prob 1
                            item['a'][c_id] = c_prob  # TODO: replace c_prob with prob 1
                return item

            def __len__(self):
                return len(outer.data['qa'][split])

        return DataLoader(Inner(),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    if platform.node() in ['chenfei-PC', 'Moymix-PC']:
        data_dir = '/root/data/VQA/download'
        process_dir = '/root/data/VQA/preprocess'
        log_dir = '/root/data/VQA/logs'
        analyze_dir = '/root/data/VQA/analyze'
    elif platform.node() == 'chenfei':
        data_dir = 'E:/data/VQA/download'
        process_dir = 'E:/data/VQA/preprocess'
        log_dir = 'E:/data/VQA/logs'
        analyze_dir = 'E:/data/VQA/analyze'
    else:
        data_dir = '/mnt/cephfs/lab/liujinlai.licio/data/VQA/download'
        process_dir = '/mnt/cephfs/lab/liujinlai.licio/data/VQA/preprocess'
        log_dir = '/mnt/cephfs/lab/liujinlai.licio/data/VQA/logs'
        analyze_dir = '/mnt/cephfs/lab/liujinlai.licio/data/VQA/analyze'

    vqa = VQA(data_dir=data_dir, process_dir=process_dir, version=2, samplingans=True, use_pos=False, tdiuc=False)
    # vqa = VQA(data_dir=data_dir, process_dir=process_dir, version=2, samplingans=False, use_pos=False, clevr=True)
    vqa.process_five(override=False)
    # vqa.process_topic(min_word_count=7, n_topics=20, n_iter=10, override=False)
    vqa.process_img(arch='rcnn', size=224)
    # vqa.process_img(arch='rcnn', size=224)
    # vqa.process_img(arch='fbresnet152', size=224)

    vqa.process_qa(nans=3000, splitnum=2, mwc=0, mql=26, override=False)
    # vqa.add_qt()
    # vqa.get_qt_acc_and_num('/root/data/VQA/logs/StackAtt_VAL/epoch_40/acc.json')

    trainloader = vqa.data_loader(split='train', target_list=['q_id', 'q_idxes'], batch_size=2, num_workers=0,
                                  shuffle=False, reverse=False)
    # trainloader = vqa.data_loader(split='train', target_list=['q_id', 'q_idxes'], batch_size=2, num_workers=0,
    #                               shuffle=False, reverse=True)

    tmp = trainloader.__iter__().__next__()
    print(tmp)

    print(vqa.data['qa']['val'][0])

    # valloader = vqa.data_loader(split='val', batch_size=20, num_workers=0, shuffle=False)
    # testloader = vqa.data_loader(split='test', batch_size=20, num_workers=0, shuffle=False)
    # testdevloader = vqa.data_loader(split='test_dev', batch_size=20, num_workers=0, shuffle=False)

    # for i, e in tqdm(enumerate(trainloader)):
    #     pass
    # valloader = vqa.data_loader(split='val', batch_size=20)


    print('Done!')
