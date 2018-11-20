import gc
import lda
import torch.utils.data as data
from nltk.parse.stanford import StanfordDependencyParser
from putils import *
import redis
import subprocess


class COCOQA(object):
    def __init__(self, data_dir, process_dir, samplingans=True, vgenome=False):
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.split = None
        self.vgenome = vgenome
        self.samplingans = samplingans
        self.data = dict()

        self.preprocess()

    def preprocess(self, override=False):
        filename = os.path.join(self.process_dir, 'cocoqa.json')
        if not os.path.exists(filename) or override:
            def get_vqa_split(split, flag):
                data_split = []
                with open(os.path.join(self.data_dir, 'cocoqa_{0}/questions.txt'.format(split))) as f_q, \
                        open(os.path.join(self.data_dir, 'cocoqa_{0}/answers.txt'.format(split))) as f_a, \
                        open(os.path.join(self.data_dir, 'cocoqa_{0}/img_ids.txt'.format(split))) as f_img, \
                        open(os.path.join(self.data_dir, 'cocoqa_{0}/types.txt'.format(split))) as f_type:
                    content_q = [e for e in f_q.read().split('\n') if e]
                    content_a = [e for e in f_a.read().split('\n') if e]
                    content_img = [e for e in f_img.read().split('\n') if e]
                    content_type = [e for e in f_type.read().split('\n') if e]
                    assert len(content_q) == len(content_a) == len(content_img) == len(content_type)
                    length = len(content_q)
                    for i in range(length):

                        img_filename = os.path.join("/root/data/VQA/download", '%s/COCO_%s_%012d.jpg'
                                                    % (flag, flag, int(content_img[i])))
                        if os.path.exists(img_filename) or os.path.exists(img_filename.replace('/root', '.')):
                            data_split.append({
                                "q_id": i + 1,
                                "q": content_q[i],
                                "a_word": content_a[i],
                                "img_filename": img_filename,
                                'a_10': sorted(
                                    collections.Counter([content_a[i]]).items(),
                                    key=lambda x: (-x[1], x[0])),
                                'type': content_type[i]
                            })
                        else:
                            raise ValueError('{} does not exist in {}.'.format(img_filename, split))
                return data_split

            self.data['raw'] = {'train': get_vqa_split('train', 'train2014'),
                                'test': get_vqa_split('test', 'val2014')}
            data2file(self.data['raw'], filename, override=override)
        else:
            self.data['raw'] = file2data(filename)

    def process_img(self, arch='fbresnet152', size=224, override=False, load_mem=None, load_splits=None,
                    pos_version=None):
        '''

        :param arch:
        :param size:
        :param override:
        :param load_mem_split: should be a list. for example, ['train', 'val']
        :return:
        '''

        if pos_version:
            print('<cocoqa.process_img>: warning: pos_version is not valid for cocoqa, not implemented.')

        hy_filename = 'size,{}_arch,{}.hy'.format(arch, size)
        txt_filename = 'size,{}_arch,{}.txt'.format(arch, size)

        extract_hy_filename = os.path.join(self.process_dir, hy_filename)
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
        if load_mem == 'MEM':
            target = []
            print(txt_data)
            for i, img_filename in enumerate(tqdm(txt_data)):
                for split in load_splits:
                    if split in img_filename:
                        target.append(feature[i])
                if len(target) == i:
                    target.append(None)
            feature = target
            print(len(feature))
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
                        if split in img_filename or (split == 'train' and 'vgenome' in img_filename):
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

        arch_img = 'img' if arch == 'rcnn' else 'resnet_img'
        data_new = {arch_img:
                        {'feature': feature,
                         'idx_to_name': idx_to_name,
                         'name_to_idx': name_to_idx}}
        self.data.update(data_new)

    def process_qa(self, nans=430, splitnum=2, mwc=0, mql=26, override=False):
        '''
        注意：本函数将覆盖self.data['raw'], self.data['q_vocab'], self.data['a_vocab']
        :param override:
        :param nans:
        :param splitnum:
        :param mwc:
        :param mql:
        :return:
        '''
        # TODO important!!!! add it in server!!!
        # override=True

        if splitnum not in [2, 3]:
            raise ValueError('split_num can only be 2 or 3')
        filename = os.path.join(self.process_dir, 'cocoqa,nans_%s,splitnum_%s,mwc_%s,mql_%s.h5' %
                                (nans, splitnum, mwc, mql))

        if not os.path.exists(filename) or override:
            raw = copy.deepcopy(self.data['raw'])

            splits = [raw['train'], raw['test']]

            # 通过trainset构建答案词典
            a_vocab = Vocabulary([e['a_word'] for e in splits[0]], vocabulary_size=nans)

            # 依据答案词典筛选trainset
            splits[0] = [e for e in splits[0] if e['a_word'] in a_vocab._vocabulary_wordlist]

            # 编码trainset答案词典

            for split in splits:
                for e in split:
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
            for splitset in splits:
                for e in tqdm(splitset):
                    e['q_words'] = preprocess_text2(e['q'])

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
            data_new = {'qa': {'train': splits[0],
                               'test': splits[1]},
                        'q_vocab': q_vocab,
                        'a_vocab': a_vocab}
            data2file(data_new, filename, override=override)

        else:
            data_new = file2data(filename)
        self.data.update(data_new)

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

    # TODO
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

                # 对于train, trainval,还需要返回答案

                if 'a' in target_list:
                    if outer.samplingans:
                        item['a'] = item_vqa['a_idx']
                    else:
                        item['a'] = torch.FloatTensor(len(outer.data['a_vocab'])).zero_()
                        item['a'][item_vqa['a_idx']] = 1

                return item

            def __len__(self):
                return len(outer.data['qa'][split])

        return DataLoader(Inner(),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    if platform.node() == 'chenfei-PC':
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
        data_dir = 'data/VQA/download'
        process_dir = 'data/VQA/preprocess'
        log_dir = 'data/VQA/logs'
        analyze_dir = 'data/VQA/analyze'

    vqa = COCOQA(data_dir=data_dir, process_dir=process_dir, samplingans=True)
    # vqa.process_img(arch='fbresnet152', size=224)
    vqa.process_qa(nans=430, splitnum=2, mwc=0, mql=26, override=False)
    # vqa.add_qt()
    # vqa.get_qt_acc_and_num('/root/data/VQA/logs/StackAtt_VAL/epoch_40/acc.json')

    trainloader = vqa.data_loader(split='train', target_list=['q_id', 'q_idxes'], batch_size=2, num_workers=0,
                                  shuffle=False, reverse=True)
    tmp = trainloader.__iter__().__next__()
    print(tmp)

    print(vqa.data['qa']['train'][0])

    # valloader = vqa.data_loader(split='val', batch_size=20, num_workers=0, shuffle=False)
    # testloader = vqa.data_loader(split='test', batch_size=20, num_workers=0, shuffle=False)
    # testdevloader = vqa.data_loader(split='test_dev', batch_size=20, num_workers=0, shuffle=False)

    # for i, e in tqdm(enumerate(trainloader)):
    #     pass
    # valloader = vqa.data_loader(split='val', batch_size=20)


    print('Done!')
