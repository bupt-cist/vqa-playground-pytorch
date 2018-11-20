from putils import *
import platform

'''
CoR model.
'''

# The data directory
YOUR_DATA_DIR = "/mnt/cephfs/lab/liujinlai.licio"
data_dir = os.path.join(YOUR_DATA_DIR, 'data/VQA/download')
process_dir = os.path.join(YOUR_DATA_DIR, 'data/VQA/preprocess')
log_dir = os.path.join(YOUR_DATA_DIR, 'data/VQA/logs')
analyze_dir = os.path.join(YOUR_DATA_DIR, 'data/VQA/analyze')

version = 2
samplingans = False
loss_metric = "KLD"
vgenome = False
version1_multiple_choices = False
# Process_img
# arch = 'fbresnet152'
arch = "rcnn"
size = 224

# Process_qa
nans = 2000
splitnum = 2
mwc = 0
mql = 26

# Train
target_list = ['v', 'q_id', 'q_idxes']
epochs = 70
restart_epoch = None
keeping_epoch = 40

resume = True
print_freq = 10
lr = 0.0001
# load_mem = 'DB'
# load_mem = 'MEM'
load_mem = None
batch_size = 100
clip_grad = True
test_dev_range = None
test_range = None
debug = False

method_name = os.path.splitext(os.path.basename(__file__))[0]
if splitnum == 2:
    method_name += '_VAL'
log_dir = os.path.join(log_dir, method_name)
analyze_dir = os.path.join(analyze_dir, method_name)


class MyConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None, p=None, af=None,
                 dim=None):
        super(MyConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('[error] putils.Conv1d(%s, %s, %s, %s): input_dim (%s) should equal to 3' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))
        # x: b*49*512
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = x.transpose(1, 2)  # b*512*49
        x = self.conv(x)  # b*450*49
        x = x.transpose(1, 2)  # b*49*450

        if self.af:
            if self.af == 'softmax':
                x = getattr(F, self.af)(x, dim=self.dim)
            else:
                x = getattr(F, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, seed=None, p=None, af=None, dim=None):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = self.linear(x)
        if self.af:
            if self.af == 'softmax':
                x = getattr(F, self.af)(x, dim=self.dim)
            else:
                x = getattr(F, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MyATT(nn.Module):
    def __init__(self, fuse_dim, glimpses, inputs_dim, att_dim, seed=None, af='tanh'):
        super(MyATT, self).__init__()
        assert att_dim % glimpses == 0
        self.glimpses = glimpses
        self.inputs_dim = inputs_dim
        self.att_dim = att_dim
        self.conv_att = MyConv1d(fuse_dim, glimpses, 1, 1, seed=seed, p=0.5, af='softmax', dim=1)  # (510, 2, 1, 1)
        self.list_linear_v_fusion = nn.ModuleList(
            [MyLinear(inputs_dim, int(att_dim / glimpses), p=0.5, af=af) for _ in range(glimpses)])  # (2048, 620/n) * n
        self.af = af

    def forward(self, inputs, fuse):
        b = inputs.size(0)
        n = inputs.size(1)
        x_att = self.conv_att(fuse)

        tmp = bmatmul(x_att.transpose(1, 2), inputs)  # b*2*2048

        list_v_att = [e.squeeze() for e in torch.split(tmp, 1, dim=1)]  # b*2048, b*2048

        list_att = torch.split(x_att, 1, dim=2)

        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = self.list_linear_v_fusion[glimpse_id](x_v_att)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)
        return x_v, list_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class Model(nn.Module):
    def __init__(self, vocab_words=None, num_ans=None):
        super(Model, self).__init__()
        self.vocab_words = vocab_words
        self.num_classes = num_ans

        self.seq2vec = SkipThoughts(vocab_list=vocab_words, data_dir=data_dir,
                                    gru='BayesianGRU', return_last=True, af='relu')
        self.compress_v = MyConv1d(2048, 310, 1, 1, p=0.5, af='relu')
        self.compress_v2 = MyConv1d(2048, 310, 1, 1, p=0.5, af='relu')
        self.compress_q = MyLinear(2400, 310, p=0.5, af='relu')

        # 获取图像最终特征
        self.fusion_vq1 = MutanFusion(310, 310, 510, 2)
        self.att1 = MyATT(fuse_dim=510, glimpses=4, inputs_dim=2048, att_dim=620, af='relu')

        self.fusion_vq2 = MutanFusion(310, 310, 510, 2)
        self.att2 = MyATT(fuse_dim=510, glimpses=4, inputs_dim=2048, att_dim=620, af='relu')

        # 获取文本最终特征
        self.linear_q = MyLinear(2400, 310, p=0.5, af='relu')
        # 融合图文最终特征
        self.fusion_final = MutanFusion(1240, 310, 510, 2)
        self.linear_classif = MyLinear(510, self.num_classes, p=0.5)

        self.compress_q_1 = MyLinear(2400, 310, p=0.5, af='relu')
        self.expand_q_1 = MyLinear(310, 2048, p=0.5, af='sigmoid')

        self.compress_q_2 = MyLinear(2400, 310, p=0.5, af='relu')
        self.expand_q_2 = MyLinear(310, 2048, p=0.5, af='sigmoid')

    def decare_cat(self, block1, block2, guidance):
        b, m, d = block1.size()
        v_feature_1 = block1.view(-1, m, 1, d).repeat(1, 1, m, 1)  # b*m*m*d
        v_feature_2 = block2.view(-1, 1, m, d).repeat(1, m, 1, 1)  # b*m*m*d
        q_feature_1 = self.expand_q_1(self.compress_q_1(guidance))
        q_feature_2 = self.expand_q_2(self.compress_q_2(guidance))

        v_feature_low_12 = bmul(v_feature_1, q_feature_1) + bmul(v_feature_2, q_feature_2)  # b*m*m*2d
        return v_feature_low_12

    def forward(self, sample):
        # 初始输入
        v_feature = sample['v'].contiguous().view(-1, 36, 2048)  # b*36*2048

        q_feature = self.seq2vec(sample['q_idxes'])  # b*26*2400　

        # 获得batch_size
        b = sample['v'].size(0)

        # 压缩文本特征到底维
        q_feature_low = self.compress_q(q_feature)  # b*310

        v_feature_low = self.compress_v(v_feature)  # b*36*310
        v1_att, alpha1 = self.att1(v_feature, self.fusion_vq1(v_feature_low, q_feature_low))  # b*620, [b*36]
        v2_cat = self.decare_cat(v_feature, v_feature, q_feature)  # b*36*36*2048
        v2_feature = (alpha1[0].contiguous().view(b, 36, 1, 1) * v2_cat).sum(1)  # b*36*2048

        v2_feature_low = self.compress_v2(v2_feature)  # b*36*310
        v2_att, alpha2 = self.att2(v2_feature, self.fusion_vq2(v2_feature_low, q_feature_low))  # b*620, [b*36]

        self.alpha_dict = {
            'alpha1': alpha1,
            'alpha2': alpha2,
            'feature': v2_feature[:, [0, 1], :]
        }

        v_f = torch.cat([v1_att, v2_att], dim=1)  # b*1240

        # 通过全连接得到文本最终特征
        q_final = self.linear_q(q_feature)  # b*310

        # 融合两个最终特征
        x = self.fusion_final(v_f, q_final)

        # 分类
        x = self.linear_classif(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(11)

    q_vocab_list = file2data('q_vocab.h5')._vocabulary_wordlist

    model = Model(q_vocab_list, num_ans=2000)
    # model = nn.DataParallel(model).cuda()
    input_visual = Variable(torch.FloatTensor(np.random.randn(2, 36, 2048)))
    input_question = Variable(torch.LongTensor(np.random.randint(100, size=[2, 26])))
    sample = {
        'v': input_visual,
        'q_idxes': input_question
    }

    output = model(sample)
    print(output)
    print('Done!')
