from putils import *
import platform

'''
This is the current best model.
'''

# Preprocess
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
    data_dir = 'data/VQA/download'
    process_dir = 'data/VQA/preprocess'
    log_dir = 'data/VQA/logs'
    analyze_dir = 'data/VQA/analyze'

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
nans = 3000
splitnum = 2

mwc = 0
mql = 26

# Train

# target_list = ['v', 'q_id', 'q_idxes', 'q_five_idxes', 'q_t']
target_list = ['v', 'q_id', 'q_idxes']
epochs = 100

resume = True
print_freq = 10
lr = 0.0001
load_mem = 'DB'
# load_mem = 'MEM'
# load_mem = None
batch_size = 256
clip_grad = True
# if test_dev is None, skip
# test_dev_range = range(51, epochs + 1, 1)
test_dev_range = range(epochs, epochs + 5, 5)

# if test_range is None, skip test.
# test_range = range(60, 61, 1)
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
        self.compress_q = MyLinear(2400, 310, p=0.5, af='relu')

        # self.cat_fusion = ThreeMutanFusion(310, 310, 310, 510, 5)

        # 获取图像最终特征
        # self.fusion_vq = MutanFusion(310, 310, 510, 5)
        self.att = MyATT(fuse_dim=11160, glimpses=4, inputs_dim=2048, att_dim=620, af='relu')
        # self.att = MyATT(fuse_dim=510, glimpses=4, inputs_dim=2048, att_dim=620, af='relu')
        # 获取文本最终特征
        self.linear_q = MyLinear(2400, 310, p=0.5, af='relu')
        # 融合图文最终特征
        self.fusion_final = MutanFusion(620, 310, 510, 5)
        self.linear_classif = MyLinear(510, self.num_classes, p=0.5)

    def forward(self, sample):
        # 初始输入
        v_feature = sample['v'].contiguous().view(-1, 36, 2048)  # b*36*2048
        q_feature = self.seq2vec(sample['q_idxes'])  # b*26*2400　

        # v_feature = F.normalize(v_feature, p=2, dim=-1)

        # 获得batch_size
        b = sample['v'].size(0)

        # 压缩图像特征到底维
        v_feature_low = self.compress_v(v_feature)  # b*36*310

        # 压缩文本特征到底维　并　拓展区块
        q_feature_low = self.compress_q(q_feature)  # b*310

        fusion = []
        vs = torch.split(v_feature_low, 1, dim=1)
        for vi in vs:
            for vj in vs:
                tmp = (vi.squeeze() - vj.squeeze()) * q_feature_low
                fusion.append(tmp)
        vq = torch.stack(fusion, dim=0).transpose(0, 1).contiguous().view(b, 36, 36 * 310)
        #
        # 融合图像特征和文本特征，并通过attention得到最终的图像特征
        # vq = self.fusion_vq(v_feature_low, q_feature_low.view(b, 1, 310).repeat(1, 36, 1))  # b*36*510
        v_final, alphas = self.att(v_feature, vq)  # b*620

        self.alpha_dict = {
            'alphas': alphas[0]
        }
        # 通过全连接得到文本最终特征

        q_final = self.linear_q(q_feature)  # b*310

        # 融合两个最终特征
        x = self.fusion_final(v_final, q_final)

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
