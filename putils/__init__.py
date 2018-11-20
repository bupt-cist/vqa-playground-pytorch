from utils import *
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import pmodels
# import spacy

print('[info] putils: Enhanced Deep Learning Putils loaded successfully, enjoy it!')


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        return self.linear(x)

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('[error] putils.Conv2d(%s, %s, %s, %s): input_dim (%s) should equal to 4' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))

        # x: b*7*7*512
        x = x.transpose(2, 3).transpose(1, 2)  # b*512*7*7
        x = self.conv(x)  # b*450*7*7
        x = x.transpose(1, 2).transpose(2, 3)  # b*7*7*450
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('[error] putils.Conv1d(%s, %s, %s, %s): input_dim (%s) should equal to 3' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))
        # x: b*49*512
        x = x.transpose(1, 2)  # b*512*49
        x = self.conv(x)  # b*450*49
        x = x.transpose(1, 2)  # b*49*450
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


def bmatmul(inputs1, inputs2):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(torch.matmul(inputs1[i], inputs2[i]))
    outputs = torch.stack(m, dim=0)
    return outputs


def bmul(inputs1, inputs2):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(inputs1[i] * inputs2[i])
    outputs = torch.stack(m, dim=0)
    return outputs


def bmul3(inputs1, inputs2, inputs3):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(inputs1[i] * inputs2[i] * inputs3[i])
    outputs = torch.stack(m, dim=0)
    return outputs


def badd(inputs, inputs2):
    b = inputs.size()[0]
    m = []
    for i in range(b):
        m.append(inputs[i] + inputs2[i])
    outputs = torch.stack(m, dim=0)
    return outputs


class RelationFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):
        '''
        Note: can only fusion two inputs with batch.

        inputs1: b*..*input_dim1
        inputs2: b*...*input_dim2
        outputs: b*...*hidden_dim

        调整为最长，最后元素调整为hidden_dim

        :param input_dim1:
        :param input_dim2:
        :param hidden_dim:
        :param R: Do element-wise product R times.
        :param seed: random seed.
        '''
        super(RelationFusion, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear_hv = nn.ModuleList(
            [nn.Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear_hq = nn.ModuleList(
            [nn.Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        x_mm = []
        for i in range(self.R):
            h1 = self.list_linear_hv[i](inputs1)
            h2 = self.list_linear_hq[i](inputs2)
            x_mm.append(torch.bmm(h1, h2.transpose(1, 2)))
        x_mm = torch.stack(x_mm, dim=1)
        return x_mm

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusionOld(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):
        '''
        Note: can only fusion two inputs with batch.

        inputs1: b*..*input_dim1
        inputs2: b*...*input_dim2
        outputs: b*...*hidden_dim

        调整为最长，最后元素调整为hidden_dim

        :param input_dim1:
        :param input_dim2:
        :param hidden_dim:
        :param R: Do element-wise product R times.
        :param seed: random seed.
        '''
        super(MutanFusionOld, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear_hv = nn.ModuleList(
            [nn.Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear_hq = nn.ModuleList(
            [nn.Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        x_mm = []
        for i in range(self.R):
            h1 = self.list_linear_hv[i](inputs1)
            h2 = self.list_linear_hq[i](inputs2)
            x_mm.append(bmul(h1, h2))
        x_mm = torch.stack(x_mm, dim=1).sum(1)
        return x_mm

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):
        '''
        Note: can only fusion two inputs with batch.

        inputs1: b*..*input_dim1
        inputs2: b*...*input_dim2
        outputs: b*...*hidden_dim

        调整为最长，最后元素调整为hidden_dim

        :param input_dim1:
        :param input_dim2:
        :param hidden_dim:
        :param R: Do element-wise product R times.
        :param seed: random seed.
        '''
        super(MutanFusion, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        total = 0
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1)
            h2 = self.list_linear2[i](inputs2)
            total += bmul(h1, h2)
        return total

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusion2D(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):
        super(MutanFusion2D, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        total = 0
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1)
            h2 = self.list_linear2[i](inputs2)
            h1 = h1.view(-1, 1, h1.size(1), h1.size(2))
            h2 = h2.view(-1, h2.size(1), 1, h2.size(2)).repeat(1, 1, h1.size(1), 1)
            total += bmul(h1, h2)
        return total

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class ATTOld(nn.Module):
    def __init__(self, fuse_dim, glimpses, inputs_dim, att_dim, af='tanh'):
        super(ATTOld, self).__init__()
        assert att_dim % glimpses == 0
        self.glimpses = glimpses
        self.inputs_dim = inputs_dim
        self.att_dim = att_dim
        self.conv_att = Conv1d(fuse_dim, glimpses, 1, 1)  # (510, 2, 1, 1)
        self.list_linear_v_fusion = nn.ModuleList(
            [nn.Linear(inputs_dim, int(att_dim / glimpses)) for _ in range(glimpses)])  # (2048, 620/n) * n
        self.af = af

    def forward(self, inputs, fuse):
        b = inputs.size(0)
        n = inputs.size(1)
        x_att = F.dropout(self.conv_att(fuse), p=0.5, training=self.training)  # b*49*2
        list_att_split = torch.split(x_att, 1, dim=2)  # (b*49*1, b*49*1)
        list_att = []  # [b*49, b*49]
        for x_att in list_att_split:
            x_att = F.softmax(x_att.squeeze(-1))  # b*49
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = inputs  # b*49*2048

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(b, n, 1).expand(b, n, self.inputs_dim)
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(b, self.inputs_dim)
            list_v_att.append(x_v_att)

        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att, p=0.5, training=self.training)
            x_v = getattr(F, self.af)(self.list_linear_v_fusion[glimpse_id](x_v))
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)
        return x_v, list_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusionFE(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):

        super(MutanFusionFE, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        total = 0
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1)
            h2 = self.list_linear2[i](inputs2)
            total += h1*h2
        return total

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusionFNE(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):

        super(MutanFusionFNE, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        total = 0
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1)
            h2 = self.list_linear2[i](inputs2)
            total += (h1 * h2.view(-1, 1, self.hidden_dim))
        return total

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)

class ATT(nn.Module):
    def __init__(self, fuse_dim, glimpses, inputs_dim, att_dim, seed=None, af='tanh'):
        super(ATT, self).__init__()
        assert att_dim % glimpses == 0
        self.glimpses = glimpses
        self.inputs_dim = inputs_dim
        self.att_dim = att_dim
        self.conv_att = Conv1d(fuse_dim, glimpses, 1, 1, seed=seed)  # (510, 2, 1, 1)
        self.list_linear_v_fusion = nn.ModuleList(
            [nn.Linear(inputs_dim, int(att_dim / glimpses)) for _ in range(glimpses)])  # (2048, 620/n) * n
        self.af = af

    def forward(self, inputs, fuse):
        b = inputs.size(0)
        n = inputs.size(1)
        x_att = F.dropout(self.conv_att(fuse), p=0.5, training=self.training)  # b*49*2
        list_att_split = torch.split(x_att, 1, dim=2)  # (b*49*1, b*49*1)
        list_att = []  # [b*49, b*49]
        for x_att in list_att_split:
            x_att = F.softmax(x_att.squeeze(-1), dim=-1)  # b*49
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = inputs  # b*49*2048

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(b, n, 1).expand(b, n, self.inputs_dim)
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(b, self.inputs_dim)
            list_v_att.append(x_v_att)

        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att, p=0.5, training=self.training)
            x_v = getattr(F, self.af)(self.list_linear_v_fusion[glimpse_id](x_v))
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)
        return x_v, list_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class ATTN(nn.Module):
    def __init__(self, inputs_dim, guidance_dim, fuse_dim, att_dim, glimpses, R, seed=None, af='tanh'):
        super(ATTN, self).__init__()
        assert att_dim % glimpses == 0
        self.inputs_dim = inputs_dim
        self.guidance_dim = guidance_dim
        self.fuse_dim = fuse_dim
        self.att_dim = att_dim
        self.glimpses = glimpses
        self.R = R
        self.af = af

        self.fuse = MutanFusion(inputs_dim, guidance_dim, fuse_dim, R, seed=seed)
        self.conv_att = Conv1d(fuse_dim, glimpses, 1, 1, seed=seed)  # (510, 2, 1, 1)
        self.list_linear_v_fusion = nn.ModuleList(
            [Linear(inputs_dim, int(att_dim / glimpses), seed=seed) for _ in range(glimpses)])  # (2048, 620/n) * n

    def forward(self, inputs, guidance):
        if inputs.dim() != 3 or guidance.dim() != 2:
            raise ValueError('[error] putils.ATTN: inputs dim should be 3, guidance dim should be 2')
        fuse = self.fuse(inputs, guidance)

        b = inputs.size(0)
        n = inputs.size(1)
        x_att = F.dropout(self.conv_att(fuse), p=0.5, training=self.training)  # b*49*2
        list_att_split = torch.split(x_att, 1, dim=2)  # (b*49*1, b*49*1)
        list_att = []  # [b*49, b*49]
        for x_att in list_att_split:
            x_att = F.softmax(x_att.squeeze(-1))  # b*49
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = inputs  # b*49*2048

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(b, n, 1).expand(b, n, self.inputs_dim)
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(b, self.inputs_dim)
            list_v_att.append(x_v_att)

        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att, p=0.5, training=self.training)
            x_v = getattr(F, self.af)(self.list_linear_v_fusion[glimpse_id](x_v))
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        if self.glimpses == 1:
            list_att = list_att[0]

        return x_v, list_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class EmbeddingDropout():
    def __init__(self, p=0.5):
        super(EmbeddingDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.training = True

    def forward(self, input):
        # input must be tensor
        if self.p > 0 and self.training:
            dim = input.dim()
            if dim == 1:
                input = input.view(1, -1)
            batch_size = input.size(0)
            for i in range(batch_size):
                x = np.unique(input[i].numpy())
                x = np.nonzero(x)[0]
                x = torch.from_numpy(x)
                noise = x.new().resize_as_(x)
                noise.bernoulli_(self.p)
                x = x.mul(noise)
                for value in x:
                    if value > 0:
                        mask = input[i].eq(value)
                        input[i].masked_fill_(mask, 0)
            if dim == 1:
                input = input.view(-1)

        return input


class SequentialDropout(nn.Module):
    def __init__(self, p=0.5):
        super(SequentialDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.restart = True

    def _make_noise(self, input):
        return Variable(input.data.new().resize_as_(input.data))

    def forward(self, input):
        if self.p > 0 and self.training:
            if self.restart:
                self.noise = self._make_noise(input)
                self.noise.data.bernoulli_(1 - self.p).div_(1 - self.p)
                if self.p == 1:
                    self.noise.data.fill_(0)
                self.noise = self.noise.expand_as(input)
                self.restart = False
            return input.mul(self.noise)

        return input

    def end_of_sequence(self):
        self.restart = True

    def backward(self, grad_output):
        self.end_of_sequence()
        if self.p > 0 and self.training:
            return grad_output.mul(self.noise)
        else:
            return grad_output

    def __repr__(self):
        return type(self).__name__ + '({:.4f})'.format(self.p)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0.0, bidirectional=False, return_last=True):
        super(GRU, self).__init__()
        self.batch_first = batch_first
        self.return_last = return_last
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias, batch_first=batch_first,
                          dropout=dropout, bidirectional=bidirectional)

    def forward(self, emb, lengths=None):
        if self.return_last:
            lengths, idx = torch.sort(lengths, dim=-1, descending=True)
            packed = pack_padded_sequence(emb[idx, :], list(lengths), batch_first=self.batch_first)
            out_packed, last = self.gru(packed)
            final = last[0][idx, :]
            return final

        else:
            o, h = self.gru(emb)
            return o


class AbstractGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False):
        super(AbstractGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

        # Modules
        self.weight_ir = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_ii = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_in = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_hr = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hi = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hn = nn.Linear(hidden_size, hidden_size, bias=bias_hh)

    def forward(self, x, hx=None):
        raise NotImplementedError


class GRUCell(AbstractGRUCell):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False, af='tanh'):
        super(GRUCell, self).__init__(input_size, hidden_size,
                                      bias_ih, bias_hh)
        self.af = af

    def forward(self, x, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        r = F.sigmoid(self.weight_ir(x) + self.weight_hr(hx))
        i = F.sigmoid(self.weight_ii(x) + self.weight_hi(hx))
        n = getattr(F, self.af)(self.weight_in(x) + r * self.weight_hn(hx))
        hx = (1 - i) * n + i * hx
        return hx


class BayesianGRUCell(AbstractGRUCell):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False,
                 dropout=0.25, af='tanh'):
        super(BayesianGRUCell, self).__init__(input_size, hidden_size,
                                              bias_ih, bias_hh)
        self.set_dropout(dropout)
        self.af = af

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.drop_ir = SequentialDropout(p=dropout)
        self.drop_ii = SequentialDropout(p=dropout)
        self.drop_in = SequentialDropout(p=dropout)
        self.drop_hr = SequentialDropout(p=dropout)
        self.drop_hi = SequentialDropout(p=dropout)
        self.drop_hn = SequentialDropout(p=dropout)

    def end_of_sequence(self):
        self.drop_ir.end_of_sequence()
        self.drop_ii.end_of_sequence()
        self.drop_in.end_of_sequence()
        self.drop_hr.end_of_sequence()
        self.drop_hi.end_of_sequence()
        self.drop_hn.end_of_sequence()

    def forward(self, x, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        x_ir = self.drop_ir(x)
        x_ii = self.drop_ii(x)
        x_in = self.drop_in(x)
        x_hr = self.drop_hr(hx)
        x_hi = self.drop_hi(hx)
        x_hn = self.drop_hn(hx)
        r = F.sigmoid(self.weight_ir(x_ir) + self.weight_hr(x_hr))
        i = F.sigmoid(self.weight_ii(x_ii) + self.weight_hi(x_hi))
        n = getattr(F, self.af)(self.weight_in(x_in) + r * self.weight_hn(x_hn))
        hx = (1 - i) * n + i * hx
        return hx


class AbstractGRU(nn.Module):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False):
        super(AbstractGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self._load_gru_cell()

    def _load_gru_cell(self):
        raise NotImplementedError

    def forward(self, x, hx=None, max_length=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(x[:, i, :], hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        output = torch.cat(output, 1)
        return output, hx


class BayesianGRU(AbstractGRU):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False,
                 dropout=0.25, return_last=True, af='tanh'):
        self.dropout = dropout
        self.return_last = return_last
        self.af = af
        super(BayesianGRU, self).__init__(input_size, hidden_size,
                                          bias_ih, bias_hh)

    def _load_gru_cell(self):
        self.gru_cell = BayesianGRUCell(self.input_size, self.hidden_size,
                                        self.bias_ih, self.bias_hh,
                                        dropout=self.dropout, af=self.af)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.gru_cell.set_dropout(dropout)

    def forward(self, x, lengths=None):
        hx = None
        batch_size = x.size(0)
        seq_length = x.size(1)
        max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(x[:, i, :], hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        self.gru_cell.end_of_sequence()
        output = torch.cat(output, 1)
        # TODO

        if self.return_last == 'all':
            self.all_hiddens = output.clone()
            x = output

            batch_size = x.size(0)

            mask = x.data.new().resize_as_(x.data).fill_(0)
            for i in range(batch_size):
                mask[i][lengths[i] - 1].fill_(1)
            mask = Variable(mask)
            x = x.mul(mask)
            x = x.sum(1).view(batch_size, 2400)
            return output, x
        if self.return_last == 'l':
            self.all_hiddens = output.clone()
            x = output

            batch_size = x.size(0)

            mask = x.data.new().resize_as_(x.data).fill_(0)
            for i in range(batch_size):
                mask[i][lengths[i] - 1].fill_(1)
            mask = Variable(mask)
            x = x.mul(mask)
            x = x.sum(1).view(batch_size, 2400)
            return output, x, lengths

        if self.return_last:
            self.all_hiddens = output.clone()
            x = output

            batch_size = x.size(0)

            mask = x.data.new().resize_as_(x.data).fill_(0)
            for i in range(batch_size):
                mask[i][lengths[i] - 1].fill_(1)
            mask = Variable(mask)
            x = x.mul(mask)
            x = x.sum(1).view(batch_size, 2400)
            return x

        else:
            return output


class RelationFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R):
        super(RelationFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear_hv = nn.ModuleList(
            [nn.Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear_hq = nn.ModuleList(
            [nn.Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, input_v, input_q):
        x_mm = []
        for i in range(self.R):
            x_hv = self.list_linear_hv[i](input_v)
            x_hq = self.list_linear_hq[i](input_q)
            x_mm.append(torch.bmm(x_hv, x_hq.transpose(1, 2)))
        x_mm = torch.stack(x_mm, dim=1)
        return x_mm

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class SeqConv(nn.Module):
    def __init__(self, vocab_list, data_dir='/root/data/VQA/download'):
        super(SeqConv, self).__init__()
        self.vocab_list = vocab_list
        self.data_dir = data_dir
        self.data = None
        self.download()

        self.embedding = nn.Embedding(num_embeddings=len(self.vocab_list),
                                      embedding_dim=620,
                                      padding_idx=0)
        self.load_emb_state_dict()

        self.seq1 = Conv1d(620, 1200, 3, 1, 1)
        self.seq2 = Conv1d(1200, 2400, 3, 1, 1)
        self.seq3 = Conv1d(2400, 2400, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out = F.dropout(self.seq1(emb), p=0.5, training=self.training)  # b*49*2
        out = F.dropout(self.seq2(out), p=0.5, training=self.training)  # b*49*2
        out = F.dropout(self.seq3(out), p=0.5, training=self.training)  # b*49*2
        return out

    def download(self):
        url_to_targets = {'http://www.cs.toronto.edu/~rkiros/models/dictionary.txt': ['dictionary.txt'],
                          'http://www.cs.toronto.edu/~rkiros/models/utable.npy': ['utable.npy'],
                          'http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz': ['uni_skip.npz']}

        for url, targets in url_to_targets.items():
            for target in targets:
                if not os.path.exists(os.path.join(self.data_dir, target)):
                    filename = download_file(url, self.data_dir)
                    extract_file(filename)
        print('[info] putils.SkipThoughts: All file prepared.')

    def load_emb_state_dict(self):
        # 获取skip-thoughts词典
        words = file2data(os.path.join(self.data_dir, 'dictionary.txt'))
        # 获取skip-thoughts预训练的表示
        represents = file2data(os.path.join(self.data_dir, 'utable.npy'))
        # 生成单词－－表示字典
        word_to_represent = {e: represents[i] for i, e in enumerate(words)}
        # 将PAD设定为[0, 0,... 0]　长度620
        # TODO
        # word_to_represent['PAD'] = np.zeros(620)
        # 计算小范围词典的表示
        vocab_represents = []
        # 统计vocab中没有在skip-thoughts词典中的单词
        count = 0
        for i, e in enumerate(self.vocab_list):
            if e in word_to_represent:
                tmp = word_to_represent[e]
            else:
                tmp = word_to_represent['UNK']
                count += 1
            if tmp.ndim == 2:
                tmp = tmp[0]
            vocab_represents.append(tmp)
        print('[info] putils.SkipThoughts: %s/%s words not in skip-thoughts representation.'
              % (count, len(self.vocab_list)))
        vocab_represents = torch.from_numpy(np.array(vocab_represents))

        self.embedding.load_state_dict({'weight': vocab_represents})


def get_emb(vocab_list, data_dir='/root/data/VQA/download'):
    url_to_targets = {'http://www.cs.toronto.edu/~rkiros/models/dictionary.txt': ['dictionary.txt'],
                      'http://www.cs.toronto.edu/~rkiros/models/utable.npy': ['utable.npy'],
                      'http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz': ['uni_skip.npz']}

    for url, targets in url_to_targets.items():
        for target in targets:
            if not os.path.exists(os.path.join(data_dir, target)):
                download_file(url, data_dir)
    print('[info] putils.SkipThoughts: All file prepared.')

    # 获取skip-thoughts词典
    words = file2data(os.path.join(data_dir, 'dictionary.txt'))
    # 获取skip-thoughts预训练的表示
    represents = file2data(os.path.join(data_dir, 'utable.npy'))
    # 生成单词－－表示字典
    word_to_represent = {e: represents[i] for i, e in enumerate(words)}
    # 将PAD设定为[0, 0,... 0]　长度620
    # TODO
    # word_to_represent['PAD'] = np.zeros(620)
    # 计算小范围词典的表示
    vocab_represents = []
    # 统计vocab中没有在skip-thoughts词典中的单词
    count = 0
    for i, e in enumerate(vocab_list):
        if e in word_to_represent:
            tmp = word_to_represent[e]
        else:
            tmp = word_to_represent['UNK']
            count += 1
        if tmp.ndim == 2:
            tmp = tmp[0]
        vocab_represents.append(tmp)
    print('[info] putils.SkipThoughts: %s/%s words not in skip-thoughts representation.'
          % (count, len(vocab_list)))
    vocab_represents = torch.from_numpy(np.array(vocab_represents))

    return vocab_represents


class SkipThoughts(nn.Module):
    def __init__(self, vocab_list, data_dir='/root/data/VQA/download', gru='BayesianGRU', return_last=True, af='tanh'):
        super(SkipThoughts, self).__init__()
        self.vocab_list = vocab_list
        self.data_dir = data_dir
        self.af = af
        self.data = None
        self.download()

        self.embedding = nn.Embedding(num_embeddings=len(self.vocab_list),
                                      embedding_dim=620,
                                      padding_idx=0)
        self.load_emb_state_dict()

        if gru == 'GRU':
            self.gru = GRU(input_size=620, hidden_size=2400, batch_first=True, dropout=0.25, return_last=return_last)
            self.load_gru_state_dict()
        elif gru == 'BayesianGRU':
            self.gru = BayesianGRU(input_size=620, hidden_size=2400, dropout=0.25,
                                   return_last=return_last, af=self.af)
            self.load_bayesiangru_state_dict()
        else:
            raise ValueError

    def download(self):
        url_to_targets = {'http://www.cs.toronto.edu/~rkiros/models/dictionary.txt': ['dictionary.txt'],
                          'http://www.cs.toronto.edu/~rkiros/models/utable.npy': ['utable.npy'],
                          'http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz': ['uni_skip.npz']}

        for url, targets in url_to_targets.items():
            for target in targets:
                if not os.path.exists(os.path.join(self.data_dir, target)):
                    filename = download_file(url, self.data_dir)
        print('[info] putils.SkipThoughts: All file prepared.')

    def load_emb_state_dict(self):
        # 获取skip-thoughts词典
        words = file2data(os.path.join(self.data_dir, 'dictionary.txt'))
        # 获取skip-thoughts预训练的表示
        represents = file2data(os.path.join(self.data_dir, 'utable.npy'))
        # 生成单词－－表示字典
        word_to_represent = {e: represents[i] for i, e in enumerate(words)}
        # 将PAD设定为[0, 0,... 0]　长度620
        # TODO
        # word_to_represent['PAD'] = np.zeros(620)
        # 计算小范围词典的表示
        vocab_represents = []
        # 统计vocab中没有在skip-thoughts词典中的单词
        count = 0
        for i, e in enumerate(self.vocab_list):
            if e in word_to_represent:
                tmp = word_to_represent[e]
            else:
                tmp = word_to_represent['UNK']
                count += 1
            if tmp.ndim == 2:
                tmp = tmp[0]
            vocab_represents.append(tmp)
        print('[info] putils.SkipThoughts: %s/%s words not in skip-thoughts representation.'
              % (count, len(self.vocab_list)))
        vocab_represents = torch.from_numpy(np.array(vocab_represents))

        self.embedding.load_state_dict({'weight': vocab_represents})

    def load_gru_state_dict(self):
        rnn_weights_file = file2data(os.path.join(self.data_dir, 'uni_skip.npz'))

        rnn_weights = collections.OrderedDict()
        rnn_weights['bias_ih_l0'] = torch.zeros(7200)
        rnn_weights['bias_hh_l0'] = torch.zeros(7200)  # must stay equal to 0
        rnn_weights['weight_ih_l0'] = torch.zeros(7200, 620)
        rnn_weights['weight_hh_l0'] = torch.zeros(7200, 2400)
        rnn_weights['weight_ih_l0'][:4800] = torch.from_numpy(rnn_weights_file['encoder_W']).t()
        rnn_weights['weight_ih_l0'][4800:] = torch.from_numpy(rnn_weights_file['encoder_Wx']).t()
        rnn_weights['bias_ih_l0'][:4800] = torch.from_numpy(rnn_weights_file['encoder_b'])
        rnn_weights['bias_ih_l0'][4800:] = torch.from_numpy(rnn_weights_file['encoder_bx'])
        rnn_weights['weight_hh_l0'][:4800] = torch.from_numpy(rnn_weights_file['encoder_U']).t()
        rnn_weights['weight_hh_l0'][4800:] = torch.from_numpy(rnn_weights_file['encoder_Ux']).t()
        self.gru.load_state_dict({'gru.%s' % k: v for k, v in rnn_weights.items()})

    def load_bayesiangru_state_dict(self):
        rnn_weights_file = file2data(os.path.join(self.data_dir, 'uni_skip.npz'))

        rnn_weights = collections.OrderedDict()
        rnn_weights['gru_cell.weight_ir.weight'] = torch.from_numpy(rnn_weights_file['encoder_W']).t()[:2400]
        rnn_weights['gru_cell.weight_ii.weight'] = torch.from_numpy(rnn_weights_file['encoder_W']).t()[2400:]
        rnn_weights['gru_cell.weight_in.weight'] = torch.from_numpy(rnn_weights_file['encoder_Wx']).t()

        rnn_weights['gru_cell.weight_ir.bias'] = torch.from_numpy(rnn_weights_file['encoder_b'])[:2400]
        rnn_weights['gru_cell.weight_ii.bias'] = torch.from_numpy(rnn_weights_file['encoder_b'])[2400:]
        rnn_weights['gru_cell.weight_in.bias'] = torch.from_numpy(rnn_weights_file['encoder_bx'])

        rnn_weights['gru_cell.weight_hr.weight'] = torch.from_numpy(rnn_weights_file['encoder_U']).t()[:2400]
        rnn_weights['gru_cell.weight_hi.weight'] = torch.from_numpy(rnn_weights_file['encoder_U']).t()[2400:]
        rnn_weights['gru_cell.weight_hn.weight'] = torch.from_numpy(rnn_weights_file['encoder_Ux']).t()
        self.gru.load_state_dict(rnn_weights)

    def forward(self, x, return_hidden=False):
        emb = self.embedding(x)
        lengths = (x.size(1) - x.data.eq(0).sum(1)).long()
        outputs = self.gru(emb, lengths)
        if return_hidden:
            return outputs, self.gru.all_hiddens
        else:
            return outputs

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class Glove(nn.Module):
    def __init__(self, vocab_list, data_dir='/root/data/VQA/download', return_last=True, af='tanh', dim=2400):
        super(Glove, self).__init__()
        self.vocab_list = vocab_list
        self.data_dir = data_dir
        self.return_last = return_last
        self.af = af
        self.data = None

        self.nlp_glove = spacy.load('en_vectors_web_lg')
        self.glove_dict = []

        self.skip_embedding = nn.Embedding(num_embeddings=len(self.vocab_list),
                                           embedding_dim=620,
                                           padding_idx=0)
        self.glove_embedding = nn.Embedding(num_embeddings=len(self.vocab_list),
                                            embedding_dim=300,
                                            padding_idx=0)

        self.lstm = nn.LSTM(input_size=620 + 300, hidden_size=dim, num_layers=1, batch_first=True)
        self.load_glove_dict()
        self.glove_embedding.weight.requires_grad = False

    def load_glove_dict(self):
        # 获取skip-thoughts词典
        words = file2data(os.path.join(self.data_dir, 'dictionary.txt'))
        # 获取skip-thoughts预训练的表示
        represents = file2data(os.path.join(self.data_dir, 'utable.npy'))
        # 生成单词－－表示字典
        word_to_represent = {e: represents[i] for i, e in enumerate(words)}
        # 将PAD设定为[0, 0,... 0]　长度620
        # TODO
        # word_to_represent['PAD'] = np.zeros(620)
        # 计算小范围词典的表示
        skip_represents = []
        glove_represents = []
        # 统计vocab中没有在skip-thoughts词典中的单词
        count = 0
        for i, e in enumerate(self.vocab_list):
            if e in word_to_represent:
                tmp = word_to_represent[e]
            else:
                tmp = word_to_represent['UNK']
                count += 1
            if tmp.ndim == 2:
                tmp = tmp[0]
            skip_represents.append(tmp)
            glove_represents.append(self.nlp_glove(u'%s' % e).vector)

        print('[info] putils.Glove: %s/%s words not in skip-thoughts representation.'
              % (count, len(self.vocab_list)))
        skip_represents = torch.from_numpy(np.array(skip_represents))
        self.skip_embedding.load_state_dict({'weight': skip_represents})
        glove_represents = torch.from_numpy(np.array(glove_represents))
        self.glove_embedding.load_state_dict({
            'weight': glove_represents})

    def forward(self, x):
        lengths = (x.size(1) - x.data.eq(0).sum(1)).long()
        skip_emb = F.tanh(self.skip_embedding(x))
        glove_emb = self.glove_embedding(x)
        emb = torch.cat([skip_emb, glove_emb], 2)
        out, _ = self.lstm(emb)
        drop_out = F.dropout(out, p=0.3)

        if self.return_last:
            self.all_hiddens = drop_out.clone()
            x = drop_out

            batch_size = x.size(0)
            mask = x.data.new().resize_as_(x.data).fill_(0)
            for i in range(batch_size):
                mask[i][lengths[i] - 1].fill_(1)
            mask = Variable(mask)
            x = x.mul(mask)
            x = x.sum(1).view(batch_size, 2400)
            return x

        else:
            return drop_out

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


def default_transform(size):
    transform = transforms.Compose([
        transforms.Scale(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # resnet imagnet
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


if __name__ == '__main__':
    default_transform(224)
