# This is a bug for deepdish, tables must add to prevent unknow error.
import torch
import utils
import tables
import argparse
import os
import time
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import datasets
import importlib
from official_test import test_local
from pprint import pprint
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from putils import get_emb


# torch.cuda.set_device(1)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if target.dim() == 2:  # multians option
        _, target = torch.max(target, 1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(split, loader, model, criterion, optimizer, scheduler, logger, epoch, max_step=None, print_freq=10,
          clip_grad=False):
    if max_step is None:
        max_step = np.Infinity

    # switch to train mode
    model.train()
    meters = logger.reset_meters(split)
    end = time.time()

    for i, sample in enumerate(loader):
        if i < max_step:

            sample = {k: Variable(v) for k, v in sample.items()}

            batch_size = sample['q_idxes'].size(0)
            sample['a'] = sample['a'].cuda(async=True)

            # measure data loading time
            meters['data_time'].update(time.time() - end, n=batch_size)

            # compute output
            output = model(sample)
            torch.cuda.synchronize()

            loss = criterion(output, sample['a'])
            meters['loss'].update(loss.item(), n=batch_size)

            # measure accuracy
            acc1, acc5 = accuracy(output.data, sample['a'].data, topk=(1, 5))
            meters['acc1'].update(acc1.item(), n=batch_size)
            meters['acc5'].update(acc5.item(), n=batch_size)

            # compute gradient and do SGD step
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()

            if cf.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            else:
                pass

            optimizer.step()
            torch.cuda.synchronize()

            # measure elapsed time
            meters['batch_time'].update(time.time() - end, n=batch_size)
            end = time.time()

            if i % print_freq == 0:
                print('{} Epoch: [{}][{}/{}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                      'Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})'.
                      format(split, epoch, i, len(loader),
                             batch_time=meters['batch_time'],
                             data_time=meters['data_time'], loss=meters['loss'],
                             acc1=meters['acc1'], acc5=meters['acc5']))
        else:
            break

    logger.log_meters(split, n=epoch)


def test(split, loader, model, logger, epoch, a_vocab, log_dir, max_step=None, print_freq=10, eval_metric='OpenEnded'):
    if max_step is None:
        max_step = np.Infinity

    results = []

    model.eval()
    meters = logger.reset_meters(split)

    end = time.time()
    for i, sample in enumerate(loader):
        if i < max_step:
            q_id = sample['q_id']
            if eval_metric == 'OpenEnded':
                # sample = {k: Variable(v.cuda(async=True), volatile=True) for k, v in sample.items()}
                with torch.no_grad():
                    # sample = {k: v.cuda(async=True) for k, v in sample.items()}
                    # sample = {k: v.cuda(async=True) for k, v in sample.items()}
                    sample = {k: v.to(torch.device('cuda')) for k, v in sample.items()}
            elif eval_metric == 'MultipleChoice':
                a_mc_idx = sample['a_mc_idx'] if 'a_mc_idx' in sample else None
                sample = {k: Variable(v.cuda(async=True), volatile=True) for k, v in sample.items() if k != 'a_mc_idx'}
                # sample = {k: v.cuda(async=True) for k, v in sample.items() if k != 'a_mc_idx'}
            else:
                raise ValueError("<train.py> %s is not allowed" % eval_metric)

            batch_size = sample['v'].size(0)

            # compute output
            output = model(sample)

            # compute predictions for OpenEnded accuracy
            _, pred = output.data.cpu().max(1)
            pred.squeeze_()

            # compute predictions for OpenEnded accuracy
            if eval_metric == 'OpenEnded':
                _, pred = output.data.cpu().max(1)
                pred.squeeze_()
            elif eval_metric == 'MultipleChoice':
                out = output.data.cpu()
                pred = torch.LongTensor(batch_size)
                for j in range(batch_size):
                    mc_idx = a_mc_idx[j].tolist()
                    # remove padding
                    mc_idx = [e for e in mc_idx if e != -1]
                    pred[j] = -1
                    ans_prob = 0
                    for k in range(len(out[j])):
                        if k in mc_idx:
                            if pred[j] == -1 or ans_prob < out[j][k]:
                                pred[j] = k
                                ans_prob = out[j][k]
            else:
                raise ValueError("<train.py> %s is not allowed" % eval_metric)

            for j in range(batch_size):
                item = {'question_id': q_id[j].item(),
                        'answer': a_vocab.idx2word(int(pred[j]))}
                results.append(item)

            # measure elapsed time
            meters['batch_time'].update(time.time() - end, n=batch_size)
            end = time.time()

            if i % print_freq == 0:
                print('{}: [{}/{}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.
                      format(split, i, len(loader), batch_time=meters['batch_time']))
        else:
            break

    logger.log_meters(split, n=epoch)
    if split == 'test_dev':
        split = 'test-dev'
    method_name_with_epoch = "%s%.3d" % (log_dir.split('/')[-1], epoch)

    filename = os.path.join(log_dir, 'epoch_%s' % epoch,
                            'vqa_%s_mscoco_%s2015_%s_results.json' % (eval_metric, split, method_name_with_epoch))
    utils.data2file(results, filename, override=True)
    utils.compress_file(filename, override=True)
    return filename


def test_cocoqa(split, loader, model, logger, epoch, a_vocab, log_dir, max_step=None, print_freq=10):
    if max_step is None:
        max_step = np.Infinity

    results = []

    model.eval()
    meters = logger.reset_meters(split)

    end = time.time()
    for i, sample in enumerate(loader):
        if i < max_step:
            q_id = sample['q_id']
            sample = {k: Variable(v.cuda(async=True), volatile=True) for k, v in sample.items()}
            batch_size = sample['v'].size(0)

            # compute output
            output = model(sample)

            # compute predictions for OpenEnded accuracy
            _, pred = output.data.cpu().max(1)
            pred.squeeze_()
            if sample['a'].dim() == 2:
                _, target = sample['a'].data.cpu().max(1)
            else:
                target = sample['a']

            for j in range(batch_size):
                item = {'question_id': q_id[j].item(),
                        'pred': int(pred[j]) == int(target[j]),
                        'pred_word': a_vocab.idx2word(int(pred[j])),
                        }
                results.append(item)

            # measure elapsed time
            meters['batch_time'].update(time.time() - end, n=batch_size)
            end = time.time()

            if i % print_freq == 0:
                print('{}: [{}/{}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.
                      format(split, i, len(loader), batch_time=meters['batch_time']))
        else:
            break

    logger.log_meters(split, n=epoch)

    acc = sum([e['pred'] for e in results]) / (len([e['pred'] for e in results]))
    print('<train.py> test_cocoqa Epoch %s: acc is %s' % (epoch, acc))
    results_filename = os.path.join(log_dir, 'epoch_%s' % epoch, 'results.json')
    acc_filename = os.path.join(log_dir, 'epoch_%s' % epoch, 'acc.json')

    utils.data2file(results, results_filename, override=True)
    utils.data2file(acc, acc_filename, override=True)

    return


def save_checkpoint(info, model, optim, log_dir):
    log_dir = os.path.join(log_dir, 'epoch_%d' % info['epoch'])
    ckpt_info_filename = os.path.join(log_dir, 'ckpt_info.pth.tar')
    ckpt_model_filename = os.path.join(log_dir, 'ckpt_model.pth.tar')
    ckpt_optim_filename = os.path.join(log_dir, 'ckpt_optim.pth.tar')
    logger_filename = os.path.join(log_dir, 'logger.json')
    info['exp_logger'].to_json(logger_filename)
    torch.save(info, ckpt_info_filename)
    retry = 5
    while retry > 0:
      try:
        torch.save(model, ckpt_model_filename)
        break
      except RuntimeError as e:
        print(e)
        retry -= 1
    torch.save(optim, ckpt_optim_filename)


def load_checkpoint(model, optimizer, path_ckpt):
    info_filename = os.path.join(path_ckpt, '%s_info.pth.tar' % 'ckpt')
    model_filename = os.path.join(path_ckpt, '%s_model.pth.tar' % 'ckpt')
    optim_filename = os.path.join(path_ckpt, '%s_optim.pth.tar' % 'ckpt')

    info = torch.load(info_filename)
    exp_logger = info['exp_logger']

    model_state = torch.load(model_filename)
    model.load_state_dict(model_state)
    optim_state = torch.load(optim_filename)
    optimizer.load_state_dict(optim_state)

    print("<train.py> Loaded checkpoint '{}'"
          .format(path_ckpt))
    return exp_logger

def learning_scheduler(cf):
    if cf.optim == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.lr, momentum=0.9)
    elif cf.optim == 'rms':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.lr)
    elif cf.optim == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.lr)
    else:
        raise ValueError('Optim is set %s' % cf.optim)
    if cf.lr_scheduler:
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / 50000))
    else:
        scheduler = None
    return optimizer, scheduler 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train/Evaluate models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--auto', default=False, type=bool)
    # TODO change here to choose your model.
    # parser.add_argument('--cf', default='config.ABR1', type=str)
    # parser.add_argument('--cf', default='config.ABRS1', type=str)
    # parser.add_argument('--cf', default='config.ABR', type=str)
    # parser.add_argument('--cf', default='config.ABRN', type=str)
    # parser.add_argument('--cf', default='config.ABRRCNNS2', type=str)
    # parser.add_argument('--cf', default='config.ABRRCNN', type=str)
    # parser.add_argument('--cf', default='config.IQR', type=str)
    # parser.add_argument('--cf', default='config.MutanEK', type=str)
    # parser.add_argument('--cf', default='config.MutanEKT', type=str)
    # parser.add_argument('--cf', default='config.TEST', type=str)
    parser.add_argument('--cf', default='config.MFH', type=str)

    args = parser.parse_args()

    cf = importlib.import_module(args.cf)

    try:
        cf.clip_grad = bool(cf.clip_grad)
        print('<train.py> clip_grad is set %s.' % bool(cf.clip_grad))
    except AttributeError:
        cf.clip_grad = False
        print('<train.py> clip_grad is set False.')

    try:
        cf.reverse_text = bool(cf.reverse_text)
        print('<train.py> reverse_text is set %s.' % bool(cf.reverse_text))
    except AttributeError:
        cf.reverse_text = False
        print('<train.py> reverse_text is set False.')

    try:
        cf.vgenome = bool(cf.vgenome)
        print('<train.py> vgenome is set %s.' % bool(cf.vgenome))
    except AttributeError:
        cf.vgenome = False
        print('<train.py> vgenome is set False.')

    try:
        cf.tdiuc = bool(cf.tdiuc)
        print('<train.py> tdiuc is set %s.' % bool(cf.tdiuc))
    except AttributeError:
        cf.tdiuc = False
        print('<train.py> tdiuc is set False.')

    try:
        cf.box100 = bool(cf.box100)
        print('<train.py> box100 is set %s.' % bool(cf.box100))
    except AttributeError:
        cf.box100 = False
        print('<train.py> box100 is set False.')

    try:
        cf.clevr = bool(cf.clevr)
        print('<train.py> clevr is set %s.' % bool(cf.clevr))
    except AttributeError:
        cf.clevr = False
        print('<train.py> clevr is set False.')
    if cf.clevr == True:
        cf.nans = 28

    try:
        cf.cocoqa = bool(cf.cocoqa)
        print('<train.py> cocoqa is set %s.' % bool(cf.cocoqa))
    except AttributeError:
        cf.cocoqa = False
        print('<train.py> cocoqa is set False.')

    try:
        cf.answer_embedding = bool(cf.answer_embedding)
        print('<train.py> match is set %s.' % bool(cf.answer_embedding))
    except AttributeError:
        cf.answer_embedding = False
        print('<train.py> match is set False.')

    try:
        cf.version1_multiple_choices = bool(cf.version1_multiple_choices)
        print('<train.py> version1_multiple_choices is set %s.' % bool(cf.version1_multiple_choices))
    except AttributeError:
        cf.version1_multiple_choices = False
        print('<train.py> version1_multiple_choices is set False.')

    try:
        cf.use_pos = bool(cf.use_pos)
        print('<train.py> use_pos is set %s.' % bool(cf.use_pos))
    except AttributeError:
        cf.use_pos = False
        print('<train.py> use_pos is set False.')

    try:
        cf.pos_version = str(cf.pos_version)
        print('<train.py> pos_version is set %s.' % str(cf.pos_version))
    except AttributeError:
        cf.pos_version = 'wh'
        print('<train.py> pos_version is set wh.')

    try:
        cf.sgd = bool(cf.sgd)
        raise ValueError('sgd has been deprecated. Please use optim')
    except AttributeError:
        pass

    try:
        cf.optim = str(cf.optim)
        print('<train.py> optim is set %s.' % cf.optim)
    except AttributeError:
        cf.optim = 'adam'
        print('<train.py> optim is set False.')

    try:
        cf.sgd = bool(cf.sgd)
        print('<train.py> sgd is set %s.' % bool(cf.sgd))
    except AttributeError:
        cf.sgd = False
        print('<train.py> sgd is set False.')

    try:
        cf.num_workers = int(cf.num_workers)
        print('<train.py> num_workers is set %s.' % int(cf.num_workers))
    except AttributeError:
        cf.num_workers = 2
        print('<train.py> num_workers is set False.')

    try:
        cf.lr
        print('<train.py> lr is set %s.' % float(cf.lr))
    except AttributeError:
        raise AttributeError('lr must be set manually.')

    try:
        cf.lr_scheduler = bool(cf.lr_scheduler)
        print('<train.py> lr_scheduler is set %s.' % bool(cf.lr_scheduler))
    except AttributeError:
        cf.lr_scheduler = True
        print('<train.py> lr_scheduler is set True.')

    try:
        cf.loss_metric = str(cf.loss_metric)
        print('<train.py> loss_metric is set %s.' % str(cf.loss_metric))
    except AttributeError:
        cf.loss_metric = "CE" if cf.samplingans else "KLD"
        print('<train.py> loss_metric is set {}.' % (cf.loss_metric))

    print('<train.py> Using config %s...' % cf.__file__)
    if args.auto:
        print('<train.py> Auto online judge is set True.')
    else:
        print('<train.py> Auto online judge is set False.')

    if cf.cocoqa and cf.splitnum != 2:
        raise ValueError(
            '<train.py> When use cf.cocoqa=True, cf.splitnum must = 2 because cocoqa has no full test, but got %s' % cf.splitnum)
    if cf.tdiuc and cf.splitnum != 2:
        raise ValueError(
            '<train.py> When use cf.tdiuc=True, cf.splitnum must = 2 because tdiuc has no full test, but got %s' % cf.splitnum)

    if cf.cocoqa and not 'a' in cf.target_list:
        raise ValueError('<train.py> When use cf.cocoqa=True, a must in cf.target_list')
    if cf.tdiuc and not 'a' in cf.target_list:
        raise ValueError('<train.py> When use cf.tdiuc=True, a must in cf.target_list')

    # if cf.cocoqa and not (cf.samplingans and cf.loss_metric == 'CE'):
    #     raise ValueError('<train.py> When use cf.cocoqa=True, cf.samplingans must = True and cf.loss_metric must be CE')

    if cf.cocoqa and cf.nans != 430:
        raise ValueError('<train.py> When use cf.cocoqa=True, cf.nans must = 430, while now is %s' % cf.nans)
    if cf.tdiuc and cf.nans != 1480:
        raise ValueError('<train.py> When use cf.tdiuc=True, cf.nans must = 1480, while now is %s' % cf.nans)

    if cf.version == 2 and cf.splitnum == 3 and not cf.cocoqa and not cf.tdiuc and cf.test_dev_range:
        raise ValueError(
            '<train.py> When use cf.version=2 and cf.splitnum=3, test_dev_range must be set None because VQA 2.0 does not need test_dev to submit.')

    #########################################################################################
    # Create needed datasets
    #########################################################################################

    # # for before config
    # if 'vgenome' not in dir(cf):
    #     cf.vgenome = False
    if not cf.cocoqa:
        vqa = datasets.VQA(data_dir=cf.data_dir, process_dir=cf.process_dir, version=cf.version,
                           samplingans=cf.samplingans, vgenome=cf.vgenome, tdiuc=cf.tdiuc, clevr=cf.clevr,
                           version1_multiple_choices=cf.version1_multiple_choices,
                           use_pos=cf.use_pos, box100=cf.box100)
        vqa.process_five(override=False)
        vqa.process_topic(min_word_count=7, n_topics=20, n_iter=10000, override=False)
        vqa.process_qa(nans=cf.nans, splitnum=cf.splitnum, mwc=cf.mwc, mql=cf.mql,
                       override=False)
    else:
        import cocoqa_dataset as datasets

        vqa = datasets.COCOQA(data_dir=cf.data_dir, process_dir=cf.process_dir,
                              samplingans=cf.samplingans, vgenome=cf.vgenome)
        vqa.process_qa(nans=cf.nans, splitnum=cf.splitnum, mwc=cf.mwc, mql=cf.mql,
                       override=False)

    #########################################################################################
    # Create model, criterion and optimizer
    #########################################################################################

    if 'q_five_idxes' in cf.target_list:
        model = cf.Model(vqa.data['q_vocab']._vocabulary_wordlist, vqa.data['q_five_vocab']._vocabulary_wordlist,
                         len(vqa.data['a_vocab']))
    else:
        if cf.answer_embedding:
            ans_emb = get_emb(vqa.data['a_vocab']._vocabulary_wordlist, data_dir=cf.data_dir).cuda()
            model = cf.Model(vqa.data['q_vocab']._vocabulary_wordlist, ans_emb, len(vqa.data['a_vocab']))
        else:
            model = cf.Model(vqa.data['q_vocab']._vocabulary_wordlist, len(vqa.data['a_vocab']))

    model = nn.DataParallel(model).cuda()

    if cf.samplingans:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        if cf.loss_metric == 'BCE':
            class MyLoss(nn.Module):

                def __init__(self):
                    super(MyLoss, self).__init__()
                    self.act = nn.Sigmoid()
                    self.loss = nn.BCELoss()

                def forward(self, input, target):
                    return self.loss(self.act(input), target)


            criterion = MyLoss().cuda()
        elif cf.loss_metric == "KLD":
            class MyLoss(nn.Module):

                def __init__(self):
                    super(MyLoss, self).__init__()

                    self.loss = nn.KLDivLoss(size_average=False)

                def forward(self, input, target):
                    return self.loss(F.log_softmax(input), target)
        else:
            raise ValueError("<train.py> loss")

        criterion = MyLoss().cuda()

    # if cf.sgd:
    #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.lr, momentum=0.9)
    # else:
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), cf.lr)


    optimizer, scheduler = learning_scheduler(cf)
    #########################################################################################
    # args.resume: resume from a checkpoint OR create logs directory
    #########################################################################################
    print('<train.py> Resume from a checkpoint or create logs directory.')

    utils.ensure_dirname(cf.log_dir, override=not cf.resume)
    # if cf.resume:
    #     exp_logger = load_checkpoint(model.module, optimizer, cf.resume)
    # else:
    exp_logger = utils.Experiment(os.path.basename(cf.log_dir))

    meters = {
        'loss': utils.AvgMeter(),
        'acc1': utils.AvgMeter(),
        'acc5': utils.AvgMeter(),
        'batch_time': utils.AvgMeter(),
        'data_time': utils.AvgMeter(),
        'epoch_time': utils.SumMeter()
    }

    for split in vqa.data['qa'].keys():
        exp_logger.add_meters(split, meters)
    exp_logger.info['model_params'] = utils.params_count(model)
    # print('Model has {} parameters'.format(exp_logger.info['model_params']))

    print('<train.py> Start training...')

    max_step = None

    if cf.debug:
        # max_step = 5
        print('<train.py>: You are in debugging mode...')

    auto_find = {
        'train': ['train'] + [False] * cf.epochs,
        'test_dev': ['test_dev'] + [False] * cf.epochs,
        'test': ['test'] + [False] * cf.epochs,
        'test_local': ['test_local'] + [False] * cf.epochs

    }
    target_range = {
        'train': range(1, cf.epochs + 1, 1),
        'test_dev': cf.test_dev_range,
        'test': cf.test_range,
        'test_local': range(1, cf.epochs + 1, 1)

    }

    if cf.resume:

        for i in range(1, cf.epochs + 1):
            if os.path.exists(os.path.join(cf.log_dir, 'epoch_%d' % i, 'ckpt_info.pth.tar')) and \
                    os.path.exists(os.path.join(cf.log_dir, 'epoch_%d' % i, 'ckpt_model.pth.tar')) and \
                    os.path.exists(os.path.join(cf.log_dir, 'epoch_%d' % i, 'ckpt_optim.pth.tar')) and \
                    os.path.exists(os.path.join(cf.log_dir, 'epoch_%d' % i, 'logger.json')):
                auto_find['train'][i] = True

            if os.path.exists(os.path.join(cf.log_dir, 'epoch_%d' % i,
                                           'vqa_OpenEnded_mscoco_%s2015_%s_results.json.zip'
                                                   % ('test-dev', "%s%.3d" % (cf.log_dir.split('/')[-1], i)))):
                auto_find['test_dev'][i] = True

            if os.path.exists(os.path.join(cf.log_dir, 'epoch_%d' % i,
                                           'vqa_OpenEnded_mscoco_%s2015_%s_results.json.zip'
                                                   % ('test', "%s%.3d" % (cf.log_dir.split('/')[-1], i)))):
                auto_find['test'][i] = True

            if os.path.exists(os.path.join(cf.log_dir, 'epoch_%d' % i, 'acc.json')):
                auto_find['test_local'][i] = True

        print('<train.py> Auto finding checkpoints: ')
        for i, e in enumerate(list(zip(*auto_find.values()))):
            print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % ('ep' if not i else i, e[0], e[1], e[2], e[3]))

        target_range['train'] = [i for i, e in enumerate(auto_find['train']) if not e]
        target_range['test_local'] = [i for i, e in enumerate(auto_find['test_local']) if not e]

        if cf.test_dev_range:
            target_range['test_dev'] = [i for i, e in enumerate(auto_find['test_dev']) if
                                        e is False and i in cf.test_dev_range]
        if cf.test_range:
            target_range['test'] = [i for i, e in enumerate(auto_find['test']) if e is False and i in cf.test_range]

        # # TODO
        # target_range['train'] = []
        if True in auto_find['train'] and target_range['train']:
            exp_logger = load_checkpoint(model.module, optimizer,
                                         os.path.join(cf.log_dir, 'epoch_%d' % (target_range['train'][0] - 1)))

    if cf.cocoqa:
        print('<train.py> You set splitnum == 2, So final target tasks: ')
        target_range = {k: v for k, v in target_range.items() if k in ['train', 'test_local']}

        print(target_range)

        if not cf.log_dir.endswith('_VAL') or not cf.analyze_dir.endswith('_VAL') or not cf.method_name.endswith(
                '_VAL'):
            raise ValueError(
                'cf.log_dir and cf.analyze_dir and cf.method_name must end with _VAL when splitnum is set 2. \n'
                'If you use an old config, you need to add \n'
                'if splitnum == 2:method_name += _VAL')

        if target_range['train']:
            vqa.process_img(arch=cf.arch, size=cf.size, load_mem=cf.load_mem,
                            load_splits=['train', 'val'] + ['vgenome'] if cf.vgenome else [],
                            pos_version=cf.pos_version)
            if 'v_resnet' in cf.target_list:
                vqa.process_img(arch='fbresnet152', size=cf.size, load_mem=cf.load_mem,
                                load_splits=['train', 'val'] + ['vgenome'] if cf.vgenome else [],
                                pos_version=cf.pos_version)

            train_loader = vqa.data_loader(split='train', target_list=cf.target_list, batch_size=cf.batch_size,
                                           num_workers=cf.num_workers, shuffle=True, reverse=cf.reverse_text,
                                           debug=cf.debug)
            test_loader = vqa.data_loader(split='test', target_list=cf.target_list, batch_size=cf.batch_size,
                                          num_workers=cf.num_workers, shuffle=False, reverse=cf.reverse_text,
                                          debug=cf.debug)

            union = sorted(set(target_range['train']) | set(target_range['test_local']))

            for epoch in union:
                if epoch in target_range['train']:
                    train('train', train_loader, model, criterion, optimizer, scheduler,
                          exp_logger, epoch, max_step=max_step, print_freq=cf.print_freq, clip_grad=cf.clip_grad)
                    # This may save a lot of space.
                    save_checkpoint({'epoch': epoch, 'exp_logger': exp_logger},
                                    model.module.state_dict(),
                                    optimizer.state_dict(),
                                    cf.log_dir)
                    test_cocoqa('test', test_loader, model, exp_logger, epoch, vqa.data['a_vocab'],
                                cf.log_dir, max_step=max_step, print_freq=cf.print_freq)
                elif epoch in target_range['test_local']:
                    test_cocoqa('test', test_loader, model, exp_logger, epoch, vqa.data['a_vocab'],
                                cf.log_dir, max_step=max_step, print_freq=cf.print_freq)


        else:
            print('<train.py> Skip train and test')
    elif cf.splitnum == 2:
        print('<train.py> You set splitnum == 2, So final target tasks: ')

        target_range = {k: v for k, v in target_range.items() if k in ['train', 'test_local']}
        print(target_range)
        # target_range['test_local'] = list(filter(lambda x: x > 31, target_range['test_local']))

        if not cf.log_dir.endswith('_VAL') or not cf.analyze_dir.endswith('_VAL') or not cf.method_name.endswith(
                '_VAL'):
            raise ValueError(
                'cf.log_dir and cf.analyze_dir and cf.method_name must end with _VAL when splitnum is set 2. \n'
                'If you use an old config, you need to add \n'
                'if splitnum == 2:method_name += _VAL')

        if target_range['train']:
            vqa.process_img(arch=cf.arch, size=cf.size, load_mem=cf.load_mem, load_splits=['train', 'val'],
                            pos_version=cf.pos_version)
            if 'v_resnet' in cf.target_list:
                vqa.process_img(arch='fbresnet152', size=cf.size, load_mem=cf.load_mem, load_splits=['train', 'val'],
                                pos_version=cf.pos_version)

            train_loader = vqa.data_loader(split='train', target_list=cf.target_list, batch_size=cf.batch_size,
                                           num_workers=cf.num_workers, shuffle=True, reverse=cf.reverse_text,
                                           debug=cf.debug)
            val_loader = vqa.data_loader(split='val', target_list=cf.target_list, batch_size=cf.batch_size,
                                         num_workers=cf.num_workers, shuffle=False, reverse=cf.reverse_text,
                                         debug=cf.debug)

            union = sorted(set(target_range['train']) | set(target_range['test_local']))

            for epoch in union:
                if 'restart_epoch' in cf.__dict__ and epoch == cf.restart_epoch:
                  optimizer, scheduler = learning_scheduler(cf)
                elif 'keeping_epoch' in cf.__dict__ and epoch < cf.keeping_epoch:
                  optimizer, scheduler = learning_scheduler(cf)
                print("The learning rate of epoch {} is {}".format(epoch, optimizer.param_groups[0]['lr']))
                if epoch in target_range['train']:
                    train('train', train_loader, model, criterion, optimizer, scheduler,
                          exp_logger, epoch, max_step=max_step, print_freq=cf.print_freq, clip_grad=cf.clip_grad)
                    # This may save a lot of space.
                    save_checkpoint({'epoch': epoch, 'exp_logger': exp_logger},
                                    model.module.state_dict(),
                                    optimizer.state_dict(),
                                    cf.log_dir)
                    test('val', val_loader, model, exp_logger, epoch, vqa.data['a_vocab'],
                         cf.log_dir, max_step=max_step, print_freq=cf.print_freq)
                if epoch in target_range['test_local']:
                    val_filename = os.path.join(cf.log_dir, 'epoch_%s' % epoch,
                                                'vqa_OpenEnded_mscoco_val2015_%s_results.json' % (
                                                    "%s%.3d" % (cf.log_dir.split('/')[-1], epoch)))
                    # utils.execute(
                    #     'python3 official_test.py --data_dir %s --result_file %s' % (cf.data_dir, val_filename),
                    #     wait=False)
                    if not os.path.exists(val_filename):
                        test('val', val_loader, model, exp_logger, epoch, vqa.data['a_vocab'],
                             cf.log_dir, max_step=max_step, print_freq=cf.print_freq)
                    if not cf.debug:
                        if cf.tdiuc:
                            test_local(cf.data_dir, val_filename, version='tdiuc')
                        elif cf.clevr:
                            test_local(cf.data_dir, val_filename, version='clevr')
                        else:
                            test_local(cf.data_dir, val_filename, version=cf.version)

        else:
            print('<train.py> Skip train and test')

    elif cf.splitnum == 3:
        print('<train.py> You set splitnum == 3, So final target tasks: ')
        target_range = {k: v for k, v in target_range.items() if k in ['train', 'test_dev', 'test']}
        # target_range = {'train': [], 'test_dev': [64], 'test': [64]}
        print(target_range)

        if target_range['train']:
            vqa.process_img(arch=cf.arch, size=cf.size, load_mem=cf.load_mem, load_splits=['train', 'val'],
                            pos_version=cf.pos_version)
            if 'v_resnet' in cf.target_list:
                vqa.process_img(arch='fbresnet152', size=cf.size, load_mem=cf.load_mem, load_splits=['train', 'val'],
                                pos_version=cf.pos_version)

            trainval_loader = vqa.data_loader(split='trainval', target_list=cf.target_list, batch_size=cf.batch_size,
                                              num_workers=cf.num_workers, shuffle=True, reverse=cf.reverse_text,
                                              debug=cf.debug)
            for epoch in target_range['train']:
                train('trainval', trainval_loader, model, criterion, optimizer, scheduler,
                      exp_logger, epoch, max_step=max_step, print_freq=cf.print_freq, clip_grad=cf.clip_grad)
                save_checkpoint({'epoch': epoch, 'exp_logger': exp_logger},
                                model.module.state_dict(),
                                optimizer.state_dict(),
                                cf.log_dir)
        else:
            print('<train.py> Skip train.')

        if target_range['test_dev']:
            vqa.process_img(arch=cf.arch, size=cf.size, load_mem=cf.load_mem, load_splits=['test'],
                            pos_version=cf.pos_version)
            if 'v_resnet' in cf.target_list:
                vqa.process_img(arch='fbresnet152', size=cf.size, load_mem=cf.load_mem, load_splits=['test'],
                                pos_version=cf.pos_version)

            test_dev_loader = vqa.data_loader(split='test_dev', target_list=cf.target_list, batch_size=cf.batch_size,
                                              num_workers=cf.num_workers, shuffle=False, reverse=cf.reverse_text,
                                              debug=cf.debug)

            for epoch in target_range['test_dev']:
                exp_logger = load_checkpoint(model.module, optimizer,
                                             os.path.join(cf.log_dir, 'epoch_%d' % epoch))

                test_dev_results_filename = test('test_dev', test_dev_loader, model, exp_logger, epoch,
                                                 vqa.data['a_vocab'],
                                                 cf.log_dir, max_step=max_step,
                                                 print_freq=cf.print_freq,
                                                 eval_metric='OpenEnded')
                if cf.version1_multiple_choices:
                    test_dev_mc_results_filename = test('test_dev', test_dev_loader, model, exp_logger, epoch,
                                                        vqa.data['a_vocab'],
                                                        cf.log_dir, max_step=max_step,
                                                        print_freq=cf.print_freq,
                                                        eval_metric='MultipleChoice')
            if args.auto:
                auto.main()

        else:
            print('<train.py> Skip test_dev.')

        if target_range['test']:
            vqa.process_img(arch=cf.arch, size=cf.size, load_mem=cf.load_mem, load_splits=['test'],
                            pos_version=cf.pos_version)
            if 'v_resnet' in cf.target_list:
                vqa.process_img(arch='fbresnet152', size=cf.size, load_mem=cf.load_mem, load_splits=['test'],
                                pos_version=cf.pos_version)

            test_loader = vqa.data_loader(split='test', target_list=cf.target_list, batch_size=cf.batch_size,
                                          num_workers=cf.num_workers, shuffle=False, reverse=cf.reverse_text,
                                          debug=cf.debug)

            for epoch in target_range['test']:
                exp_logger = load_checkpoint(model.module, optimizer,
                                             os.path.join(cf.log_dir, 'epoch_%d' % epoch))
                test_results_filename = test('test', test_loader, model, exp_logger, epoch, vqa.data['a_vocab'],
                                             cf.log_dir, max_step=max_step,
                                             print_freq=cf.print_freq,
                                             eval_metric='OpenEnded')
                if cf.version1_multiple_choices:
                    test_results_mc_filename = test('test', test_loader, model, exp_logger, epoch,
                                                    vqa.data['a_vocab'],
                                                    cf.log_dir, max_step=max_step,
                                                    print_freq=cf.print_freq,
                                                    eval_metric='MultipleChoice')
        else:
            print('<train.py> Skip test.')

    else:
        raise ValueError
