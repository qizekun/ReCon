import torch
from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from models.CrossModal import TextTransformer
import numpy as np
from pointnet2_ops import pointnet2_utils


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    print_log('Start zero-shot test... ', logger=logger)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    config.model['transformer_config'] = config.model
    base_model = builder.model_builder(config.model)
    text_model = TextTransformer(config.model)

    base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)
        text_model.to(args.local_rank)

    base_model.eval()
    text_model.eval()

    if config.dataset.train._base_.NAME == 'ModelNet':
        if config.dataset.test._base_.NUM_CATEGORY == 40:
            with open(config.dataset.test._base_.DATA_PATH + '/modelnet40_shape_names.txt', 'r') as f:
                names_list = f.read().split('\n')[:-1]
        elif config.dataset.test._base_.NUM_CATEGORY == 10:
            with open(config.dataset.test._base_.DATA_PATH + '/modelnet10_shape_names.txt', 'r') as f:
                names_list = f.read().split('\n')[:-1]
        else:
            raise NotImplementedError()

    elif 'ScanObjectNN' in config.dataset.train._base_.NAME:
        names_list = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door',
                      'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toile']
    else:
        raise NotImplementedError()

    text_feature_list = torch.zeros([len(names_list), text_model.embed_dim], dtype=torch.float).cuda()
    for i in range(len(names_list)):
        text = names_list[i]
        for j in range(60):
            text_feature = text_model(text, index=j)
            text_feature_list[i] = text_feature_list[i] + text_feature

    batch_start_time = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    npoints = config.npoints

    rignt_num = 0
    sample_num = 0
    # for (taxonomy_ids, model_ids, data) in itr_merge(test_dataloader):      # for only test data
    for (taxonomy_ids, model_ids, data) in itr_merge(train_dataloader, test_dataloader):  # for train and test data
        data_time.update(time.time() - batch_start_time)

        points = data[0].cuda()
        label = data[1].cuda()

        if npoints == 1024:
            point_all = 1200
        elif npoints == 2048:
            point_all = 2400
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        if points.size(1) < point_all:
            point_all = points.size(1)

        fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(),
                                                  fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
        features = base_model(points)

        for i in range(features.shape[0]):
            similarity = torch.nn.CosineSimilarity(dim=-1)(features[i], text_feature_list)
            predit = torch.argmax(similarity)
            if predit == label[i]:
                rignt_num += 1
            sample_num += 1

        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()

    acc = rignt_num / sample_num * 100
    print_log('[TEST_ZERO-SHOT] acc = %.4f' % acc, logger=logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
