import time
import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
from models.CrossModal import TextTransformer

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from utils.config import *


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
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


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC(C=0.075)
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def run_net(args, config, train_writer=None, val_writer=None):

    logger = get_logger(args.log_name)
    # build dataset
    # config.dataset.train.others.whole = True
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
    if config.validate != "none":
        print_log("Load extra data to validate ...")
        (_, extra_train_dataloader), (_, extra_test_dataloader) = builder.dataset_builder(args, config.dataset.extra_train), \
                                      builder.dataset_builder(args, config.dataset.extra_val)

    # build model
    base_model = builder.model_builder(config.model)

    for p in base_model.named_parameters():
        if p[1].requires_grad is True:
            print(p[0])

    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    train_transforms = transforms.Compose([data_transforms.PointcloudRotate()])

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        idx = 0
        for taxonomy_ids, model_ids, pc, img, text, _ in train_dataloader:
            num_iter += 1
            idx += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            points = pc.cuda()

            assert points.size(1) == npoints
            points = train_transforms(points)
            img = img.cuda()
            loss = base_model(points, img, text)
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item() * 1000])
            else:
                losses.update([loss.item() * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)

        if config.validate != "none" and epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(extra_train_dataloader, extra_test_dataloader, epoch, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)

        if epoch % 25 == 0 and epoch >= 250:
            builder.save_pretrain_model(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    args, logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(extra_train_dataloader, extra_test_dataloader, epoch, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)

    if config.validate == "zeroshot":
        config = cfg_from_yaml_file('cfgs/zeroshot/modelnet40.yaml')
        config.model['transformer_config'] = config.model

        base_model = builder.model_builder(config.model)
        text_model = TextTransformer(config.model)
        base_model.load_model_from_ckpt(os.path.join(args.experiment_path, 'ckpt-last.pth'), log=False)

        if args.use_gpu:
            base_model.to(args.local_rank)
            text_model.to(args.local_rank)

        base_model.eval()
        text_model.eval()

        with open(config.dataset.train._base_.DATA_PATH + '/modelnet40_shape_names.txt', 'r') as f:
            names_list = f.read().split('\n')[:-1]

        text_feature_list = torch.zeros([len(names_list), text_model.embed_dim], dtype=torch.float).cuda()
        for i in range(len(names_list)):
            text = names_list[i]
            for j in range(60):
                text_feature = text_model(text, index=j)
                text_feature_list[i] = text_feature_list[i] + text_feature

        npoints = config.npoints

        rignt_num = 0
        sample_num = 0
        for (taxonomy_ids, model_ids, data) in itr_merge(extra_train_dataloader, extra_test_dataloader):

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
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

            features = base_model(points)
            for i in range(features.shape[0]):
                similarity = torch.nn.CosineSimilarity(dim=-1)(features[i], text_feature_list)
                predit = torch.argmax(similarity)
                if predit == label[i]:
                    rignt_num += 1
                sample_num += 1

        acc = rignt_num / sample_num * 100
        print_log('[Validation] zeroshot acc = %.4f' % acc, logger=logger)

    elif config.validate == "svm":

        config = cfg_from_yaml_file('cfgs/svm/modelnet40.yaml')
        base_model = builder.model_builder(config.model)
        base_model.load_model_from_ckpt(os.path.join(args.experiment_path, 'ckpt-last.pth'), log=False)

        if args.use_gpu:
            base_model.to(args.local_rank)
        base_model.eval()

        test_features = []
        test_label = []

        train_features = []
        train_label = []
        npoints = config.npoints
        with torch.no_grad():
            for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
                points = data[0].cuda()
                label = data[1].cuda()

                points = misc.fps(points, npoints)

                assert points.size(1) == npoints
                feature = base_model(points)
                target = label.view(-1)

                train_features.append(feature.detach())
                train_label.append(target.detach())

            for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_test_dataloader):
                points = data[0].cuda()
                label = data[1].cuda()

                points = misc.fps(points, npoints)
                assert points.size(1) == npoints
                feature = base_model(points)
                target = label.view(-1)

                test_features.append(feature.detach())
                test_label.append(target.detach())

            train_features = torch.cat(train_features, dim=0)
            train_label = torch.cat(train_label, dim=0)
            test_features = torch.cat(test_features, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                train_features = dist_utils.gather_tensor(train_features, args)
                train_label = dist_utils.gather_tensor(train_label, args)
                test_features = dist_utils.gather_tensor(test_features, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                                   test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

            print_log('[Validation] EPOCH: %d  svm acc = %.4f' % (epoch, acc * 100), logger=logger)

            if args.distributed:
                torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)
