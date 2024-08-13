import argparse
import time
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import torchvision
from dataset import CV_Dataset
import numpy as np
from distributed_utils import accuracy, AverageMeter, cleanup
from torch.utils.tensorboard import SummaryWriter


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    init_seeds(local_rank + 1)  # set different seed for each worker
    init_method = 'tcp://' + args.ip + ':' + args.port
    cudnn.benchmark = True
    dist.init_process_group(
        backend='nccl', init_method=init_method, world_size=args.nprocs, rank=local_rank
    )

    if args.model == 'resnet18':
        model = torchvision.models.resnet18(num_classes=2)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(num_classes=2)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=2)
    elif args.model == 'resnet101':
        model = torchvision.models.resnet101(num_classes=2)
    elif args.model == 'resnet152':
        model = torchvision.models.resnet152(num_classes=2)
    else:
        raise ValueError('model type error')

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    ) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    # train_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    batch_size = int(args.batch)

    train_data_info = np.load(
        f'{args.data_path}/data_info/train_{args.img_type}_{args.start_time}_{args.end_time}_{args.lag_order}_{args.horizon}.npy'
    )
    train_data_info = train_data_info[:1000]
    train_dataset = CV_Dataset(
        f'{args.img_path}/{args.img_type}',
        args.start_time,
        args.end_time,
        args.lag_order,
        args.horizon,
        train_data_info,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.worker,
        pin_memory=True,
        sampler=train_sampler,
    )

    valid_data_info = np.load(
        f'{args.data_path}/data_info/valid_{args.img_type}_{args.start_time}_{args.end_time}_{args.lag_order}_{args.horizon}.npy'
    )
    valid_data_info = valid_data_info[:1000]
    valid_dataset = CV_Dataset(
        f'{args.img_path}/{args.img_type}',
        args.start_time,
        args.end_time,
        args.lag_order,
        args.horizon,
        valid_data_info,
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=args.worker,
        pin_memory=True,
        sampler=valid_sampler,
    )

    if args.local_rank == 0:
        model_log_dir = f'{args.log_dir}/{args.img_type}/{args.model}/I{args.lag_order}R{args.horizon}/random_{args.seed}_{args.lr}_{args.batch}_{args.start_time}_{args.end_time}'
        if not os.path.exists(model_log_dir):
            os.makedirs(model_log_dir)
        writer = SummaryWriter(log_dir=model_log_dir)

    best_score = 1e20
    early_stopping_count = torch.zeros(1).cuda()

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        for step, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.squeeze(-1).cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc1 = accuracy(outputs, labels, topk=(1,))[0]

            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # train_scheduler.step()

        finish = time.time()

        valid_loss, valid_acc = validate(
            valid_loader, model, criterion, local_rank, args
        )

        if args.local_rank == 0:
            writer.add_scalar('train/loss', losses.avg, epoch)
            writer.add_scalar('train/accuracy', top1.avg, epoch)
            writer.add_scalar('valid/loss', valid_loss, epoch)
            writer.add_scalar('valid/accuracy', valid_acc, epoch)
            torch.save(
                model.state_dict(), os.path.join(model_log_dir, 'network_final.pth')
            )

            print(
                f'epoch {epoch}, time consumed {finish - start:.2f}, Train Acc@1 {top1.avg:.3f}, Train Loss {losses.avg:.4f}, Valid Acc@1 {valid_acc:.3f}, Valid Loss {valid_loss:.4f}'
            )

            if valid_loss < best_score:
                print('update model')
                best_score = valid_loss
                torch.save(
                    model.state_dict(), os.path.join(model_log_dir, 'network_best.pth')
                )
                early_stopping_count -= early_stopping_count
            else:
                early_stopping_count += 1

        if reduce_sum(early_stopping_count) >= args.patience:
            break
        
    cleanup()



def validate(valid_loader, model, criterion, local_rank, args):
    model.eval()

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    with torch.no_grad():
        for i, (images, target) in enumerate(valid_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.squeeze(-1).cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))

    return losses.avg, top1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Multi-GPUs Training')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Patience para')
    parser.add_argument('--model', type=str, default='resnet18', help='Model type')

    parser.add_argument(
        '--img-path',
        type=str,
        default='Your/Image/Path/Here',
        help='Path of saved images',
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='Your/Data/Path/Here',
        help='Path of saved data',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='Your/Model/Checkpoint/Path/Here',
        help='Save path of trained models',
    )
    
    parser.add_argument('--img-type', type=str, default='ohlc', help='Type of images')
    parser.add_argument(
        '--start-time',
        type=str,
        default='2014-01-01',
        help='Train+valid dataset start time',
    )
    parser.add_argument(
        '--end-time',
        type=str,
        default='2021-01-01',
        help='Train+valid dataset end time',
    )
    parser.add_argument('--lag-order', type=int, default=20, help='Lag order of the images')
    parser.add_argument('--horizon', type=int, default=5, help='Forecasting horizon')
    parser.add_argument(
        '--worker',
        type=int,
        default=10,
        help='Number of processes used for loading data',
    )
    parser.add_argument(
        '--local_rank', default=-1, type=int, help='Node rank for distributed training'
    )
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--ip', default='127.0.0.1', type=str)
    parser.add_argument('--port', default='23456', type=str)

    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
