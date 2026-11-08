import argparse
import copy
import csv
import json
import os
import random
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Proper ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on the test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

# ---------------------------------------------------------------------------
# Reproducibility / fidelity options added for the MIAE01 replication.
# Every one of them defaults to the previous behaviour so that the script can
# still be run exactly as upstream intended.
# ---------------------------------------------------------------------------
parser.add_argument('--seed', default=0, type=int,
                    help='random seed for python/numpy/torch (default: 0)')
parser.add_argument('--val-split', dest='val_split', default=5000, type=int,
                    help='hold out N images of the training set for model '
                         'selection as in the paper (default: 5000; 0 = keep '
                         'the upstream behaviour of selecting on the test set)')
parser.add_argument('--lr-milestones', dest='lr_milestones', default='100,150',
                    type=str,
                    help='comma separated epoch indices at which the learning '
                         'rate is divided by 10 (default: 100,150). The paper '
                         'decays at 32k and 48k iterations, which is 82 and 123 '
                         'epochs for 391 iterations/epoch.')
parser.add_argument('--warmup-iters', dest='warmup_iters', default=0, type=int,
                    help='number of initial iterations trained at 0.1x the base '
                         'learning rate (the paper uses ~400 iterations of 0.01 '
                         'to warm up ResNet-110/1202 before switching to 0.1)')
parser.add_argument('--log-csv', dest='log_csv', default='', type=str,
                    help='append one row of metrics per epoch to this CSV file')
parser.add_argument('--run-csv', dest='run_csv', default='', type=str,
                    help='append one row of final results (test error, wall '
                         'clock, git commit) to this CSV file; '
                         'scripts/summarize.py aggregates it into results/')
parser.add_argument('--config-out', dest='config_out', default='', type=str,
                    help='dump the resolved configuration as JSON to this path')
parser.add_argument('--test-every', dest='test_every', default=0, type=int,
                    help='(debug only) evaluate the test set every N epochs; '
                         '0 disables it so that the test set stays untouched '
                         'until the final evaluation')

best_prec1 = 0


def set_seed(seed):
    """Seed every random number generator we depend on.

    ``cudnn.deterministic`` disables the fastest (but non-deterministic)
    convolution algorithms, which costs a few percent of throughput but is
    required for the mean +- std reporting of the assignment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def git_commit():
    """Short hash of the checked-out commit, recorded with every run."""
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return 'unknown'


def worker_init_fn(worker_id):
    """Give every DataLoader worker its own deterministic seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def lr_scale(it, args, iters_per_epoch, milestones):
    """Learning-rate multiplier for iteration ``it``.

    Implements both parts of the paper's schedule in one place:
    a short warm-up at 0.1x base LR, then a step decay by 10x at the
    given epoch milestones.
    """
    scale = 1.0
    if args.warmup_iters > 0 and it < args.warmup_iters:
        scale *= 0.1
    epoch_f = it / float(max(1, iters_per_epoch))
    for m in milestones:
        if epoch_f >= m:
            scale *= 0.1
    return scale


def parse_milestones(text):
    values = [float(v) for v in str(text).split(',') if v.strip() != '']
    if not values:
        raise ValueError('--lr-milestones must contain at least one epoch index')
    return sorted(values)


def load_state_dict_compat(model, state_dict):
    """Load a checkpoint saved either with or without a DataParallel wrapper.

    The released upstream checkpoints were saved from a DataParallel model, so
    every key carries a "module." prefix; a model trained here on a single GPU
    does not. Stripping the prefix lets both kinds load into the same module.
    """
    if state_dict and all(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Seed before anything touches the RNG (weight init, shuffling, augmentation).
    set_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --log-csv / --run-csv / --config-out point into logs/, which may not exist yet
    for path in (args.log_csv, args.run_csv, args.config_out):
        if path:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: no CUDA device found, training on CPU (slow)')

    model = resnet.__dict__[args.arch]()
    model = model.to(device)
    # DataParallel only adds value with more than one GPU; keeping it off for a
    # single GPU also keeps the checkpoint keys free of the "module." prefix.
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # optionally resume from a checkpoint
    checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # weights_only=False keeps compatibility with the legacy (pre 1.6)
            # pickle format of the upstream *.th files
            checkpoint = torch.load(args.resume, map_location=device,
                                    weights_only=False)
            args.start_epoch = checkpoint.get('epoch', 0)
            best_prec1 = checkpoint.get('best_prec1', 0)
            load_state_dict_compat(model, checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint.get('epoch', 0)))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    # Evaluation must not be augmented, so the validation split needs its own
    # dataset object with the evaluation transform; both objects wrap the same
    # 50k images on disk.
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_aug = datasets.CIFAR10(root='./data', train=True, download=True,
                                 transform=train_transform)
    train_eval = datasets.CIFAR10(root='./data', train=True, download=True,
                                  transform=eval_transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True,
                               transform=eval_transform)

    if args.val_split > 0:
        # Deterministic 45k/5k split, as in the paper. The permutation uses its
        # own generator so that the split does not depend on how many random
        # numbers the model/augmentation consumed before.
        split_generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(train_aug), generator=split_generator).tolist()
        val_idx = perm[:args.val_split]
        train_idx = perm[args.val_split:]
        trainset = torch.utils.data.Subset(train_aug, train_idx)
        valset = torch.utils.data.Subset(train_eval, val_idx)
        print('split: {} train / {} val (seed {})'.format(
            len(trainset), len(valset), args.seed))
    else:
        # Upstream behaviour: "validation" is the test set, which makes the
        # reported number optimistic. Kept only for backwards compatibility.
        print('WARNING: --val-split 0 selects the model on the test set; '
              'the result is not comparable to the paper')
        trainset = train_aug
        valset = torch.utils.data.Subset(testset, range(len(testset)))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=worker_init_fn, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    iters_per_epoch = len(train_loader)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    milestones = parse_milestones(args.lr_milestones)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda it: lr_scale(it, args, iters_per_epoch, milestones))

    if checkpoint is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Dump everything needed to reproduce this exact run.
    config = {
        'args': vars(args),
        'torch': torch.__version__,
        'cuda': torch.version.cuda if device.type == 'cuda' else None,
        'cudnn': cudnn.version() if device.type == 'cuda' else None,
        'device': str(device),
        'gpu': (torch.cuda.get_device_name(0) if device.type == 'cuda' else None),
        'n_train': len(trainset),
        'n_val': len(valset),
        'n_test': len(testset),
        'iters_per_epoch': iters_per_epoch,
        'milestones': milestones,
        'total_iters': iters_per_epoch * args.epochs,
    }
    print(json.dumps(config, indent=2, default=str))
    if args.config_out:
        with open(args.config_out, 'w') as fh:
            json.dump(config, fh, indent=2, default=str)

    if args.log_csv:
        new_file = not os.path.exists(args.log_csv)
        csv_fh = open(args.log_csv, 'a', newline='')
        csv_writer = csv.writer(csv_fh)
        if new_file:
            csv_writer.writerow(['epoch', 'train_loss', 'train_acc',
                                 'val_loss', 'val_acc', 'lr', 'secs'])
            csv_fh.flush()
    else:
        csv_fh, csv_writer = None, None

    if args.evaluate:
        validate(test_loader, model, criterion, device)
        return

    best_state = None
    best_epoch = -1
    run_start = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer,
                                      scheduler, epoch, device)

        # evaluate on the held-out validation split (never on the test set)
        val_loss, val_acc = validate(val_loader, model, criterion, device)

        elapsed = time.time() - start

        if csv_writer is not None:
            csv_writer.writerow([epoch + 1, '{:.4f}'.format(train_loss),
                                 '{:.3f}'.format(train_acc),
                                 '{:.4f}'.format(val_loss),
                                 '{:.3f}'.format(val_acc),
                                 '{:.6f}'.format(optimizer.param_groups[0]['lr']),
                                 '{:.1f}'.format(elapsed)])
            csv_fh.flush()

        # remember best accuracy on the validation split and save a checkpoint
        is_best = val_acc > best_prec1
        if is_best:
            best_prec1 = val_acc
            best_epoch = epoch + 1
            # keep the weights in memory so the final test evaluation does not
            # depend on which checkpoint happened to be written to disk
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone()
                 for k, v in model.state_dict().items()})

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'scheduler': scheduler.state_dict(),
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'scheduler': scheduler.state_dict(),
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

        if args.test_every > 0 and (epoch + 1) % args.test_every == 0:
            print('(debug) test set at epoch {}:'.format(epoch + 1))
            validate(test_loader, model, criterion, device)

    if csv_fh is not None:
        csv_fh.close()

    # Single, final evaluation of the model selected on the validation split.
    test_acc = None
    if args.val_split > 0 and best_state is not None:
        model.load_state_dict(best_state)
        print('best validation Prec@1 {:.3f} at epoch {}'.format(
            best_prec1, best_epoch))
        print('final evaluation on the 10k test images:')
        _, test_acc = validate(test_loader, model, criterion, device)
        print('RESULT arch={} seed={} epochs={} best_val={:.3f} '
              'test_acc={:.3f} test_err={:.3f}'.format(
                  args.arch, args.seed, args.epochs, best_prec1,
                  test_acc, 100.0 - test_acc))
    else:
        print('RESULT arch={} seed={} epochs={} best_prec1={:.3f}'.format(
            args.arch, args.seed, args.epochs, best_prec1))

    # One row per run: this is the only place where the test error is stored,
    # so that scripts/summarize.py can compute mean +- std across seeds.
    if args.run_csv:
        new_file = not os.path.exists(args.run_csv)
        with open(args.run_csv, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(['timestamp', 'arch', 'seed', 'epochs',
                                 'batch_size', 'lr_milestones', 'warmup_iters',
                                 'best_val', 'test_acc', 'test_err',
                                 'minutes', 'git_commit'])
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                args.arch, args.seed, args.epochs, args.batch_size,
                args.lr_milestones, args.warmup_iters,
                '{:.3f}'.format(best_prec1),
                '' if test_acc is None else '{:.3f}'.format(test_acc),
                '' if test_acc is None else '{:.3f}'.format(100.0 - test_acc),
                '{:.1f}'.format((time.time() - run_start) / 60.0),
                git_commit(),
            ])


def train(train_loader, model, criterion, optimizer, scheduler, epoch, device):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device, non_blocking=True)
        input_var = input.to(device, non_blocking=True)
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # the schedule is defined in iterations (64k in the paper), so the
        # scheduler is stepped once per mini-batch instead of once per epoch
        scheduler.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device, non_blocking=True)
            input_var = input.to(device, non_blocking=True)
            target_var = target.to(device, non_blocking=True)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best.th'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
