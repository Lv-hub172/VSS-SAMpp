import logging
import os
import random
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms





def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    low_res_logits = outputs
    loss_ce = ce_loss(low_res_logits, low_res_label_batch.permute(0, 2, 3, 1))
    loss_dice = dice_loss(low_res_logits, low_res_label_batch.permute(0, 2, 3, 1), softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset import dataset_reader, RandomGenerator

    if not os.path.exists('./training_log'):
        os.mkdir('./training_log')
    logging.basicConfig(filename='./training_log/' + args.output.split('/')[-1] + '_log.txt',
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    db_train = dataset_reader(
        base_dir=args.root_path, split="train", num_classes=args.num_classes,
        transform=transforms.Compose([
            RandomGenerator(output_size=[args.img_size, args.img_size],
                            low_res=[low_res, low_res])
        ])
    )
    logging.info("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    batch_size = args.batch_size * args.n_gpu

    trainloader = DataLoader(db_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss(ignore_index=-100)
    dice_loss = DiceLoss(args.num_classes + 1)

    base_lr = args.base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)


    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    def lr_lambda(current_iter):

        if args.warmup_period > 0:
            if current_iter < args.warmup_period:
                return float(current_iter) / float(args.warmup_period)


        progress = (current_iter - args.warmup_period) / float(max_iterations - args.warmup_period)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']


            image_batch = image_batch.unsqueeze(2)
            image_batch = torch.cat([image_batch] * 3, dim=2)

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(image_batch, multimask_output, args.img_size)
                    loss, loss_ce_, loss_dice_ = calc_loss(outputs, label_batch,
                                                           ce_loss, dice_loss,
                                                           args.dice_param)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce_, loss_dice_ = calc_loss(outputs, label_batch,
                                                       ce_loss, dice_loss,
                                                       args.dice_param)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            lr_current = scheduler.get_last_lr()[0]

            iter_num += 1


            writer.add_scalar('lr', lr_current, iter_num)
            writer.add_scalar('loss/total', loss.item(), iter_num)
            writer.add_scalar('loss/ce', loss_ce_.item(), iter_num)
            writer.add_scalar('loss/dice', loss_dice_.item(), iter_num)


            logging.info(
                'iter %d : loss=%f, ce=%f, dice=%f, lr=%f'
                % (iter_num, loss.item(), loss_ce_.item(), loss_dice_.item(), lr_current)
            )

        if (epoch_num + 1) % 20 == 0:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            try:
                model.save_parameters(save_path)
            except:
                model.module.save_parameters(save_path)
            logging.info("Saved model: {}".format(save_path))

    writer.close()
    return "Training Finished!"



