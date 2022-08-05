import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolyScheduler
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

import io
from contextlib import redirect_stdout
from torchinfo import summary
import utils.checkpoint as cu

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, help="py config file", required=False, default="configs/webface42m_bottlenet_resnet.py")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    args = parser.parse_args()
    return args

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):    

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    train_loader = get_dataloader(
        cfg.DATASET,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    IM_SHAPE = (112, 112, 3)

    f = io.StringIO()
    with redirect_stdout(f):        
        summary(backbone, input_size=(1, IM_SHAPE[2], IM_SHAPE[0], IM_SHAPE[1]))
    lines = f.getvalue()
    # print("".join(lines))

    with open( os.path.join(cfg.output, "summary.txt") ,"w") as f:
        [f.write(line) for line in lines]

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()


    # checkpoint_file = cu.save_checkpoint(cfg, backbone)

    # exit(0)

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    start_epoch = 0
    global_step = 0
   
    if cfg.resume and cfg.restore_epoch != 0:
        dict_checkpoint = os.path.join(cfg.output, f"checkpoint_epoch_{cfg.restore_epoch}_gpu_{rank}.pt")
        logging.info("Load resume checkpoints: {}".format(dict_checkpoint))
        start_epoch, global_step = cu.load_checkpoint(
            dict_checkpoint, backbone, partial_fc=module_partial_fc, optimizer=opt, exp_lr_scheduler=lr_scheduler, rank=rank)
        del dict_checkpoint

    elif cfg.resume and cu.has_checkpoint(cfg):
        last_checkpoint = cu.get_last_checkpoint(cfg)
        start_epoch, global_step = cu.load_checkpoint(
            last_checkpoint, backbone, partial_fc=module_partial_fc, optimizer=opt, exp_lr_scheduler=lr_scheduler)
        logging.info("Load path: %s" %(last_checkpoint))

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.valrec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img) # get pair embedding
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)  # input embedding & batch label

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()
            
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)
                
                    if rank == 0:
                        os.system("rm " + cfg.output + "/model_epoch_*" )
                        logging.info("save ckpt")
                        checkpoint_file = cu.save_checkpoint(cfg, backbone, module_partial_fc, opt, lr_scheduler, global_step, epoch)
                        logging.info("Wrote checkpoint to: {}".format(checkpoint_file))
            # break        

        if cfg.save_all_states:
            logging.info("save all states")
            checkpoint_file = cu.save_checkpoint(cfg, backbone, module_partial_fc, opt, lr_scheduler, global_step, epoch,
                name = os.path.join(cfg.output, f"checkpoint_epoch_{epoch}_gpu_{rank}.pt"))  
            logging.info("wrote all states")

        if rank == 0:
            path_module = os.path.join(cfg.output, f"epoch_{epoch}.pt")            
            logging.info("save epoch model")
            checkpoint_file = cu.save_checkpoint(cfg, backbone, module_partial_fc, opt, lr_scheduler, global_step, epoch, name=path_module)
            logging.info("Wrote epoch model to: {}".format(checkpoint_file))

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        logging.info("save full model")
        checkpoint_file = cu.save_checkpoint(cfg, backbone, module_partial_fc, opt, lr_scheduler, global_step, epoch, name=path_module)
        logging.info("Wrote full model to: {}".format(checkpoint_file))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    main(args)
