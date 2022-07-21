"""Functions that handle saving and loading of checkpoints."""

import os

import torch
# from frlibs.config.config import cfg
# from configs.ms1mv3_bottlenet_resnet import config as cfg


# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"


def get_checkpoint_dir(cfg):
    """Retrieves the location for storing checkpoints."""
    # return os.path.join(cfg.SAVE_DIR, cfg.CHECKPOINT_DIR)
    return os.path.join(cfg.output)


def get_checkpoint(cfg, step, epoch):
    """Retrieves the path to a checkpoint file."""
    # name = "{}{:04d}_step_{:06d}.pyth".format(_NAME_PREFIX, epoch, step)
    name = "{}{:04d}_step_{:06d}.pt".format(_NAME_PREFIX, epoch, step)
    return os.path.join(get_checkpoint_dir(cfg), name)


def get_last_checkpoint(cfg):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir(cfg)
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint(cfg):
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir(cfg)
    if not os.path.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))


def is_checkpoint_epoch(cur_epoch):
    """Determines if a checkpoint should be saved on current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(cfg, model, module_partial_fc=None, optimizer=None, lr_scheduler=None, global_step=0, epoch=0, conf=True, name=None):
    """Saves a checkpoint."""

    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(cfg), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict_backbone": model.module.state_dict(),
        "state_dict_softmax_fc": module_partial_fc.state_dict(),
        "state_optimizer": optimizer.state_dict(),
        "state_lr_scheduler": lr_scheduler.state_dict()
    }

    if conf:
        # checkpoint.update({"cfg": cfg.dump()})
        checkpoint.update({"cfg": str(cfg)})

    # Write the checkpoint
    if name==None:
        checkpoint_file = get_checkpoint(cfg, global_step, epoch)
    else:
        checkpoint_file = name
    torch.save(checkpoint, checkpoint_file, _use_new_zipfile_serialization=False)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, partial_fc=None, optimizer=None, exp_lr_scheduler=None):
    """Loads the checkpoint from the given file."""
    assert os.path.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )
    if not os.path.exists(checkpoint_file): return 0

    dict_checkpoint = torch.load(checkpoint_file, map_location="cpu")
    start_epoch = dict_checkpoint["epoch"]
    global_step = dict_checkpoint["global_step"]
    model.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
    partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
    optimizer.load_state_dict(dict_checkpoint["state_optimizer"])
    exp_lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
    del dict_checkpoint
   
    return start_epoch, global_step
