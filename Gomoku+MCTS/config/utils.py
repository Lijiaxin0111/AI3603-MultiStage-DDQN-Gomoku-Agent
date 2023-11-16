import os, shutil
import torch
from tensorboardX import SummaryWriter
from config.options import *
import torch.distributed as dist
import time

""" ==================== Save ======================== """

def make_path():
    return "{}_{}_bs{}_lr{}".format(opts.expri,opts.savepath,opts.batch_size,opts.learn_rate)




def save_model(model,name):
    save_path = make_path()
    if not os.path.isdir(os.path.join(config['checkpoint_base'], save_path)):
        os.makedirs(os.path.join(config['checkpoint_base'], save_path), exist_ok=True)
    model_name = os.path.join(config['checkpoint_base'], save_path, name)
    torch.save(model.state_dict(), model_name) 




""" ==================== Tools ======================== """
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)


def visualizer():
    if get_rank() == 0:
        # filewriter_path = config['visual_base']+opts.savepath+'/'
        save_path = make_path()
        filewriter_path = os.path.join(config['visual_base'], save_path)
        if opts.clear_visualizer and os.path.exists(filewriter_path):   # 删掉以前的summary，以免重合
            shutil.rmtree(filewriter_path)
        makedir(filewriter_path)
        writer = SummaryWriter(filewriter_path, comment='visualizer')
        return writer
