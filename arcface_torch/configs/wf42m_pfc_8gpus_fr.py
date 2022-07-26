from easydict import EasyDict as edict
import time
import os
import shutil

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /home/leyan/DataSet

ROOT = "/home/zimdytsai/leyan/DataSet/"
# ROOT = "/home/leyan/DataSet/"

t = time.gmtime()
config = edict()
config.margin_list = (1.0, 0.0, 0.4)

# config.network = "resnet_269" 
# config.network = "resnext152_8x14d" 
# config.network = "resnest152_8x14d"
# config.network = "elannet"        
# config.network = "RepVGG_B3g4"   
# config.network = "RepVGG_B3"   
config.network = "r200"  




config.embedding_size = 512
config.sample_rate = 0.1  #random sample partal fc class rate choose 0.1-0.3
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.dropout_ratio = 0.4
config.ada = False

config.batch_size = 64    # total_batch_size = batch_size * num_gpus * gradient_acc
config.gradient_acc = 8
# config.batch_size = 64    # total_batch_size = batch_size * num_gpus * gradient_acc
# config.gradient_acc = 8

config.lr = 0.1
config.verbose = 4000
config.dali = False
config.IM_SHAPE = (112, 112, 3)

config.rec = "WebFace42M"
# config.num_classes = 2059906
# config.num_image = 42474557
config.num_classes = 1000000
config.num_image = 20653718
config.DATASET= ROOT + config.rec + "/WebFace260M"
config.LABEL_PATH= ROOT + config.rec + "/WebFace260M_dataset.csv"

config.valrec = ROOT + "FR-val"
config.num_epoch = 20
# config.warmup_epoch = config.num_epoch // 10 # init warm lr
config.warmup_epoch = 0 # init warm lr
# config.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_ff','cfp_fp', "agedb_30", "vgg2_fp"]
config.val_targets = ['lfw', 'cplfw', 'cfp_fp']

config.resume = False
config.output = "work_dirs/%s_"%(config.rec) + config.network
config.restore_epoch = 0

# torch.distributed.launch命令介紹
# 我們在訓練分佈式時候，會使用到 torch.distributed.launch
# 可以通過命令，來打印該模塊提供的可選參數 python -m torch.distributed.launch --help

# torch.ditributed.launch參數解析（終端運行命令的參數）：

# nnodes：節點的數量，通常一個節點對應一個主機，方便記憶，直接表述為主機
# node_rank：節點的序號，從0開始
# nproc_per_node：一個節點中顯卡的數量
# -master_addr：master節點的ip地址，也就是0號主機的IP地址，該參數是為了讓 其他節點 知道0號節點的位，來將自己訓練的參數傳送過去處理
# -master_port：master節點的port號，在不同的節點上master_addr和master_port的設置是一樣的，用來進行通信
#  
# torch.ditributed.launch相關環境變量解析（代碼中os.environ中的參數）：

# WORLD_SIZE：os.environ[“WORLD_SIZE”]所有進程的數量
# LOCAL_RANK：os.environ[“LOCAL_RANK”]每張顯卡在自己主機中的序號，從0開始
# RANK：os.environ[“RANK”]進程的序號，一般是1個gpu對應一個進程

# https://blog.csdn.net/magic_ll/article/details/122359490