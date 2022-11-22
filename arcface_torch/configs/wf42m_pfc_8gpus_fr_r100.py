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
config.network = "r100"  




config.embedding_size = 512
config.sample_rate = 0.1  #random sample partal fc class rate choose 0.1-0.3
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 25e-5
config.dropout_ratio = 0.4
config.ada = False

config.batch_size = 256    # total_batch_size = batch_size * num_gpus * gradient_acc
config.gradient_acc = 2
# config.batch_size = 64    # total_batch_size = batch_size * num_gpus * gradient_acc
# config.gradient_acc = 8

config.lr = 0.1
config.verbose = 12000
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
config.num_epoch = 41
# config.warmup_epoch = config.num_epoch // 10 # init warm lr
config.warmup_epoch = 0 # init warm lr
# config.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_ff','cfp_fp', "agedb_30", "vgg2_fp"]
config.val_targets = ['lfw', 'cplfw', 'cfp_fp']

config.resume = False
config.output = "work_dirs/%s_"%(config.rec) + config.network
config.restore_epoch = 0