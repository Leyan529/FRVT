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
config.margin_list = (1.0, 0.5, 0.0)

# 07/15
# config.network = "r200"
config.network = "resnet_269" 
# config.network = "resnext152_8x14d" 
# config.network = "resnest152_8x14d"


config.embedding_size = 512
config.sample_rate = 0.1  #random sample partal fc class rate choose 0.1-0.3
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.dropout_ratio = 0.4

config.batch_size = 32    # total_batch_size = batch_size * num_gpus * gradient_acc
config.gradient_acc = 4

config.lr = 0.1
config.verbose = 40000
config.dali = False
config.IM_SHAPE = (112, 112, 3)

config.rec = "WebFace42M"
config.num_classes = 2059906
config.num_image = 42474557
config.DATASET= ROOT + config.rec + "/WebFace260M"
config.LABEL_PATH= ROOT + config.rec + "/WebFace260M_dataset.csv"

# config.rec = "WebFace4M"
# config.num_classes = 205990
# config.num_image = 4247455
# config.DATASET= ROOT + config.rec + "/WebFace260M"
# config.LABEL_PATH= ROOT + config.rec + "/WebFace260M_dataset.csv"

config.valrec = ROOT + "FR-val"
config.num_epoch = 20
config.warmup_epoch = 0 # init warm lr
config.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_ff','cfp_fp', "agedb_30", "vgg2_fp"]

config.resume = True
config.output = "work_dirs/%s_"%(config.rec) + config.network + "_%s_%s_%s"%(t.tm_year, t.tm_mon, t.tm_mday)
config.restore_epoch = 18
# config.WEIGHTS= config.output + "/" + "model_epoch_0005_step_840000.pt"


# config.resume = False
# config.output = "work_dirs/%s_"%(config.rec) + config.network + "_%s_%s_%s"%(t.tm_year, t.tm_mon, t.tm_mday)
# config.WEIGHTS= ""


# config.WEIGHT_FROM = "work_dirs/ms1m-retinaface-t1_resnest200_8x14d_2022_7_19/model_epoch_0000_step_006000.pt"

# if not os.path.exists(config.WEIGHTS):
#     if not os.path.exists(config.WEIGHT_FROM): 
#         print("cannot find weight")
#         exit(0)
#     print("copy weight")
#     shutil.copyfile(config.WEIGHT_FROM, config.WEIGHTS)
