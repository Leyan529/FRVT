from easydict import EasyDict as edict
import time
import os
import shutil

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /home/leyan/DataSet

ROOT = "/home/zimdytsai/leyan/DataSet/"
ROOT = "/home/leyan/DataSet/"


t = time.gmtime()
config = edict()
config.margin_list = (1.0, 0.5, 0.0)

# config.network = "r200"
config.network = "resnet_269"  

"""<cardinality, bottleneck_width> =>    (1, 64). (2, 40), (4, 24), (8, 14), (32, 4)"""
# config.network = "resnext200_1x64d" 
# config.network = "resnext200_2x40d" 
# config.network = "resnext200_4x24d" 
# config.network = "resnext200_8x14d"
# config.network = "resnext200_32x4d" 

# config.network = "resnest152_1x64d" 
# config.network = "resnest152_2x40d" 
# config.network = "resnest152_4x24d" 
# config.network = "resnest152_8x14d"
# config.network = "resnest101_8x14d"
# config.network = "resnest152_32x4d"

# config.network = "resnext200_8x14d"
# config.network = "resnest200_8x14d"

# config.network = "resnest200_1x64d"
# config.network = "resnest200_2x40d"
# config.network = "resnest200_4x24d"
# config.network = "resnest200_32x4d"

# config.network = "resnest152_1x64d_r4" 
# config.network = "resnest200_1x64d_r4" 

# # 07/15
# # config.network = "r200"
# # config.network = "resnet_269" 
# # config.network = "resnext152_32x4d" 
# config.network = "resnest152_32x4d"


config.embedding_size = 512
config.sample_rate = 0.1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.dropout_ratio = 0.4
# config.batch_size = 128

config.batch_size = 8    # total_batch_size = batch_size * num_gpus
# config.batch_size = 28
# config.batch_size = 40
config.lr = 0.1
# config.verbose = 2
config.verbose = 2000
config.dali = False
config.IM_SHAPE = (112, 112, 3)

config.rec = "ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.DATASET = ROOT + config.rec + "/ms1m_retinaface"
config.LABEL_PATH= ROOT + config.rec + "/ms1m_retinaface_dataset.csv"

# config.rec = "glint360k"
# config.num_classes = 360232
# config.num_image = 17091657
# config.batch_size = 2

# config.rec = ROOT + "WebFace42M"
# config.num_classes = 2059906
# config.num_image = 42474557

# config.rec =  "WebFace4M"
# config.num_classes = 205990
# config.num_image = 4247455
# config.DATASET=  ROOT + config.rec + "/WebFace260M"
# config.LABEL_PATH=  ROOT + config.rec + "/WebFace260M_dataset.csv"

config.valrec = ROOT + "FR-val2"
config.num_epoch = 20
config.warmup_epoch = 0
# config.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_fp', "agedb_30"]
# config.val_targets = ['lfw', 'calfw', 'cfp_fp', "agedb_30"]
# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.val_targets = ['cplfw']

config.resume = False
# config.output = "work_dirs/%s_"%(config.rec) + config.network + "_%s_%s_%s"%(t.tm_year, t.tm_mon, t.tm_mday)
config.output = "work_dirs/%s_"%(config.rec) + config.network + "_local"
# config.WEIGHTS= config.output + "/" + "model_epoch_0000_step_020000.pt"
config.WEIGHTS= ""
config.WEIGHT_FROM = "work_dirs/ms1m-retinaface-t1_resnest200_8x14d_2022_7_19/model_epoch_0000_step_006000.pt"

# if config.resume and (config.WEIGHTS != ""):
#     if not os.path.exists(config.WEIGHT_FROM): 
#         print("cannot find weight")
#         exit(0)
#     print("copy weight")
#     shutil.copyfile(config.WEIGHT_FROM, config.WEIGHTS)

