from collections import defaultdict
from pathlib import Path
import argparse
from tqdm import tqdm

from signal import signal, SIGPIPE, SIG_DFL  

import sys
import errno
import os

def main(dataset_dir):
    """
    ImageDatasetReader Generate dataset label with image file
    :param datasetdirs:face dataset root dir, ImageDataset Folder structure example:
    |---FaceDataset
        |---kobe
            |---000.jpg
            |---001.jpg
            |---002.jpg
        |---james
            |---001.jpg
            |---002.jpg
            |---003.jpg
        |---paul
            |---001.jpg
            |---002.jpg
    """
    rootpath = Path(dataset_dir)
    dataset_name = rootpath.stem
    if not rootpath.is_dir():
        print("please input right dir")
        exit(-1)

    identity_dict = defaultdict(list)

    # signal(SIGPIPE,SIG_DFL)




    paths = rootpath.rglob("*")

    for filepath in tqdm(paths):
        if filepath.is_file():
            identity_name = filepath.parent.stem
            filename = filepath.name
            identity_dict[identity_name].append("%s/%s" % (identity_name, filename))
    
    print("identity_dict finished")

    if(not os.path.exists("tools/tmp")): os.makedirs("tools/tmp")
    with open("tools/tmp/%s_dataset.csv" % dataset_name, "w") as label_f:
        for index, (key, value) in enumerate(tqdm(identity_dict.items())):
            for v in value:
                try:
                    label_f.write("%s,%d\n" % (v, index))
                except IOError as e:  
                    if e.errno == errno.EPIPE:  
                        print()
                        # Handling of the error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--inputdataset', type=str, default="/home/leyan/DataSet/data/ms1m_retinaface", help='trainer name', required=False)
    # parser.add_argument('-i', '--inputdataset', type=str, default="/media/leyan/E/DataSet/glint360k/glint360k_out", help='trainer name', required=False)
    # parser.add_argument('-i', '--inputdataset', type=str, default="/home/leyan/DataSet/WebFace4M/WebFace260M", help='trainer name', required=False)
    parser.add_argument('-i', '--inputdataset', type=str, default="/home/leyan/DataSet/FR-val/IJBC/loose_crop", help='trainer name', required=False)
    args = parser.parse_args()
    dataset = main(args.inputdataset)
