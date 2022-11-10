from collections import defaultdict
from pathlib import Path
import argparse
from tqdm import tqdm

from signal import signal, SIGPIPE, SIG_DFL  

import sys
import errno
import csv
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

    if os.path.exists('/home/zimdytsai/leyan/DataSet/WebFace42M/mycsvfile.csv'):
        with open('/home/zimdytsai/leyan/DataSet/WebFace42M/mycsvfile.csv') as f:
            r = csv.reader(f)
            for row in r:
                identity_dict[row[0]].append(row[1])

    else:
        with open('/home/zimdytsai/leyan/DataSet/WebFace42M/mycsvfile.csv', 'w') as f:
            writer = csv.writer(f)
            for filepath in tqdm(paths):
                if filepath.is_file():
                    identity_name = filepath.parent.stem
                    filename = filepath.name
                    identity_dict[identity_name].append("%s/%s" % (identity_name, filename))
                    writer.writerow([identity_name , (identity_name, filename)])
        
        print("identity_dict finished")

    num_images = 0
    with open("/home/zimdytsai/leyan/DataSet/WebFace42M/%s_dataset.csv" % dataset_name, "w") as label_f:
        count = 0
        for index, (key, value) in enumerate(identity_dict.items()):
            count = index
            for v in value:
                try:
                    label_f.write("%s,%d\n" % (v, index))
                    num_images = num_images + 1
                except IOError as e:  
                    print("index: ", count, " Error!!")
                    if e.errno == errno.EPIPE:  
                        print()
                        # Handling of the error
            if count+1 >= 1000000: break

    print(f"num_images: {num_images}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--inputdataset', type=str, default="/home/leyan/DataSet/data/ms1m_retinaface", help='trainer name', required=False)
    # parser.add_argument('-i', '--inputdataset', type=str, default="/media/leyan/E/DataSet/glint360k/glint360k_out", help='trainer name', required=False)
    # parser.add_argument('-i', '--inputdataset', type=str, default="/home/leyan/DataSet/WebFace4M/WebFace260M", help='trainer name', required=False)
    parser.add_argument('-i', '--inputdataset', type=str, default="/home/zimdytsai/leyan/DataSet/WebFace42M/WebFace260M", help='trainer name', required=False)
    args = parser.parse_args()
    dataset = main(args.inputdataset)
