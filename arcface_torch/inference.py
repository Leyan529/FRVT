import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("w", img)
    # cv2.waitKey(0)

    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    # img.div_(255).sub_(0.5).div_(0.5)
    img.sub_(127.5).div_(128.0)
    net = get_model(name, fp16=False)
    weight = torch.load(weight)
    net.load_state_dict(weight["state_dict_backbone"])
    net.eval()
    feat = net(img).numpy()
    with open("python_feature_result.txt", "w", encoding="utf-8") as f:
        for i in range(len(feat[0])):
            f.write(str(float(feat[0][i])) + "\n")
            print(feat[0][i])
    # print(feat[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='work_dirs/r50/checkpoint_gpu_0_19_2220000.pt')
    # parser.add_argument('--weight', type=str, default="work_dirs/arcface_LResNet269E/LResnetNet269.pt")
    parser.add_argument('--img', type=str, default="tests/data/00000006.jpg")
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
