import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
from utils.augmenter import Augmenter
from PIL import Image
import torchvision.datasets as datasets
import cv2
from imgaug import augmenters as iaa

IMG_NORM_MEAN = [123.675, 116.28, 103.53]
IMG_NORM_STD = [58.395, 57.12, 57.375]

def readDatasetLabel(rootdir, labelfile):
    image_list = []
    label_list = []
    with open(labelfile, "r") as f:
        for line in f.readlines():
            line = line.strip()
            infor = line.split(",")
            image_list.append(os.path.join(rootdir, infor[0]))
            label_list.append(int(infor[1]))
    return image_list, label_list

class CustomImageFolderDataset(ImageFolder):

    def __init__(self,
                 root,
                #  labelfile,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=False,
                #  output_dir='./',
                 ):

        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
        self.root = root
        # self.labelfile = labelfile
        # self.image_list, self.label_list = readDatasetLabel(self.root, labelfile)
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.swap_color_channel = swap_color_channel
        # self.output_dir = output_dir  # for checking the sanity of input images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        sample = self.augmenter.augment(sample)

        # sample_save_path = os.path.join(self.output_dir, 'training_samples', 'sample.jpg')
        # if not os.path.isfile(sample_save_path):
        #     os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
        #     cv2.imwrite(sample_save_path, np.array(sample))  # the result has to look okay (Not color swapped)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target # 86235

class FrDataset(Dataset):
    def __init__(self, data_folder, labelfile, image_size=(112, 112), open_aug=False, 
                 transform=None,
                 target_transform=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=False,):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception("%s  not exists." % self.data_folder)

        self.image_size = image_size
        self.image_list, self.label_list = readDatasetLabel(data_folder, labelfile)
        self.num_classes = len(set(self.label_list))
        self.open_aug = open_aug

        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.swap_color_channel = swap_color_channel

        self.transform = transform
        self.target_transform = target_transform

    def _image_preprocess(self, image, image_size):
        processedimage = cv2.resize(image, image_size)
        processedimage = image_preprocess_fromarray(processedimage)
        processedimage = torch.FloatTensor(processedimage.transpose((2, 0, 1)).astype(float))
        return processedimage

    def __getitem__(self, index):
        f = self.image_list[index]
        sample = Image.open(f).convert('RGB')   
        # target = torch.tensor(self.label_list[index])
        target = self.label_list[index]

        sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        sample = self.augmenter.augment(sample)

        # sample_save_path = os.path.join(self.output_dir, 'training_samples', 'sample.jpg')
        # if not os.path.isfile(sample_save_path):
        #     os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
        #     cv2.imwrite(sample_save_path, np.array(sample))  # the result has to look okay (Not color swapped)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)     

        return sample, target

        # if self.open_aug:
        #     bgr_image = image_aug(bgr_image)

        # processed_image = self._image_preprocess(bgr_image, self.image_size)
        # label = torch.tensor(self.label_list[index])
        # return processed_image, label        

    def __getitem_o__(self, index):
        bgr_image = cv2.imread(self.image_list[index]) # (112, 112, 3)

        if self.open_aug:
            bgr_image = image_aug(bgr_image)

        processed_image = self._image_preprocess(bgr_image, self.image_size)
        label = torch.tensor(self.label_list[index])
        return processed_image, label

    def __len__(self):
        return len(self.image_list)
        
def get_dataloader(
    root_dir,
    labelfile,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # Image Folder
    else:
        transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        # train_set = ImageFolder(root_dir, transform)

        train_set = FrDataset(root_dir,
                            labelfile,
                            image_size=(112, 112),
                            open_aug=True,
                            transform=transform,
                            low_res_augmentation_prob=0.2,
                            crop_augmentation_prob=0.2,
                            photometric_augmentation_prob=0.2,
                            swap_color_channel=False
                            )

    # DALI
    if dali:
        rec = "/media/leyan/E/DataSet/ms1m-retinaface-t1/train.rec"
        idx = "/media/leyan/E/DataSet/ms1m-retinaface-t1/train.idx"
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5)):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()

def image_preprocess_fromarray(image):
    bgr_image = image
    # if cfg.IMG_NORM.TO_RGB:
    tobeprocessimg = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # else:
    #     tobeprocessimg = bgr_image
    tobeprocessimg = tobeprocessimg.astype(np.float32)
    tobeprocessimg[:, :, 0] = (tobeprocessimg[:, :, 0] - IMG_NORM_MEAN[0]) / IMG_NORM_STD[0]
    tobeprocessimg[:, :, 1] = (tobeprocessimg[:, :, 1] - IMG_NORM_MEAN[1]) / IMG_NORM_STD[1]
    tobeprocessimg[:, :, 2] = (tobeprocessimg[:, :, 2] - IMG_NORM_MEAN[2]) / IMG_NORM_STD[2]
    return tobeprocessimg

def image_aug(image):
    #seq = iaa.SomeOf((1, 5),
    #                 [
    #                     iaa.Multiply((0.4, 1.5), name="Multiply"),
    #                     iaa.Fliplr(0.5, name="Flip"),
    #                     #iaa.CoarseDropout((0.03, 0.1), size_percent=(0.01, 0.03), name="RandomDropout"),
    #                     iaa.AdditiveGaussianNoise(scale=0.05 * 224),
    #                     iaa.OneOf([iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
    #                                iaa.AverageBlur(k=(0, 3), name="AverageBlur"),
    #                                iaa.MotionBlur(k=6, name="MotionBlur")
    #                                ]),
    #                     iaa.Rotate((-10, 10))
    #                 ])
    seq = iaa.Sequential([
        iaa.Fliplr(0.5, name="Flip")
    ])
    processed_image = np.squeeze(seq(images=np.expand_dims(image, axis=0)), axis=0)
    return processed_image    