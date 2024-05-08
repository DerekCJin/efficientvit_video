import csv
import glob
import os.path as osp
import pickle
import random
import numpy as np
import pandas as pd
import torch

import decord

import sys
sys.path.append("../AVION")

import yaml
from pathlib import Path
from argparse import Namespace

#from train_video_cls_model import args

from avion.data.clip_dataset import get_downstream_dataset
from avion.utils.misc import check_loss_nan, generate_label_map

#from imagenet.py:
import copy
import math
import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from efficientvit.apps.data_provider import DataProvider
from efficientvit.apps.data_provider.augment import RandAug
from efficientvit.apps.data_provider.random_resolution import MyRandomResizedCrop, get_interpolate
from efficientvit.apps.utils import partial_update_config
from efficientvit.models.utils import val2list
#end from imagenet.py

__all__ = ["EK100ClsDataProvider"]

#from imagenet.py
class EK100ClsDataProvider(DataProvider):
    name = "ek100_cls"

    data_dir = "EK100_256p/videos"
    n_classes = 1000

    _DEFAULT_RRC_CONFIG = {
        "train_interpolate": "random",
        "test_interpolate": "bicubic",
        "test_crop_ratio": 1.0,
    }


    def __init__(
        self,
        data_dir: str or None = None,
        rrc_config: dict or None = None,
        data_aug: dict or list[dict] or None = None,
        ###########################################
        train_batch_size=128,
        test_batch_size=128,
        valid_size: int or float or None = None,
        n_worker=8,
        image_size: int or list[int] = 224,
        num_replicas: int or None = None,
        rank: int or None = None,
        train_ratio: float or None = None,
        drop_last: bool = False,
    ):
        self.data_dir = data_dir or self.data_dir
        self.rrc_config = partial_update_config(
            copy.deepcopy(self._DEFAULT_RRC_CONFIG),
            rrc_config or {},
        )
        self.data_aug = data_aug

        super().__init__(
            train_batch_size,
            test_batch_size,
            valid_size,
            n_worker,
            image_size,
            num_replicas,
            rank,
            train_ratio,
            drop_last,
        )

    def build_valid_transform(self, image_size: tuple[int, int] or None = None) -> any:
        image_size = (image_size or self.active_image_size)[0]
        crop_size = int(math.ceil(image_size / self.rrc_config["test_crop_ratio"]))
        return transforms.Compose(
            [
                transforms.Resize(
                    crop_size,
                    interpolation=get_interpolate(self.rrc_config["test_interpolate"]),
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std),
            ]
        )

    def build_train_transform(self, image_size: tuple[int, int] or None = None) -> any:
        image_size = image_size or self.image_size

        # random_resize_crop -> random_horizontal_flip
        train_transforms = [
            MyRandomResizedCrop(interpolation=self.rrc_config["train_interpolate"]),
            transforms.RandomHorizontalFlip(),
        ]

        # data augmentation
        post_aug = []
        if self.data_aug is not None:
            for aug_op in val2list(self.data_aug):
                if aug_op["name"] == "randaug":
                    data_aug = RandAug(aug_op, mean=self.mean_std["mean"])
                elif aug_op["name"] == "erase":
                    from timm.data.random_erasing import RandomErasing

                    random_erase = RandomErasing(aug_op["p"], device="cpu")
                    post_aug.append(random_erase)
                    data_aug = None
                else:
                    raise NotImplementedError
                if data_aug is not None:
                    train_transforms.append(data_aug)
        train_transforms = [
            *train_transforms,
            transforms.ToTensor(),
            transforms.Normalize(**self.mean_std),
            *post_aug,
        ]
        return transforms.Compose(train_transforms)

    def build_datasets(self) -> tuple[any, any, any]:
        train_transform = self.build_train_transform()
        valid_transform = self.build_valid_transform()

        #replace efficientvit train_dataset with AVION train_dataset
        #Q: waht is the datatype of train_dataset for efficientvit & for AVION? They must match.
        #train_dataset is an ImageFolder <- DatasetFolder <- VisionDataset <- torch.data.Dataset 

        #In AVION, avion.data.clip_dataset.get_downstream_dataset() returns avion.data.clip_dataset.VideoClassyDataset
        #VideoClassyDataset <- VideoCaptionDatasetBase <- torch.utils.data.Dataset
        #train_dataset = ImageFolder(os.path.join(self.data_dir, "train"), train_transform)
        #test_dataset = ImageFolder(os.path.join(self.data_dir, "val"), valid_transform)

        cfg = yaml.safe_load(Path(".exp/cls/imagenet/b1_r288/args.yaml").read_text())
        args = Namespace(**cfg)
   
        crop_size = 224
        _, mapping_vn2act = generate_label_map(args.dataset)
        train_dataset = get_downstream_dataset(train_transform, crop_size, args, subset='train', label_mapping=mapping_vn2act,)
        test_dataset = get_downstream_dataset(valid_transform, crop_size, args, subset='val', label_mapping=mapping_vn2act,)

        train_dataset, val_dataset = self.sample_val_dataset(train_dataset, valid_transform)
        return train_dataset, val_dataset, test_dataset

#end from imagenet.py



#from AVION clip_dataset.py
def datetime2sec(str):
    if str.count(':')==1:
        str = "00:"+str
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    print("******End of ekk100_cls.get_frame_ids()")
    return frame_ids.astype(int).tolist()


def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0], height=rrc_params[0],
            use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0], height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    print("******End of ekk100_cls.get_video_reader()")
    return video_reader


def video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    assert fps > 0, 'fps should be greater than 0'

    if chunk_len == -1:
        vr = get_video_reader(
            osp.join(root, '{}.{}'.format(vid, ext)),
            num_threads=threads,
            fast_rrc=fast_rrc, rrc_params=rrc_params,
            fast_rcc=fast_rcc, rcc_params=rcc_params,
        )
        end_second = min(end_second, len(vr) / fps)

        # calculate frame_ids
        frame_offset = int(np.round(second * fps))
        total_duration = max(int((end_second - second) * fps), clip_length)
        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

        # load frames
        assert max(frame_ids) < len(vr)
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    
        return torch.from_numpy(frames.astype(np.float32))

    else:
        chunk_start = int(second) // chunk_len * chunk_len
        chunk_end = int(end_second) // chunk_len * chunk_len
        while True:
            video_filename = osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk_end, ext))
            if not osp.exists(video_filename):
                # print("{} does not exists!".format(video_filename))
                chunk_end -= chunk_len
            else:
                vr = decord.VideoReader(video_filename)
                end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
                assert chunk_start <= chunk_end
                break
        # calculate frame_ids
        frame_ids = get_frame_ids(
            int(np.round(second * fps)),
            int(np.round(end_second * fps)),
            num_segments=clip_length, jitter=jitter
        )
        all_frames = []
        # allocate absolute frame-ids into the relative ones
        for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
            rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
            rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
            vr = get_video_reader(
                osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )
            try:
                frames = vr.get_batch(rel_frame_ids).asnumpy()
            except decord.DECORDError as error:
                print(error)
                frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
            except IndexError:
                print(root, vid, ext, second, end_second)
            all_frames.append(frames)
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        print("******End of ekk100_cls.video_loader()")
        return res


class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, is_trimmed=True):
        self.dataset = dataset
        self.root = root
        self.metadata = metadata
        self.is_trimmed = is_trimmed

        if self.dataset == 'ego4d':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset in ['ek100_cls', 'ek100_mir']:
            video_list = glob.glob(osp.join(self.root, '*/*.MP4'))
            fps_dict = {video: decord.VideoReader(video + '/0.MP4').get_avg_fps() for video in video_list}
            self.samples = []
            #print(f"video_list = {video_list}\nfps_dict = {fps_dict}")
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    #print(f"row = {row}")
                    pid, vid = row[1:3]
                    start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                    narration = row[8]
                    verb, noun = int(row[10]), int(row[12])
                    vid_path = '{}/{}'.format(pid, vid)
                    fps = fps_dict[osp.join(self.root, vid_path + '.MP4')]
                    # start_frame = int(np.round(fps * start_timestamp))
                    # end_frame = int(np.ceil(fps * end_timestamp))
                    self.samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun))
            if self.dataset == 'ek100_mir':
                self.metadata_sentence = pd.read_csv(metadata[:metadata.index('.csv')] + '_sentence.csv')
                if 'train' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
                elif 'test' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
                else:
                    raise ValueError('{} should contain either "train" or "test"!'.format(metadata))
                self.relevancy = .1
        else:
            raise NotImplementedError

    def get_raw_item(
        self, i, is_training=True, num_clips=1,
        chunk_len=300, clip_length=32, clip_stride=2,
        sparse_sample=False,
        narration_selection='random',
        threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False, rcc_params=(224,),
    ):
        print("******Entering ek100_cls.get_raw_item()")
        if self.dataset == 'ego4d':
            vid, start_second, end_second, narration = self.samples[i][:4]
            frames = video_loader(self.root, vid, 'mp4',
                                  start_second, end_second,
                                  chunk_len=chunk_len,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if isinstance(narration, list):
                if narration_selection == 'random':
                    narration = random.choice(narration)
                elif narration_selection == 'concat':
                    narration = '. '.join(narration)
                elif narration_selection == 'list':
                    pass
                else:
                    raise ValueError
            return frames, narration
        elif self.dataset == 'ek100_mir':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            frames = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if is_training:
                positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[0].tolist()
                if positive_list != []:
                    pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                    if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                        return frames, (self.metadata_sentence.iloc[pos][1], self.relevancy_mat[i][pos])
            else:
                return frames, (narration, 1)
        elif self.dataset == 'ek100_cls':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            frames = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            return frames, '{}:{}'.format(verb, noun)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoCaptionDatasetCLIP(VideoCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None,
                 is_training=True, tokenizer=None,
                 chunk_len=300,
                 clip_length=32, clip_stride=2,
                 threads=1,
                 fast_rrc=False,
                 rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False,
                 rcc_params=(224,),
                 subsample_stride=None):
        super().__init__(dataset, root, metadata)

        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params

    def __getitem__(self, i):
        frames, caption = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
        )

        # ek100_mir will also output relevancy value
        if isinstance(caption, tuple):
            caption, relevancy = caption
        else:
            relevancy = 0.

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)[0]

        if isinstance(caption, tuple):
            caption, mask = caption
            return frames, caption, mask, relevancy
        else:
            return frames, caption, relevancy


class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, root, metadata, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        chunk_len=300,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample

    def __getitem__(self, i):
        frames, label = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, label


def get_downstream_dataset(transform, crop_size, args, subset='train', label_mapping=None):
    print("******entering ek100_cls.get_downstream_dataset()")
    if subset == 'train':
        return VideoClassyDataset(
            args.dataset, args.root, args.train_metadata, transform,
            is_training=True, label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rrc=args.fused_decode_crop, rrc_params=(crop_size, (0.5, 1.0)),
        )
    elif subset == 'val':
        return VideoClassyDataset(
            args.dataset, args.root, args.val_metadata, transform,
            is_training=False, label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rcc=args.fused_decode_crop, rcc_params=(crop_size, ),
            is_trimmed=not args.dataset == 'charades_ego',
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")
#end from clip_dataset.py