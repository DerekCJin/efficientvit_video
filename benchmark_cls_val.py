# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
#
# model: /data/models/efficientvit/cls/b1-r288.pt
# run in /opt/efficientvit: python3 benchmark_cls.py --model b1-r288

import argparse
import math
import os
import datetime
import socket
#import pandas as pd
import pandas as pd
import numpy as np
import cv2
from PIL import Image

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.cls_model_zoo import create_cls_model

import pandas as pd

import csv

#import timeit
import time
from time import process_time
import torch.utils.benchmark as benchmark


def load_image(data_path: str, mode="rgb"):
    #print(f"***data_path={data_path}")
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/ssd/data/datasets/EK100_256p/images/val_1frame/")
    parser.add_argument("--gpu", type=str, default="all")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=50)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    #parser.add_argument("--model", type=str, default="l2")
    parser.add_argument("--model", type=str, default="/ssd/data/models/l1-r224.pt")
    #op notes: put model also in ~/efficientvit/efficientvit/assets/checkpoints/cls
    #parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--weight_url", type=str, default="/ssd/data/models/l1-r224.pt")
    #parser.add_argument("--image_path", type=str, default="assets/fig/cat.jpg") #get 1 image in tiny_imagenet/test folder
    parser.add_argument("--output_path", type=str, default="/home/ubuntu/downloads/data/benchmarks/benchmark_efficientvit_cls_l1_r244_val_1frame.txt")
    parser.add_argument('-s', '--save', type=str, default='/home/ubuntu/downloads/data/benchmarks/benchmark_efficientvit_cls_l1_r244_val_1frame.txt', help='txt file to save benchmarking results to')

    nfx = 2.00
    walk_root_dir = '/ssd/data/datasets/EK100_256p/images/val_1frame/'

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    

    #get 3 images from imagenet/test
    #if not args.images:

    print(args)

    model = create_cls_model(args.model, args.weight_url)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(
                int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(args.image_size),
                                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cls_to_idx = {}
    idx_to_cls = {}

    #flip key/value confirmed to be done correctly cls_to_idx_ek100_train.csv
    reader = csv.reader(open('./cls_to_idx_ek100_train.csv', 'r'))
    for row in reader:
        k, v = row
        cls_to_idx[k] = v
    idx_to_cls = dict((v, k) for k, v in cls_to_idx.items())

    predicted_idx = {}
    predicted_cls = {}
    actual_cls = {}
    actual_idx = {}
    rt = {}
    stem = ""    
  
    total_pred, correct_pred = 0,0
    not_in_train_split = 0
    not_in_train_split_classes = []

    #df = pd.read_csv("/data/files/EPIC_100_train.csv")
    df = pd.read_csv("./EPIC_100_validation.csv")
    
    stop = False
    n_files = 0
    arr_frames = []
    arr = []
    seg_id_to_frames = {} # {seg_id:arr_of_frames}
    seg_id = ""
    
    stop = False
    print("***before os.walk()")
    for root, dirs, files in os.walk(walk_root_dir):  
        path = root.split(os.sep)
        print(f"root = {root} path = {path}")
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            print(len(path) * '---', file)
            stem, ext = os.path.splitext(file)
            #fn, ext = os.path.splitext(file)
            #stem = fn[:fn.rfind('_')]
            print(f"stem = {stem}")
            # group iamges belonging to a segment.
            if ext == ".jpg":
                n_files += 1
                #if n_files > 500:
                #    stop = True
                seg_id = stem[:stem.rfind("_")]
                if seg_id in seg_id_to_frames.keys():
                    # print(f"in if")
                    seg_id_to_frames[seg_id].append(file)
                    # print(f"seg_id_to_frames[{seg_id}] = {seg_id_to_frames[seg_id]}")
                else:
                    # print(f"root = {root}")
                    seg_id_to_frames[seg_id] = [root]
                    seg_id_to_frames[seg_id].append(file)
                    # print(f"in else: frames for {seg_id} = {seg_id_to_frames[seg_id]}")
        #if stop == True:
        #    break
    import pickle
    pickle.dump(seg_id_to_frames, open('g.pkl', 'wb'))

    #print(f"seg_id_to_frames={seg_id_to_frames}")
    #print(f"seg_id_to_frames.keys = {seg_id_to_frames.keys()}")
    print(f"...........starting  inference")

    #seg_id format: P01_01_01
    n_seg = 0
    s_id_1 = ""
    sum_output = torch.zeros([1, 3568]).cuda()
    for s_id in seg_id_to_frames.keys():
        #n_seg += 1 #REMOVE
        #if n_seg >100:
        #    stop = True
        #    break
        frames = seg_id_to_frames[s_id]
        #print(f"seg_id_to_frames={seg_id_to_frames[s_id]}")
        img_dir = frames[0]
        df1 = df[df['narration_id']==s_id]
        vcls = df1['verb_class'].iloc[0]
        ncls = df1['noun_class'].iloc[0]
        actual_cls0 = str(vcls)+"_"+str(ncls)

        print(f"seg_id = {s_id}, vcls = {vcls}, ncls = {ncls}")

        try:
            actual_idx0 = cls_to_idx[actual_cls0]
        except:
            #print(f"not in train split: {actual_cls0}")
            not_in_train_split_classes.append(actual_cls0)
            not_in_train_split += 1
            continue
    
        actual_cls[s_id] = actual_cls0
        actual_idx[s_id] = actual_idx0
        print(f"actual_cls0 = {actual_cls0} actual_idx0 = {actual_idx0}\n") 
        #reset sum_output for each segment
        sum_output = torch.zeros([1, 3568]).cuda()
        
        n_imgs = 0
        #t0=time.process_time()
        t0=time.time_ns()
        for frame in frames:
            #print(f"in for frame in frames loop. frame={frame}")
            #The first element is the dir of the imgs, skip it
            if frame.endswith(".jpg") == False:
                continue
            n_imgs += 1
            img_full_path = img_dir+"/"+frame
            #print(f"s_id = {s_id}, img full path = {img_full_path}")
            image = load_image(img_full_path)
            image = transform(image)
            image = image.unsqueeze(0)
            with torch.no_grad():
                output = model(image)
            sum_output += output
            #print(f"output = {output}")
        #rt0 = (time.process_time()-t0)*1000.00
        rt0 = ((time.time_ns()-t0)/1000000.00)/float(len(frames))
        rt[s_id]=rt0    

        total_pred = total_pred+1
        if sum_output.numel() == 0:
            continue
        l = torch.argmax(sum_output)

        #predicted index is wrong
        predicted_idx0 = str(l.item())
        predicted_idx[s_id]=predicted_idx0
        try:
            predicted_cls0 = idx_to_cls[predicted_idx0]
        except:
            predicted_cls0 = ""

        predicted_cls[s_id]=predicted_cls0
        print(f"predicted_cls0 = {predicted_cls0} predicted_idx0 = {predicted_idx0}\n")
        #key_type = type(list(idx_to_cls.keys())[0])
        #cls0 = idx_to_cls['3812']
        if actual_cls0 == predicted_cls0:
            correct_pred = correct_pred+1
        #if stop == True: #REMOVE
        #    break
        print(f"end of iteration: {s_id}")
    acc = round(float(correct_pred/total_pred),6)
   
    rt_list = list(rt.values())
    rt_list.pop(0)
    print(f"rt_list[0] = {rt_list[0]}")
    avg_rt = sum(rt_list) / len(rt_list)     

    print(f"args.save = {args.save}")
    if args.save:

        if not os.path.isfile(args.save):  # txt header
            with open(args.save, 'w') as file:
                file.write(f"timestamp, hostname, api, model, weight_url\n\n")
        else:
            os.remove(args.save)
            with open(args.save, 'w') as file:
                file.write(f"timestamp, hostname, api, model, weight_url\n\n")
        with open(args.save, 'a') as file:
            file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
            file.write(f"efficientvit-python, {args.model}, {args.weight_url}\n\n")
            #image = name of image = stem
            #predicted label = l.item() = predicted_idx[stem]
            #actual label = cls_to_idx[df[stem][narration]]
            #predicted class = predicted_cls[stem]
            #actual class = actual_cls[stem], actual cls from df
            file.write(f"image\t\tpredicted class\tactual_class\tpredicted_label\tactual_label\t\truntime(ms/img)\n")
            for k,v in predicted_cls.items():
                file.write(f"{k}\t{v}\t{actual_cls[k]}\t{predicted_idx[k]}\t{actual_idx[k]}\t\t{rt[k]}\n")

            file.write(f"\n\nTotal predictions: {total_pred}\nCorrect predictions: {correct_pred}\nAccuracy = {acc}")
            file.write(f"\n\nNot in train split: {not_in_train_split}\n\n")
            file.write(f"average inference time per img: {avg_rt}")
if __name__ == "__main__":
    main()

