from datetime import timedelta
from signal import valid_signals
import cv2
import numpy as np
import os
import time
import pandas as pd

import shutil

import argparse

#EPIC-100-validation.csv. useful fiedls: 
# video_id  P01_01
# start_frame: 8
# stop_frame: 202
# narration: open door

# number of frames per video segment to extract
nfx = 0

df = pd.DataFrame()
dfg = pd.DataFrame()

#project_root_dir = "~/Documents/GitHub/pythoncode-tutorials/python-for-multimedia/extract-frames-from-video/"
#video_root_dir = "./EK100_256p/"
#train_img_root_dir = "~/Documents/GitHub/pythoncode-tutorials/python-for-multimedia/extract-frames-from-video/extracted_images/train/"
#img_root_dir = "~/Documents/GitHub/pythoncode-tutorials/python-for-multimedia/extract-frames-from-video/extracted_images/"

#project_root_dir = "~/Documents/GitHub/pythoncode-tutorials/python-for-multimedia/extract-frames-from-video/"
project_root_dir = ""
ek100_root_dir = "EK100_256p"
video_root_dir = "videos/"
img_root_dir = "images/"

set = ""
set_img_root_dir = ""
n=0
def make_img_dirs():
    os.chdir(img_root_dir)
    if os.path.exists(set_img_root_dir):
        print(f"before calling rmtree")
        shutil.rmtree(set_img_root_dir)

    print(f"current dir = {os.getcwd()}, set_img_root_dir = {set_img_root_dir}")
    #os.mkdir(set)
    #os.chdir(set)
    print(f"in make_img_dirs(): dfg.size = {dfg.size}")
    arr_vcls_ncls = []
    x=0     
    for idx, r in dfg.iterrows():
        vcls = r['verb_class']
        ncls = r['noun_class']
        v_n_img = str(vcls)+"_"+str(ncls)+"/images/"
        #print(f"vcls = {vcls}, ncls = {ncls}, v_n_img = {v_n_img}, {set_img_root_dir+v_n_img}")
        x+=1
        #print(f"{x}")
        os.makedirs(set_img_root_dir+v_n_img)
    print(f"x={x}")
    os.chdir(project_root_dir)

def is_narration_valid(nid):
    n = df[df['narration_id']==nid]['narration'].iloc[0].strip()
    #print(f"*****narration for nid {nid} = {n}")
    return ' ' in n

#
# vid in the form of P01_1
        
def extract_frames_for_video(vid):
    print(f"extracting vid = {vid}")
    pid = vid[:vid.index('_')]                     #this is the participant id, e.g., P01
    video_dir_4_pid = os.path.join(video_root_dir, pid) #e.g., ./EK100_256p/P01
    #print(f"video_dir_4_pid = {video_dir_4_pid}")

    video_filename =video_dir_4_pid +"/"+vid+".MP4" #e.g., ./EK100_256p/P01/P01_01.MP4
    #cap should be opened for the enteir video, not for each snippet. So should be in this function
    cap = cv2.VideoCapture(video_filename)
    #df4vid = df4val[df4val['video_id']==vid]
    df4vid = df[df['video_id']==vid]
    n=0
    for nid in df4vid['narration_id']:                  # for each snippet
        print(f"nid = {nid}")
        #if is_narration_valid(nid):
        #    n += 1
        extract_frame_for_snippet(cap, nid)
    print(f"vid = {vid}, number of valid narration_id = {n}")
    cap.release()
    #cv2.destroyAllWindows()
    return n

# nid example P01_01_0
def extract_frame_for_snippet (cap, nid):
    df4nid = df[df['narration_id']==nid]
    vid = df4nid.iloc[0]['video_id']
    pid = vid[:vid.index('_')] 
 
    #frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame = df4nid.iloc[0]['start_frame']
    stop_frame = df4nid.iloc[0]['stop_frame']
    frame_count = stop_frame-start_frame+1
    
    frame_num_increment = int(frame_count/(nfx+1))
    if frame_num_increment == 0:
        frame_num_increment = 1
    
    selected_frames = []
    print(f"start_frame = {start_frame} stop_frame = {stop_frame}")
    print(f"frame_count = {stop_frame-start_frame+1}\tframe_num_increment = {frame_num_increment}\n")
    #print(f"selected frames = {selected_frames}")
    n_img = 0
    n=0
    #for i in range(frame_num_increment,frame_count, frame_num_increment):
    for i in range(start_frame+frame_num_increment, stop_frame, frame_num_increment):
        #n+=1
        #if n>100:
        #    break
        selected_frames.append(i)
        fn = str(i)
        #print(f"frame_num_increment = {frame_num_increment}, fn = {fn}")

        v = df4nid[df4nid['narration_id']==nid]['verb_class'].iloc[0]
        n = df4nid[df4nid['narration_id']==nid]['noun_class'].iloc[0]
        v_n = str(v)+"_"+str(n)
        img_filename = str(nid)+"_"+fn+".jpg"
        #img_filename = v_n+"_"+fn+".jpg" # nid = P01_01_0, img_filename needs to be P01_01_0_nn.jpg. nn=frame_number. Put this file under a folder named after the narration/action of the nid/snippet
        #img_file_path = img_root_dir+action+"/"+img_filename #e.g., ./extracted_images/EK100_256p/open door/P01_01_0.jpg     
        img_file_path = set_img_root_dir+v_n+"/images/"
        img_file_full_path = img_file_path+img_filename
        print(f"img_file_full_path = {img_file_full_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if success == False:
            j = i+15 
            if j >= stop_frame:
                j=stop_frame-15
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            success, frame = cap.read()    

            if success == False:
                j = i-15
                if j <= stop_frame:
                    j=start_frame+15
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                success, frame = cap.read()    


        #if v_n == "71_115" or v_n == "71_115" or v_n =="9_115" or v_n == "80_245" or v_n == "46_125":
        #   print(f"v_n = {v_n}, i = {i}\n")
        #    print(f"img_file_path = {img_file_path}, img_file_path exists = {os.path.exists(img_file_path)}, success = {success}, img_file_full_path = {img_file_full_path}\n")
        #    time.sleep(100000)

        #cv2.imshow("Video", frame)        
        if os.path.exists(img_file_path) & success:
            #print(f"write frame to {img_file_path}")
            cv2.imwrite(img_file_full_path, frame)
        else:
            if success == False:
                print("cap.read failed")
            else:
                print(f"path doesnt exist: {img_file_path}")

if __name__ == "__main__":
    print(f"project_root_dir = {project_root_dir}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="train")
    parser.add_argument("--nfx", type=int, default=16)

    args = parser.parse_args()

    set = args.set
    nfx = args.nfx
    # nfx == 0: don't extract frames, put video files in the right vcls_ncls
    print(args)

    project_root_dir = os.getcwd()+"/"
    ek100_root_dir = project_root_dir+"EK100_256p/"
    video_root_dir = ek100_root_dir+"videos/"
    img_root_dir = ek100_root_dir+"images/"

    
    if args.set == "train":
        df = pd.read_csv("./EPIC_100_train.csv")
    elif args.set == "val":
        print(f"****set = {args.set}")
        df = pd.read_csv("./EPIC_100_validation.csv")
    elif args.set == "test":
        df = pd.read_csv("./EPIC_100_test_timestamps.csv")
    #dfg = df.groupby(['verb_class','noun_class']).size().reset_index().rename(columns={0:'count'})
    dfg = df[['verb_class', 'noun_class']].drop_duplicates()
    print(f"dfg.size = {dfg.size}")
    
    set_img_root_dir = img_root_dir+args.set+"/"
    
    print(f"set_img_root_dir = {set_img_root_dir}")
    make_img_dirs()

    #vid = video_id (e.g., P01_01. 2 parts: PID_VID), identifies a video (file), a row in the validation.csv file.  
    #   1st part = participant id, 2nd part video id for the participant
    #nid = narration_id. Each video has multiple snippets, each snippet = an action segment. 
    #   Action's name is in the narration column. 

    # extract frames for each video. A video may have multiple snippets. 
    # A frame is extracted from each snippet and saved in a folder named after narration/action
    # Example vid = P01_01. The video P01_01.MP4 is under folder P01

    #walk_root_dir = "./extracted_images_EK100_256p/EK100_256p"
    walk_root_dir = video_root_dir
    # stem: example P01_01

    stop = False
    n = 0
    
    os.chdir(project_root_dir)
    for root, dirs, files in os.walk(walk_root_dir):  
        path = root.split(os.sep)
        print(f"root = {root} path = {path}")
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            print(len(path) * '---', file)
            stem, ext = os.path.splitext(file)
            if ext == ".MP4":
                #if len(df[df['video_id']==stem]) > 0:
                extract_frames_for_video(stem)
    
    
    
