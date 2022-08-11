'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:52
Description: 
'''
import os 
import cv2
import glob
import math
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage, reverse2wholeimage3
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import time

def setup(world_size):
    dist.init_process_group("nccl")#, rank = dist.get_rank(), world_size=world_size)
    
    
def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False, n_batch_size = 1):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    #frame_index = 0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    if use_mask:

        
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         device = torch.device('cuda:0')# if torch.cuda.is_available() else 'cpu')
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth, map_location=device))
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        net.to(device)
        net.eval()

    else:
        net =None

        
    frame_align_crop_tenor_list_3D =[]
    swap_result_list_3D=[]
    frame_mat_list_3D=[]
    frame_li=[]
    frame_index_li=[]
    
    cnt_li=[]
    frame_align_crop_tenor_list2 = []
    
    
    bbox_li=[]
    kpss_li=[]
    for frame_index in tqdm(range(frame_count)): 
        #print('detect and swap model...... ')
        
#         ##########
#         if frame_index==10:
#             break
        
#         #######
        ret, frame = video.read()
        if not ret:
            break
        
        else:
            #st = time.time()
            detect_results = detect_model.get(frame, crop_size)            
            #end = time.time()
            #print(f'======= {end - st} sec, detect_model')
            
            # 일단은 detect_results 가 None 이건 아니건 append 되도록.
            bboxes = detect_results[2]
            kpss = detect_results[3]
            bbox_li.append(bboxes) 
            kpss_li.append(kpss)
            
            
            if detect_results is not None:
                # print(frame_index)
                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                frame_align_crop_list = detect_results[0]
                frame_mat_list = detect_results[1]

                
                swap_result_list = []
                frame_align_crop_tenor_list = []
                for frame_align_crop in frame_align_crop_list:

                    # BGR TO RGB
                    # frame_align_crop_RGB = frame_align_crop[...,::-1]

                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                    
                    #st = time.time()
                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    #end = time.time()
                    #print(f'======= {end - st} sec, swap_model')
                    
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                    swap_result_list.append(swap_result.cpu())
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor.cpu())

                
                frame_align_crop_tenor_list_3D.append(frame_align_crop_tenor_list)
                swap_result_list_3D.append(swap_result_list)
                frame_mat_list_3D.append(frame_mat_list)
                frame_li.append(frame)
                frame_index_li.append(frame_index)
    
                

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                

    video.release()
    
    np.save('/data/GCP_Backup/yerang/bbox_arr.npy',np.array(bbox_li))
    np.save('/data/GCP_Backup/yerang/kpss_arr.npy',np.array(kpss_li))
    

    
    frame_align_crop_tenor_list_3D = np.array(frame_align_crop_tenor_list_3D)
    swap_result_list_3D = np.array(swap_result_list_3D)
    frame_mat_list_3D = np.array(frame_mat_list_3D)
    frame_li = np.array(frame_li)
    frame_index_li = np.array(frame_index_li)
    
    print('frame_align_crop_tenor_list_3D.shape ===',frame_align_crop_tenor_list_3D.shape)
    print('frame_align_crop_tenor_list_3D[0].shape ===',frame_align_crop_tenor_list_3D[0].shape)
    print('frame_align_crop_tenor_list_3D[0][0].shape ===',frame_align_crop_tenor_list_3D[0][0].shape)

    reverse2wholeimage3(frame_align_crop_tenor_list_3D, swap_result_list_3D, frame_mat_list_3D, crop_size, frame_li, logoclass,\
                        temp_results_dir, frame_index_li, no_simswaplogo, \
                        pasring_model =net, use_mask=use_mask, norm = spNorm, batch_size = n_batch_size)
    
    
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))
    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    clips.write_videofile(save_path,audio_codec='aac')
    
    
