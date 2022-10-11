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
from util.reverse2original import reverse2wholeimage,reverse2wholeimage2, reverse2wholeimage3
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


#         setup(world_size=2)
        
#         rank = dist.get_rank()
#         net = BiSeNet(19).to(rank)
#         net = DDP(net, device_ids=[rank])
#         dist.barrier()
#         map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#         CHECKPOINT_PATH = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
#         net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))
        
#         n_classes = 19
#         net = BiSeNet(n_classes=n_classes)
        
        
#         dist.init_process_group("nccl")
#         rank = dist.get_rank()
#         print(f"Start running basic DDP example on rank {rank}.")

#         # create model and move it to GPU with id rank
#         device_id = rank % torch.cuda.device_count()
#         net = net.to(device_id)
#         net = DDP(net, device_ids=[device_id])
        
        
        
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
    
    
    
    for frame_index in tqdm(range(frame_count)): 
        
#         ##########
#         if frame_index==10:
#             break
        
#         #######
        ret, frame = video.read()
        if not ret:
            break
        
        else:
            detect_results = detect_model.get(frame,crop_size)
            
            
            # ==================================================== 
            
#             if detect_results is not None:
#                 # print(frame_index)
#                 if not os.path.exists(temp_results_dir):
#                         os.mkdir(temp_results_dir)
#                 frame_align_crop_list = detect_results[0]
#                 frame_mat_list = detect_results[1]
#                 #swap_result_list = []
#                 frame_align_crop_tenor_list = []
                
#                 cnt_li.append(len(frame_align_crop_list))
#                 for frame_align_crop in frame_align_crop_list:
#                     frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))#[None,...]#.cuda()
#                     frame_align_crop_tenor_list.append(frame_align_crop_tenor[None,...])#.cpu())
#                     frame_align_crop_tenor_list2.append(frame_align_crop_tenor)
                    

# #                     # BGR TO RGB
# #                     # frame_align_crop_RGB = frame_align_crop[...,::-1]

# #                     frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

# #                     swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
#                     cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
# #                     swap_result_list.append(swap_result.cpu())
# #                     frame_align_crop_tenor_list.append(frame_align_crop_tenor.cpu())
                
                

                
#                 frame_align_crop_tenor_list_3D.append(frame_align_crop_tenor_list)
#                 #swap_result_list_3D.append(swap_result_list) ######
#                 frame_mat_list_3D.append(frame_mat_list) 
#                 frame_li.append(frame)
#                 frame_index_li.append(frame_index)
                
#             else:
#                 if not os.path.exists(temp_results_dir):
#                     os.mkdir(temp_results_dir)
#                 frame = frame.astype(np.uint8)
#                 cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
    
    
#     # for frame ~~ 구문 끝
#     swap_result_list=[]
#     if len(frame_align_crop_tenor_list2) > 0:
#         frame_align_crop_tenor_list2 = np.array(frame_align_crop_tenor_list2)
#         print('frame_align_crop_tenor_list2.shape = ',frame_align_crop_tenor_list2.shape)
#         for dataset in tqdm(DataLoader(frame_align_crop_tenor_list2, n_batch_size, shuffle=False)):
#             print('dataset.shape=', dataset.shape)
#             swap_result = swap_model(None, dataset.cuda(), id_vetor, None, True)[0]

#             for out in swap_result:
#                 #print('out.shape ===', out.shape)
#                 swap_result_list.append(out[None,...].cpu()) 

#         ii=0
#         for cnt in cnt_li:
#             tmp_li=[]
#             for i in range(cnt):
#                 tmp_li.append(swap_result_list[ii])
#                 ii+=1
#             swap_result_list_3D.append(tmp_li)
        
        
            
            
            
#             if detect_results is not None:
#                 frame_align_crop_tenor_list = torch.cat(frame_align_crop_tenor_list, dim = 0)
                
#                 swap_result_list=[]
#                 #for frame_align_crop_tenor in frame_align_crop_tenor_list:
                
#                 print('frame_align_crop_tenor_list.shape = ',frame_align_crop_tenor_list.shape)
#                 for dataset in tqdm(DataLoader(frame_align_crop_tenor_list, n_batch_size, shuffle=False)):
#                     print('n_batch_size',n_batch_size)
#                     print('dataset.shape ======',dataset.shape)
#                     swap_result = swap_model(None, dataset.cuda(), id_vetor, None, True)[0]
#                     swap_result_list.append(swap_result.cpu())

#                 swap_result_list_3D.append(swap_result_list)
            
            
            # ==================================================== 
            
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
                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
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
    
    
#     path = os.path.join(temp_results_dir,'*.jpg')
#     image_filenames = sorted(glob.glob(path))
#     clips = ImageSequenceClip(image_filenames,fps = fps)

#     if not no_audio:
#         clips = clips.set_audio(video_audio_clip)

#     clips.write_videofile(save_path,audio_codec='aac')
    
    
