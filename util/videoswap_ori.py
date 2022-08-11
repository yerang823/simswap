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
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False, n_batch_size=1): ########################
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
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

        
        
        
    # while ret:
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            detect_results = detect_model.get(frame,crop_size)

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
                    swap_result_list.append(swap_result)
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                    

                reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                                   os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo, \
                                   pasring_model =net,use_mask=use_mask, norm = spNorm)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                # if not no_simswaplogo:
                #     frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
        else:
            break

    video.release()

#     # =====================================================
#     # detection
#     # =====================================================
#     detect_results_li1=[]
#     detect_results_li2=[]
#     #for frame_index in tqdm(range(frame_count)): 
#     for idx,frame_index in enumerate(tqdm(range(frame_count))):  ####
#         if idx==10: #####
#             break ######3
#         ret, frame = video.read()
#         if  ret:
#             detect_results = detect_model.get(frame,crop_size)
#             detect_results_li1.append(detect_results[0])
#             detect_results_li2.append(detect_results[1])
#         else:
#             break

#     video.release()
#     print('len(detect_results_li1)===', len(detect_results_li1))
#     print('len(detect_results_li2)===', len(detect_results_li2))


#     # =====================================================
#     # swap_model
#     # =====================================================

#     data_loader = torch.utils.data.DataLoader(np.array(detect_results_li1), batch_size = 2, shuffle=False, num_workers=4)
#     #print('detect_results_li[0].shape',detect_results_li[0].shape)
#     #print('detect_results_li1[0]====\n',detect_results_li1[0])
#     #print('detect_results_li2[0]====\n',detect_results_li2[0])

#     #for detect_results in tqdm(data_loader):
#     for detect_results in tqdm(data_loader):
#             if not os.path.exists(temp_results_dir):
#                     os.mkdir(temp_results_dir)
#             #frame_align_crop_list = detect_results[0]
#             #frame_mat_list = detect_results[1]

#             frame_align_crop_list=[]
#             for det_res1 in detect_results:
#                 frame_align_crop_list.append(det_res1[0])

#             frame_mat_list=[]
#             for det_res2 in frame_mat_list:
#                 frame_mat_list.append(det_res2[0])

#             swap_result_list = []
#             frame_align_crop_tenor_list = []
#             #for frame_align_crop in frame_align_crop_list:

#                 # BGR TO RGB
#                 # frame_align_crop_RGB = frame_align_crop[...,::-1]

#             frame_align = []
#             for img in frame_align_crop_list:
#                 print('img =======',img)
#                 img = cv2.cvtColor(sum(img,[]), cv2.COLOR_BGR2RGB)
#                 frame_align.append(img)

# #             frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop_list,cv2.COLOR_BGR2RGB))[None,...].cuda()
#             frame_align_crop_tenor = _totensor(frame_align)[None,...].cuda()

#             swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
#             cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
#             #swap_result_list.append(swap_result)
#             #frame_align_crop_tenor_list.append(frame_align_crop_tenor)



#             #reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
#             #    os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)

    

    

    


#     for frame_index, det_res in enumerate(tqdm(detect_results_li)):

#         if det_res is not None:
#             # print(frame_index)
#             if not os.path.exists(temp_results_dir):
#                     os.mkdir(temp_results_dir)
#             frame_align_crop_list = det_res[0]
#             print('len(frame_align_crop_list)====',len(frame_align_crop_list))
#             frame_mat_list = det_res[1]
#             swap_result_list = []
#             frame_align_crop_tenor_list = []
#             for frame_align_crop in frame_align_crop_list:

#                 # BGR TO RGB
#                 # frame_align_crop_RGB = frame_align_crop[...,::-1]

#                 frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

#                 swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
#                 cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
#                 swap_result_list.append(swap_result)
#                 frame_align_crop_tenor_list.append(frame_align_crop_tenor)



#             reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
#                 os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)

#         else:
#             if not os.path.exists(temp_results_dir):
#                 os.mkdir(temp_results_dir)
#             frame = frame.astype(np.uint8)
#             # if not no_simswaplogo:
#             #     frame = logoclass.apply_frames(frame)
#             cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

    
    
    
    
    
    
    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac')

