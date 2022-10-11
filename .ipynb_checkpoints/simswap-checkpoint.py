# +
import os
import cv2
import glob
import sys
import click
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
#from options.test_options import TestOptions
# from insightface_func.face_detect_crop_multi import Face_detect_crop
# from util.videoswap_specific import video_swap
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap import video_swap
from util.add_watermark import watermark_image
from options.test_options import TestOptions

import time, math


# -

# @click.command()
# @click.option('--pics', 'pics', help='Target image folder to project to', required=True, metavar='DIR')
# @click.option('--vids', help='Where to save the output images', required=True, metavar='DIR')
# @click.option('--output_folder', help='Where to save the output images', required=True, metavar='DIR')
#def main(pics, vids, output_folder, num_batch_size=1):
def main():

    opt = TestOptions()
    opt.initialize()
    opt.parser.add_argument('-f') ## dummy arg to avoid bug
    opt = opt.parse()
    
    pics = glob.glob(opt.pics+'/*')
    vids = glob.glob(opt.vids+'/*')
    pics.sort()
    vids.sort()
    
    os.makedirs(opt.output_folder, exist_ok=True)

    transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    detransformer = transforms.Compose([
            transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
        ])

    

    for v in vids:
        start = time.time()
        math.factorial(100000)
        
        for p in pics:
            print(f"Processing {p} to {v}.")
            #from options.test_options import TestOptions
            #opt = TestOptions()
            #opt.initialize()
            #opt.parser.add_argument('-f') ## dummy arg to avoid bug
            #opt = opt.parse()
            opt.pic_a_path = p ## or replace it with image from your own google drive
            opt.video_path = v ## or replace it with video from your own google drive
            v_name = v.split('/')[-1].split('.')[0]
            p_name = p.split('/')[-1].split('.')[0]
            #opt.output_path = f'{output_folder}/{v_name}_{p_name}.mp4'
            #if os.path.exists(opt.output_path):
            #    continue
            
                
                
            opt.temp_path = opt.output_folder+f'{v_name}+{p_name}'            
            #os.makedirs(opt.temp_path, exist_ok=True)
            
            
            opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
            opt.isTrain = False
            opt.use_mask = True  ## new feature up-to-date

            crop_size = opt.crop_size

            torch.nn.Module.dump_patches = True
            model = create_model(opt)
            model.eval()

            app = Face_detect_crop(name='antelope', root='./insightface_func/models')
            app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))


            with torch.no_grad():
                pic_a = opt.pic_a_path
                # img_a = Image.open(pic_a).convert('RGB')
                img_a_whole = cv2.imread(pic_a)
                img_a_align_crop, _,_,_ = app.get(img_a_whole,crop_size)
                img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
                img_a = transformer_Arcface(img_a_align_crop_pil)                
                img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

                # convert numpy to tensor
                img_id = img_id.cuda()

                #create latent id
                img_id_downsample = F.interpolate(img_id, size=(112,112))
                latend_id = model.netArc(img_id_downsample)
                #latend_id = model.module.netArc(img_id_downsample)
                latend_id = latend_id.detach().to('cpu')
                latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
                latend_id = latend_id.to('cuda')
                
                

                video_swap(opt.video_path, latend_id, model, app, temp_results_dir=opt.temp_path, use_mask=opt.use_mask, n_batch_size=opt.num_batch_size, vid_name = v_name)

            print(f"Success {p} to {v}.")
        end = time.time()
        print('===========================================')
        print(f"{end - start:.5f} sec")
        print('===========================================')


if __name__ == '__main__':
    #pics = '../dataset/img/bhn/'
    #vids = '../dataset/video/shinhan/'
    #output_folder = '../result/tmp_sim/bhn'
    #batch_size = 1
    #main(pics, vids, output_folder, batch_size)
    main()


