# +
import os
import cv2
from pathlib import Path
import glob
import click


@click.command()
@click.option('--vids', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--output_folder', help='Where to save the output images', required=True, metavar='DIR')
def main(vids, output_folder):
    #video_li = glob.glob('./jtbc_output_mp4/*.mp4')
    video_li = glob.glob(f'{vids}/*.mp4')

    for video_path in video_li:
        print('video_path', video_path)
        vidcap = cv2.VideoCapture(video_path)
        nameonly = Path(video_path).name.split('.')[0]
        
        os.makedirs(f'{output_folder}', exist_ok=True)
        os.makedirs(f'{output_folder}/{nameonly}', exist_ok=True)
        
        cnt = 0
        while(1):
            try:
                ret, image = vidcap.read()

                if(int(vidcap.get(1))):
                    string = str(cnt).zfill(5)
                    cv2.imwrite(f"{output_folder}/{nameonly}/{string}.png",image)
                    cnt += 1
            except:
                break
        vidcap.release()


# -

if __name__=='__main__':
    main()
