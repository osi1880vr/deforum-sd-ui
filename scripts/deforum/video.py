import os
import subprocess
from base64 import b64encode


fps = 12 #@param {type:"number"}
#@markdown **Manual Settings**

image_path = "./content/output/2022-09/23_07_20/20220915230720_%05d.png" #@param {type:"string"}
mp4_path = "./content/output/2022-09/23_07_20/20220915230720.mp4" #@param {type:"string"}
max_frames = '137'


def produce_video(image_path, mp4_path, max_frames, fps = 12):

    print(f"{image_path} -> {mp4_path}")

    # make video
    cmd = [
        'C:\\tools\\ffmpeg\\bin\\ffmpeg.exe',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', image_path,
        '-frames:v', str(max_frames),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryslow',
        mp4_path
    ]
    print(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)

    mp4 = open(mp4_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    #display.display( display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )