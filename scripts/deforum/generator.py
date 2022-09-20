import os, subprocess
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import requests

from ldm.util import instantiate_from_config
import cv2
import math
import torch
from skimage.exposure import match_histograms
from einops import rearrange, repeat
import numpy as np
import py3d_tools as p3d
import os
from helpers import DepthModel, sampler_fn
import pandas as pd
import json
from omegaconf import OmegaConf
import random
import sys
import pathlib

from tools.modelloader import load_models
import platform
import streamlit as st

sys.path.extend([
    'src/taming-transformers',
    'src/clip',
    'src/stable-diffusion/',
    'src/k-diffusion',
    'src/pytorch3d-lite',
    'src/AdaBins',
    'src/MiDaS',
])

if "Linux" in platform.platform():
    ffmpeg = 'ffmpeg'
else:
    ffmpeg = 'C:\\tools\\ffmpeg\\bin\\ffmpeg.exe'


from k_diffusion.external import CompVisDenoiser

device = ('cuda')
models_path = "./content/models"  # @param {type:"string"}
#model = None

def sanitize(prompt):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tmp = ''.join(filter(whitelist.__contains__, prompt))
    return tmp.replace(' ', '_')


def render_image_batch(args):

    load_models()
    if st.session_state["pathmode"] == 'root':
        args.outdir = f'{args.outdir}/_batch_images'
    #= args.prompts

    prompts = list(args.prompts.split("\n"))

    args.prompts = {k: f"{v:05d}" for v, k in enumerate(prompts)}
    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_settings or args.save_samples:
        print(f"Saving to {os.path.join(args.outdir, args.timestring)}_*")

    image_pipe = st.session_state["preview_image"]
    video_pipe = st.session_state["preview_video"]

    # save settings for the batch
    if args.save_settings:
        if st.session_state["pathmode"] == "subfolders":
            settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        else:
            os.makedirs(os.path.join(args.outdir, f"_configs/_batch_configs"), exist_ok=True)
            settings_filename = os.path.join(args.outdir, f"_configs/_batch_configs/{args.timestring}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(args.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)

    index = 0

    # function for init image batching
    init_array = []
    if args.use_init:
        if args.init_image == "":
            raise FileNotFoundError("No path was given for init_image")
        if args.init_image.startswith('http://') or args.init_image.startswith('https://'):
            init_array.append(args.init_image)
        elif not os.path.isfile(args.init_image):
            if args.init_image[-1] != "/":  # avoids path error by adding / to end if not there
                args.init_image += "/"
            for image in sorted(os.listdir(args.init_image)):  # iterates dir and appends images to init_array
                if image.split(".")[-1] in ("png", "jpg", "jpeg"):
                    init_array.append(args.init_image + image)
        else:
            init_array.append(args.init_image)
    else:
        init_array = [""]

    # when doing large batches don't flood browser with images
    clear_between_batches = args.n_batch >= 32

    for iprompt, prompt in enumerate(prompts):
        args.prompt = prompt
        print(f"Prompt {iprompt + 1} of {len(prompts)}")
        print(f"{args.prompt}")

        all_images = []

        for batch_index in range(args.n_batch):
            # if clear_between_batches and batch_index % 32 == 0:
            # display.clear_output(wait=True)
            print(f"Batch {batch_index + 1} of {args.n_batch}")

            for image in init_array:  # iterates the init images
                args.init_image = image
                results = generate(args)
                for image in results:
                    if args.make_grid:
                        all_images.append(T.functional.pil_to_tensor(image))
                    if args.save_samples:
                        if args.filename_format == "{timestring}_{index}_{prompt}.png":
                            filename = f"{args.timestring}_{index:05}_{sanitize(prompt)[:160]}.png"
                        else:
                            filename = f"{args.timestring}_{index:05}_{args.seed}.png"
                        fpath = os.path.join(args.outdir, filename)
                        image.save(fpath)
                    st.session_state['node_pipe'] = image
                    image_pipe.image(image)
                    st.session_state['currentImages'].append(fpath)
                    index += 1
                args.seed = next_seed(args)

        # print(len(all_images))
        if args.make_grid:
            grid = make_grid(all_images, nrow=int(len(all_images) / args.grid_rows))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            filename = f"{args.timestring}_{iprompt:05d}_grid_{args.seed}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))
            grid_image.save(os.path.join(args.outdir, filename))


class DeformAnimKeys:
    def __init__(self, anim_args):
        self.angle_series = get_inbetweens(parse_key_frames(anim_args.angle), anim_args.max_frames)
        self.zoom_series = get_inbetweens(parse_key_frames(anim_args.zoom), anim_args.max_frames)
        self.translation_x_series = get_inbetweens(parse_key_frames(anim_args.translation_x), anim_args.max_frames)
        self.translation_y_series = get_inbetweens(parse_key_frames(anim_args.translation_y), anim_args.max_frames)
        self.translation_z_series = get_inbetweens(parse_key_frames(anim_args.translation_z), anim_args.max_frames)
        self.perspective_flip_theta_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_theta), anim_args.max_frames)
        self.perspective_flip_phi_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_phi), anim_args.max_frames)
        self.perspective_flip_gamma_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_gamma), anim_args.max_frames)
        self.perspective_flip_fv_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_fv), anim_args.max_frames)
        self.rotation_3d_x_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_x), anim_args.max_frames)
        self.rotation_3d_y_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_y), anim_args.max_frames)
        self.rotation_3d_z_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_z), anim_args.max_frames)
        self.noise_schedule_series = get_inbetweens(parse_key_frames(anim_args.noise_schedule), anim_args.max_frames)
        self.strength_schedule_series = get_inbetweens(parse_key_frames(anim_args.strength_schedule),
                                                       anim_args.max_frames)
        self.contrast_schedule_series = get_inbetweens(parse_key_frames(anim_args.contrast_schedule),
                                                       anim_args.max_frames)

def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def parse_key_frames(string, prompt_parser=None):
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = {}
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def next_seed(args):
    if args.seed_behavior == 'iter':
        args.seed += 1
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2 ** 32 - 1)
    return args.seed

def set_model_path(set_model_path):
    global model_path
    model_path = set_model_path

def anim_frame_warp_2d(prev_img_cv2, args, anim_args, keys, frame_idx):
    angle = keys.angle_series[frame_idx]
    zoom = keys.zoom_series[frame_idx]
    translation_x = keys.translation_x_series[frame_idx]
    translation_y = keys.translation_y_series[frame_idx]

    center = (args.W // 2, args.H // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    if anim_args.flip_2d_perspective:
        perspective_flip_theta = keys.perspective_flip_theta_series[frame_idx]
        perspective_flip_phi = keys.perspective_flip_phi_series[frame_idx]
        perspective_flip_gamma = keys.perspective_flip_gamma_series[frame_idx]
        perspective_flip_fv = keys.perspective_flip_fv_series[frame_idx]
        M,sl = warpMatrix(args.W, args.H, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, 1., perspective_flip_fv);
        post_trans_mat = np.float32([[1, 0, (args.W-sl)/2], [0, 1, (args.H-sl)/2]])
        post_trans_mat = np.vstack([post_trans_mat, [0,0,1]])
        bM = np.matmul(M, post_trans_mat)
        xform = np.matmul(bM, rot_mat, trans_mat)
    else:
        xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
    )
def anim_frame_warp_2d_orig(prev_img_cv2, args, anim_args, keys, frame_idx):
    angle = keys.angle_series[frame_idx]
    zoom = keys.zoom_series[frame_idx]
    translation_x = keys.translation_x_series[frame_idx]
    translation_y = keys.translation_y_series[frame_idx]

    center = (args.W // 2, args.H // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
    )

def anim_frame_warp_3d(prev_img_cv2, depth, anim_args, keys, frame_idx):
    global device
    TRANSLATION_SCALE = 1.0 / 200.0  # matches Disco
    translate_xyz = [
        -keys.translation_x_series[frame_idx] * TRANSLATION_SCALE,
        keys.translation_y_series[frame_idx] * TRANSLATION_SCALE,
        -keys.translation_z_series[frame_idx] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(keys.rotation_3d_x_series[frame_idx]),
        math.radians(keys.rotation_3d_y_series[frame_idx]),
        math.radians(keys.rotation_3d_z_series[frame_idx])
    ]
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    result = transform_image_3d(prev_img_cv2, depth, rot_mat, translate_xyz, anim_args)
    torch.cuda.empty_cache()
    return result



def add_noise( sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    global device
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def sample_from_cv2( sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def sample_to_cv2( sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def render_animation( args, anim_args, animation_prompts, model_path, half_precision=True):

    global device

    load_models()


    # animations use key framed prompts
    #args.prompts = animation_prompts
    animation_prompts = args.prompt

    # expand key frame strings to values
    keys = DeformAnimKeys(anim_args)
    print(keys)
    # resume animation
    start_frame = 0
    if anim_args.resume_from_timestring:
        for tmp in os.listdir(args.outdir):
            if tmp.split("_")[0] == anim_args.resume_timestring:
                start_frame += 1
        start_frame = start_frame - 1

    # create output folder for the batch
    #os.makedirs(args.outdir, exist_ok=True)
    #print(f"Saving animation frames to {args.outdir}")

    image_pipe = st.session_state["preview_image"]
    video_pipe = st.session_state["preview_video"]
    video_pipe.empty()

    # save settings for the batch
    if args.save_settings:
        if st.session_state["pathmode"] == "subfolders":
            settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        else:
            settings_filename = os.path.join(args.outdir, f"_configs/{args.timestring}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)

    # resume from timestring
    if anim_args.resume_from_timestring:
        args.timestring = anim_args.resume_timestring
    if st.session_state["pathmode"] == "root":
        args.firstseed = args.seed
        args.rootoutdir = args.outdir
        args.outdir = f'{args.outdir}/_anim_stills/{args.batch_name}_{args.firstseed}'
    # expand prompts out to per-frame
    prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])

    if args.keyframes == '':
        args.keyframes = "0"
    prom = args.prompt
    key = args.keyframes

    new_prom = list(prom.split("\n"))
    new_key = list(key.split("\n"))

    prompts = dict(zip(new_key, new_prom))


    #for i in animation_prompts:
    #    prompt_series[i] = animation_prompts
    #    print(prompt_series[i])

    for i, prompt in prompts.items():
        n = int(i)
        prompt_series[n] = prompt
    prompt_series = prompt_series.ffill().bfill()

    print(prompt_series)

    # check for video inits
    using_vid_init = anim_args.animation_mode == 'Video Input'

    if using_vid_init:
        video_in_frame_path = os.path.join(args.outdir,'', 'inputframes')
        os.makedirs(os.path.join(video_in_frame_path), exist_ok=True)

        # save the video frames from input video
        print(f"Exporting Video Frames (1 every {1}) frames to {video_in_frame_path}...")
        try:
            for f in pathlib.Path(video_in_frame_path).glob('*.jpg'):
                f.unlink()
        except:
            pass
        vf = r'select=not(mod(n\,'+str(1)+'))'
        process = subprocess.Popen([
            ffmpeg, '-i', f'{st.session_state["init_path"]}',
            '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2',
            '-loglevel', 'error', '-stats',
            os.path.join(video_in_frame_path, '%04d.jpg')
        ],stdout=subprocess.PIPE, stderr=subprocess.PIPE)   #.stdout.decode('utf-8')

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)

        # determine max frames from length of input frames
        max_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.jpg')])

        use_init = True
        print(f"Loading {max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}")


    # load depth model for 3D
    predict_depths = (anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
    if predict_depths:
        if "depth_model" in st.session_state:
            print("Using depth model from cache")
            depth_model = st.session_state["depth_model"]
        else:
            st.session_state["depth_model"] = DepthModel(device)
            depth_model = st.session_state["depth_model"]
            depth_model.load_midas(model_path)
        if anim_args.midas_weight < 1.0:
            depth_model.load_adabins()
    else:
        depth_model = None
        anim_args.save_depth_maps = False

    # state for interpolating between diffusion steps
    turbo_steps = 1 if using_vid_init else int(anim_args.diffusion_cadence)
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    # resume animation
    prev_sample = None
    color_match_sample = None
    if anim_args.resume_from_timestring:
        last_frame = start_frame - 1
        if turbo_steps > 1:
            last_frame -= last_frame % turbo_steps
        path = os.path.join(args.outdir, f"{args.timestring}_{last_frame:05}.png")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prev_sample = sample_from_cv2(img)
        if anim_args.color_coherence != 'None':
            color_match_sample = img
        if turbo_steps > 1:
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(prev_sample, type=np.float32), last_frame
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            start_frame = last_frame + turbo_steps

    args.n_samples = 1
    frame_idx = start_frame
    while frame_idx < anim_args.max_frames:
        print(f"Rendering animation frame {frame_idx} of {anim_args.max_frames}")
        noise = keys.noise_schedule_series[frame_idx]
        strength = keys.strength_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        depth = None

        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(0, frame_idx - turbo_steps)
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                print(f"  creating in between frame {tween_frame_idx} tween:{tween:0.2f}")

                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx

                if depth_model is not None:
                    assert (turbo_next_image is not None)
                    depth = depth_model.predict(turbo_next_image, anim_args)

                if anim_args.animation_mode == '2D':
                    if advance_prev:
                        turbo_prev_image = anim_frame_warp_2d(turbo_prev_image, args, anim_args, keys, tween_frame_idx)
                    if advance_next:
                        turbo_next_image = anim_frame_warp_2d(turbo_next_image, args, anim_args, keys, tween_frame_idx)
                else:  # '3D'
                    if advance_prev:
                        turbo_prev_image = anim_frame_warp_3d(turbo_prev_image, depth, anim_args, keys, tween_frame_idx)
                    if advance_next:
                        turbo_next_image = anim_frame_warp_3d(turbo_next_image, depth, anim_args, keys, tween_frame_idx)
                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image * (1.0 - tween) + turbo_next_image * tween
                else:
                    img = turbo_next_image

                filename = f"{args.timestring}_{tween_frame_idx:05}.png"
                cv2.imwrite(os.path.join(args.outdir, filename), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                if anim_args.save_depth_maps:
                    depth_model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{tween_frame_idx:05}.png"),
                                     depth)
            if turbo_next_image is not None:
                prev_sample = sample_from_cv2(turbo_next_image)

        # apply transforms to previous frame
        if prev_sample is not None:
            if anim_args.animation_mode == '2D':
                prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), args, anim_args, keys, frame_idx)
            else:  # '3D'
                prev_img_cv2 = sample_to_cv2(prev_sample)
                depth = depth_model.predict(prev_img_cv2, anim_args) if depth_model else None
                prev_img = anim_frame_warp_3d(prev_img_cv2, depth, anim_args, keys, frame_idx)

            # apply color matching
            if anim_args.color_coherence != 'None':
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

            # apply scaling
            contrast_sample = prev_img * contrast
            # apply frame noising
            noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

            # use transformed previous frame as init for current
            args.use_init = True
            if half_precision:
                args.init_sample = noised_sample.half().to(device)
            else:
                args.init_sample = noised_sample.to(device)
            args.strength = max(0.0, min(1.0, strength))

        # grab prompt for current frame
        args.prompt = prompt_series[frame_idx]
        print(f"{args.prompt} {args.seed}")

        # grab init image for current frame
        if using_vid_init:
            init_frame = os.path.join(args.outdir, 'inputframes', f"{frame_idx + 1:04}.jpg")
            print(f"Using video init frame {init_frame}")
            args.init_image = init_frame

        # sample the diffusion model
        sample, image = generate(args, return_latent=False, return_sample=True)
        if not using_vid_init:
            prev_sample = sample

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
            frame_idx += turbo_steps
        else:
            filename = f"{args.timestring}_{frame_idx:05}.png"
            image.save(os.path.join(args.outdir, filename))
            if anim_args.save_depth_maps:
                if depth is None:
                    depth = depth_model.predict(sample_to_cv2(sample), anim_args)
                depth_model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{frame_idx:05}.png"), depth)
            frame_idx += 1

        image_pipe.image(image)

        #display.clear_output(wait=True)
        #display.display(image)

        args.seed = next_seed(args)


def maintain_colors( prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)



def transform_image_3d( prev_img_cv2, depth_tensor, rot_mat, translate, anim_args):
    global device
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    aspect_ratio = float(w) / float(h)
    near, far, fov_deg = anim_args.near_plane, anim_args.far_plane, anim_args.fov
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat,
                                              T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y, x = torch.meshgrid(torch.linspace(-1., 1., h, dtype=torch.float32, device=device),
                          torch.linspace(-1., 1., w, dtype=torch.float32, device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1, 1, h, w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)

    image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1 / 512 - 0.0001).unsqueeze(0),
        offset_coords_2d,
        mode=anim_args.sampling_mode,
        padding_mode=anim_args.padding_mode,
        align_corners=False
    )

    # convert back to cv2 style numpy array
    result = rearrange(
        new_image.squeeze().clamp(0, 255),
        'c h w -> h w c'
    ).cpu().numpy().astype(prev_img_cv2.dtype)
    return result


def load_mask_latent(mask_input, shape):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape

    if isinstance(mask_input, str):  # mask input is probably a file name
        if mask_input.startswith('http://') or mask_input.startswith('https://'):
            mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask


def prepare_mask(mask_input, mask_shape, invert_mask, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge,
    #     0 is black, 1 is no adjustment, >1 is brighter
    # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image,
    #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast

    mask = load_mask_latent(mask_input, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    # Mask image to array
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask)

    if invert_mask:
        mask = ((mask - 0.5) * -1) + 0.5

    mask = np.clip(mask, 0, 1)
    return mask


def load_img(path, shape, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.

    return image, mask_image

def make_callback(sampler_name, dynamic_threshold=None, static_threshold=None, mask=None, init_latent=None, sigmas=None,
                  sampler=None, masked_noise_modifier=1.0):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image at each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(args_dict):
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(args_dict['x'], -1 * static_threshold, static_threshold)
        if mask is not None:
            init_noise = init_latent + noise * args_dict['sigma']
            is_masked = torch.logical_and(mask >= mask_schedule[args_dict['i']], mask != 0)
            new_img = init_noise * torch.where(is_masked, 1, 0) + args_dict['x'] * torch.where(is_masked, 0, 1)
            args_dict['x'].copy_(new_img)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(img, i):
        global device
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1 * static_threshold, static_threshold)
        if mask is not None:
            i_inv = len(sigmas) - i - 1
            init_noise = sampler.stochastic_encode(init_latent, torch.tensor([i_inv] * batch_size).to(device),
                                                   noise=noise)
            is_masked = torch.logical_and(mask >= mask_schedule[i], mask != 0)
            new_img = init_noise * torch.where(is_masked, 1, 0) + img * torch.where(is_masked, 0, 1)
            img.copy_(new_img)

    if init_latent is not None:
        noise = torch.randn_like(init_latent, device=device) * masked_noise_modifier
    if sigmas is not None and len(sigmas) > 0:
        mask_schedule, _ = torch.sort(sigmas / torch.max(sigmas))
    elif len(sigmas) == 0:
        mask = None  # no mask needed if no steps (usually happens because strength==1.0)
    if sampler_name in ["plms", "ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        if mask is not None:
            assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"
            batch_size = init_latent.shape[0]

        callback = img_callback_
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback_

    return callback




def generate(args, return_latent=False, return_sample=False, return_c=False):
    global device
    device = st.session_state["device"]
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    #model = st.session_state["model"]

    sampler = PLMSSampler(st.session_state["model"]) if args.sampler == 'plms' else DDIMSampler(st.session_state["model"])
    model_wrap = CompVisDenoiser(st.session_state["model"])
    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            init_latent = st.session_state["model"].get_first_stage_encoding(st.session_state["model"].encode_first_stage(args.init_sample))
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = st.session_state["model"].get_first_stage_encoding(st.session_state["model"].encode_first_stage(init_image))  # move to latent space

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"

        mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                            init_latent.shape,
                            args.invert_mask,
                            args.mask_contrast_adjust,
                            args.mask_brightness_adjust)

        if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning(
                "use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")

        mask = mask.to(device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    t_enc = int((1.0 - args.strength) * args.steps)

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(args.steps)
    k_sigmas = k_sigmas[len(k_sigmas) - t_enc - 1:]

    if args.sampler in ['plms', 'ddim']:
        sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, ddim_discretize='fill', verbose=False)

    callback = make_callback(sampler_name=args.sampler,
                             dynamic_threshold=args.dynamic_threshold,
                             static_threshold=args.static_threshold,
                             mask=mask,
                             init_latent=init_latent,
                             sigmas=k_sigmas,
                             sampler=sampler)

    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with st.session_state["model"].ema_scope():
                for prompts in data:
                    uc = None
                    if args.scale != 1.0:
                        uc = st.session_state["model"].get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = st.session_state["model"].get_learned_conditioning(prompts)

                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in ["klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"]:
                        samples = sampler_fn(
                            c=c,
                            uc=uc,
                            args=args,
                            model_wrap=model_wrap,
                            init_latent=init_latent,
                            t_enc=t_enc,
                            device=device,
                            cb=callback)
                    else:
                        # args.sampler == 'plms' or args.sampler == 'ddim':
                        if init_latent is not None and args.strength > 0:
                            z_enc = sampler.stochastic_encode(init_latent,
                                                              torch.tensor([t_enc] * batch_size).to(device))
                        else:
                            z_enc = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f],
                                                device=device)
                        if args.sampler == 'ddim':
                            samples = sampler.decode(z_enc,
                                                     c,
                                                     t_enc,
                                                     unconditional_guidance_scale=args.scale,
                                                     unconditional_conditioning=uc,
                                                     img_callback=callback)
                        elif args.sampler == 'plms':  # no "decode" function in plms, so use "sample"
                            shape = [args.C, args.H // args.f, args.W // args.f]
                            samples, _ = sampler.sample(S=args.steps,
                                                        conditioning=c,
                                                        batch_size=args.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=args.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=args.ddim_eta,
                                                        x_T=z_enc,
                                                        img_callback=callback)
                        else:
                            raise Exception(f"Sampler {args.sampler} not recognised.")

                    if return_latent:
                        results.append(samples.clone())

                    x_samples = st.session_state["model"].decode_first_stage(samples)
                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        results.append(image)
    return results




from functools import reduce
def construct_RotationMatrixHomogenous(rotation_angles):
    assert(type(rotation_angles)==list and len(rotation_angles)==3)
    RH = np.eye(4,4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH

# https://en.wikipedia.org/wiki/Rotation_matrix
def getRotationMatrixManual(rotation_angles):

    rotation_angles = [np.deg2rad(x) for x in rotation_angles]

    phi = rotation_angles[0] # around x
    gamma       = rotation_angles[1] # around y
    theta       = rotation_angles[2] # around z

    # X rotation
    Rphi        = np.eye(4,4)
    sp          = np.sin(phi)
    cp          = np.cos(phi)
    Rphi[1,1]   = cp
    Rphi[2,2]   = Rphi[1,1]
    Rphi[1,2]   = -sp
    Rphi[2,1]   = sp

    # Y rotation
    Rgamma        = np.eye(4,4)
    sg            = np.sin(gamma)
    cg            = np.cos(gamma)
    Rgamma[0,0]   = cg
    Rgamma[2,2]   = Rgamma[0,0]
    Rgamma[0,2]   = sg
    Rgamma[2,0]   = -sg

    # Z rotation (in-image-plane)
    Rtheta      = np.eye(4,4)
    st          = np.sin(theta)
    ct          = np.cos(theta)
    Rtheta[0,0] = ct
    Rtheta[1,1] = Rtheta[0,0]
    Rtheta[0,1] = -st
    Rtheta[1,0] = st

    R           = reduce(lambda x,y : np.matmul(x,y), [Rphi, Rgamma, Rtheta])

    return R


def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):

    ptsIn2D      =  ptsIn[0,:]
    ptsOut2D     =  ptsOut[0,:]
    ptsOut2Dlist =  []
    ptsIn2Dlist  =  []

    for i in range(0,4):
        ptsOut2Dlist.append([ptsOut2D[i,0], ptsOut2D[i,1]])
        ptsIn2Dlist.append([ptsIn2D[i,0], ptsIn2D[i,1]])

    pin  =  np.array(ptsIn2Dlist)   +  [W/2.,H/2.]
    pout = (np.array(ptsOut2Dlist)  +  [1.,1.]) * (0.5*sidelength)
    pin  = pin.astype(np.float32)
    pout = pout.astype(np.float32)

    return pin, pout

def warpMatrix(W, H, theta, phi, gamma, scale, fV):

    # M is to be estimated
    M          = np.eye(4, 4)

    fVhalf     = np.deg2rad(fV/2.)
    d          = np.sqrt(W*W+H*H)
    sideLength = scale*d/np.cos(fVhalf)
    h          = d/(2.0*np.sin(fVhalf))
    n          = h-(d/2.0)
    f          = h+(d/2.0)

    # Translation along Z-axis by -h
    T       = np.eye(4,4)
    T[2,3]  = -h

    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([phi, gamma, theta])


    # Projection Matrix
    P       = np.eye(4,4)
    P[0,0]  = 1.0/np.tan(fVhalf)
    P[1,1]  = P[0,0]
    P[2,2]  = -(f+n)/(f-n)
    P[2,3]  = -(2.0*f*n)/(f-n)
    P[3,2]  = -1.0

    # pythonic matrix multiplication
    F       = reduce(lambda x,y : np.matmul(x,y), [P, T, R])

    # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way.
    # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
    ptsIn = np.array([[
        [-W/2., H/2., 0.],[ W/2., H/2., 0.],[ W/2.,-H/2., 0.],[-W/2.,-H/2., 0.]
    ]])
    ptsOut  = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut  = cv2.perspectiveTransform(ptsIn, F)

    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sideLength)

    # check float32 otherwise OpenCV throws an error
    assert(ptsInPt2f.dtype  == np.float32)
    assert(ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f,ptsOutPt2f)

    return M33, sideLength