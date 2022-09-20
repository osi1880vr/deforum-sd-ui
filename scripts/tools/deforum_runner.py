import os
import time
from datetime import datetime
import sys
import random
import torch
import gc
from datetime import date
import streamlit as st


from types import SimpleNamespace

from deforum import generator
from deforum import video


class runner:

    def get_output_folder(self, output_path, batch_folder):
        out_path = os.path.join(os.getcwd(), output_path, time.strftime('%Y-%m'), str(date.today().day))
        if batch_folder != "":
            out_path = os.path.join(out_path, batch_folder)
        print(f"Saving animation frames to {out_path}")
        os.makedirs(out_path, exist_ok=True)
        return out_path

    def get_args(self):
        # SimpleNamespace = type(sys.implementation)

        W, H = map(lambda x: x - x % 64, (st.session_state['W'], st.session_state['H']))  # resize to integer multiple of 64

        now = datetime.now()  # current date and time
        batch_name = now.strftime("%H_%M_%S")
        if st.session_state["pathmode"] == "subfolders":
            out_folder = self.get_output_folder('./content/output', batch_name)
        else:
            out_folder = st.session_state["outdir"]
        if st.session_state['seed'] == '':
            st.session_state['seed'] = int(random.randrange(0, 4200000000))
        else:
            st.session_state['seed'] = int(st.session_state['seed'])
        seed = int(random.randrange(0, 4200000000))
        DeforumArgs = {#'image': st.session_state['preview_image'],
            #'video': st.session_state['preview_video'],
            'W': W,
            'H': H,
            'seed': st.session_state['seed'],  # @param
            'sampler': st.session_state['sampler'],
            # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
            'steps': st.session_state['steps'],  # @param
            'scale': st.session_state['scale'],  # @param
            'ddim_eta': st.session_state['ddim_eta'],  # @param
            'dynamic_threshold': None,
            'static_threshold': None,

            # @markdown **Save & Display Settings**
            'save_samples': st.session_state['save_samples'],  # @param {type:"boolean"}
            'save_settings': st.session_state['save_settings'],  # @param {type:"boolean"}
            'display_samples': st.session_state['display_samples'],  # @param {type:"boolean"}

            # @markdown **Batch Settings**
            'n_batch': st.session_state["iterations"],  # @param
            'batch_name': batch_name,  # @param {type:"string"}
            'filename_format': st.session_state['filename_format'],
            # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
            'seed_behavior': st.session_state['seed_behavior'],  # @param ["iter","fixed","random"]
            'make_grid': st.session_state['make_grid'],  # @param {type:"boolean"}
            'grid_rows': st.session_state['grid_rows'],  # @param
            'outdir': out_folder,

            # @markdown **Init Settings**
            'use_init': st.session_state['use_init'],  # @param {type:"boolean"}
            'strength': st.session_state['strength'],  # @param {type:"number"}
            'strength_0_no_init': st.session_state['strength_0_no_init'],
            # Set the strength to 0 automatically when no init image is used
            'init_image': st.session_state['init_image'],  # @param {type:"string"}
            # Whiter areas of the mask are areas that change more
            'use_mask': st.session_state['use_mask'],  # @param {type:"boolean"}
            'use_alpha_as_mask': st.session_state['use_alpha_as_mask'],
            # use the alpha channel of the init image as the mask
            'mask_file': st.session_state['mask_file'],  # @param {type:"string"}
            'invert_mask': st.session_state['invert_mask'],  # @param {type:"boolean"}
            # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
            'mask_brightness_adjust': st.session_state['mask_brightness_adjust'],  # @param {type:"number"}
            'mask_contrast_adjust': st.session_state['mask_contrast_adjust'],  # @param {type:"number"}

            'n_samples': st.session_state["batch_size"],
            'precision': 'autocast',
            'C': 4,
            'f': 8,


            'keyframes': st.session_state['keyframes'],
            'prompt': st.session_state['prompt'],

            'timestring': "",
            'init_latent': None,
            'init_sample': None,
            'init_c': None,
        }

        DeforumAnimArgs = {'animation_mode': st.session_state['animation_mode'],
                           # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
                           'max_frames': st.session_state['max_frames'],  # @param {type:"number"}
                           'border': st.session_state['border'],  # @param ['wrap', 'replicate'] {type:'string'}

                           # @markdown ####**Motion Parameters:**
                           'angle': st.session_state['angle'],  # @param {type:"string"}
                           'zoom': st.session_state['zoom'],  # @param {type:"string"}
                           'translation_x': st.session_state['translation_x'],  # @param {type:"string"}
                           'translation_y': st.session_state['translation_y'],  # @param {type:"string"}
                           'translation_z': st.session_state['translation_z'],  # @param {type:"string"}
                           'rotation_3d_x': st.session_state['rotation_3d_x'],  # @param {type:"string"}
                           'rotation_3d_y': st.session_state['rotation_3d_y'],  # @param {type:"string"}
                           'rotation_3d_z': st.session_state['rotation_3d_z'],  # @param {type:"string"}
                           'noise_schedule': st.session_state['noise_schedule'],  # @param {type:"string"}
                           'flip_2d_perspective': st.session_state['flip_2d_perspective'],  # @param {type:"boolean"}
                           'perspective_flip_theta': st.session_state['perspective_flip_theta'],  # @param {type:"string"}
                           'perspective_flip_phi': st.session_state['perspective_flip_phi'],  # @param {type:"string"}
                           'perspective_flip_gamma': st.session_state['perspective_flip_gamma'],  # @param {type:"string"}
                           'perspective_flip_fv': st.session_state['perspective_flip_fv'],  # @param {type:"string"}
                           'strength_schedule': st.session_state['strength_schedule'],  # @param {type:"string"}
                           'contrast_schedule': st.session_state['contrast_schedule'],  # @param {type:"string"}

                           # @markdown ####**Coherence:**
                           'color_coherence': st.session_state['color_coherence'],
                           # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
                           'diffusion_cadence': st.session_state['diffusion_cadence'],
                           # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

                           # @markdown ####**3D Depth Warping:**
                           'use_depth_warping': st.session_state['use_depth_warping'],  # @param {type:"boolean"}
                           'midas_weight': st.session_state['midas_weight'],  # @param {type:"number"}
                           'near_plane': st.session_state['near_plane'],
                           'far_plane': st.session_state['far_plane'],
                           'fov': st.session_state['fov'],  # @param {type:"number"}
                           'padding_mode': st.session_state['padding_mode'],
                           # @param ['border', 'reflection', 'zeros'] {type:'string'}
                           'sampling_mode': st.session_state['sampling_mode'],
                           # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
                           'save_depth_maps': st.session_state['save_depth_maps'],  # @param {type:"boolean"}

                           # @markdown ####**Video Input:**
                           'video_init_path': st.session_state['video_init_path'],  # @param {type:"string"}
                           'extract_nth_frame': st.session_state['extract_nth_frame'],  # @param {type:"number"}

                           # @markdown ####**Interpolation:**
                           'interpolate_key_frames': st.session_state['interpolate_key_frames'],  # @param {type:"boolean"}
                           'interpolate_x_frames': st.session_state['interpolate_x_frames'],  # @param {type:"number"}

                           # @markdown ####**Resume Animation:**
                           'resume_from_timestring': st.session_state['resume_from_timestring'],  # @param {type:"boolean"}
                           'resume_timestring': st.session_state['resume_timestring']  # @param {type:"string"}

                           }

        args = SimpleNamespace(**DeforumArgs)
        anim_args = SimpleNamespace(**DeforumAnimArgs)

        return args, anim_args

    def run_batch(self):

        args, anim_args = self.get_args()

        args.timestring = time.strftime('%Y%m%d%H%M%S')
        args.strength = max(0.0, min(1.0, args.strength))

        if args.seed == -1:
            args.seed = random.randint(0, 2 ** 32 - 1)
        if not args.use_init:
            args.init_image = None
        if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
            print(f"Init images aren't supported with PLMS yet, switching to KLMS")
            args.sampler = 'klms'
        if args.sampler != 'ddim':
            args.ddim_eta = 0

        if anim_args.animation_mode == 'None':
            anim_args.max_frames = 1
        elif anim_args.animation_mode == 'Video Input':
            args.use_init = True

        # clean up unused memory
        gc.collect()
        torch.cuda.empty_cache()

        models_path = os.path.join(os.getcwd(), 'content', 'models')

        animation_prompts = {
            0: st.session_state['prompt'],
        }

        # dispatch to appropriate renderer
        if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D' or anim_args.animation_mode == 'Video Input':
            generator.render_animation(args, anim_args, animation_prompts, models_path)



            #args.outdir = f'{args.outdir}/_anim_stills/{args.batch_name}_{args.firstseed}'
            if st.session_state["pathmode"] == "subfolders":
                image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
                mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
            else:
                image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
                os.makedirs(os.path.join(args.rootoutdir, "_mp4s"), exist_ok=True)
                mp4_path = os.path.join(args.rootoutdir, f"_mp4s/{args.timestring}_{args.firstseed}.mp4")



            max_frames = anim_args.max_frames
            st.session_state.preview_image.empty()

            video.produce_video(args, image_path, mp4_path, max_frames)

        elif anim_args.animation_mode == 'None':
            generator.render_animation(args, anim_args, animation_prompts, models_path)

            image_path = os.path.join(args.outdir, f"{args.timestring}_willsanitizeprompt.png")




    def run_txt2img(self):

        args, anim_args = self.get_args()

        args.timestring = time.strftime('%Y%m%d%H%M%S')
        args.strength = max(0.0, min(1.0, args.strength))

        if args.seed == -1:
            args.seed = random.randint(0, 2 ** 32 - 1)
        if not args.use_init:
            args.init_image = None
        if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
            print(f"Init images aren't supported with PLMS yet, switching to KLMS")
            args.sampler = 'klms'
        if args.sampler != 'ddim':
            args.ddim_eta = 0

        if anim_args.animation_mode == 'None':
            anim_args.max_frames = 1
        elif anim_args.animation_mode == 'Video Input':
            args.use_init = True

        # clean up unused memory
        gc.collect()
        torch.cuda.empty_cache()

        models_path = os.path.join(os.getcwd(), 'content', 'models')

        args.prompts = st.session_state["prompt"]

        generator.render_image_batch(args)
