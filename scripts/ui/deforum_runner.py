import os
import time
from datetime import datetime
import sys
import random
import torch
import gc
from datetime import date

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

    def get_args(self,data):
        #SimpleNamespace = type(sys.implementation)

        W, H = map(lambda x: x - x % 64, (data['W'], data['H']))  # resize to integer multiple of 64

        now = datetime.now()  # current date and time
        batch_name = now.strftime("%H_%M_%S")
        out_folder = self.get_output_folder('./content/output', batch_name)



        if data['seed'] == '':
            seed = int(random.randrange(0,4200000000))
        else:
            data['seed'] = int(data['seed'])
        seed = int(random.randrange(0,4200000000))
        DeforumArgs= {'image': data['preview_image'],
                      'W': W,
                      'H': H,
                      'seed': seed,  # @param
                      'sampler': data['sampler'],  # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
                      'steps': data['steps'],  # @param
                      'scale': data['scale'],  # @param
                      'ddim_eta': data['ddim_eta'],  # @param
                      'dynamic_threshold': None,
                      'static_threshold': None,

                      # @markdown **Save & Display Settings**
                      'save_samples': data['save_samples'],  # @param {type:"boolean"}
                      'save_settings': data['save_settings'],  # @param {type:"boolean"}
                      'display_samples': data['display_samples'],  # @param {type:"boolean"}

                      # @markdown **Batch Settings**
                      'n_batch': 1,  # @param
                      'batch_name': batch_name,  # @param {type:"string"}
                      'filename_format': data['filename_format'],  # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
                      'seed_behavior': data['seed_behavior'],  # @param ["iter","fixed","random"]
                      'make_grid': data['make_grid'],  # @param {type:"boolean"}
                      'grid_rows': data['grid_rows'] , # @param
                      'outdir': out_folder,

                      # @markdown **Init Settings**
                      'use_init': data['use_init'],  # @param {type:"boolean"}
                      'strength': data['strength'],  # @param {type:"number"}
                      'strength_0_no_init': data['strength_0_no_init'],  # Set the strength to 0 automatically when no init image is used
                      'init_image': data['init_image'],  # @param {type:"string"}
                      # Whiter areas of the mask are areas that change more
                      'use_mask': data['use_mask'],  # @param {type:"boolean"}
                      'use_alpha_as_mask': data['use_alpha_as_mask'],  # use the alpha channel of the init image as the mask
                      'mask_file': data['mask_file'],  # @param {type:"string"}
                      'invert_mask': data['invert_mask'],  # @param {type:"boolean"}
                      # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                      'mask_brightness_adjust': data['mask_brightness_adjust'],  # @param {type:"number"}
                      'mask_contrast_adjust': data['mask_contrast_adjust'],  # @param {type:"number"}

                      'n_samples': 1,  # doesnt do anything
                      'precision': 'autocast',
                      'C': 4,
                      'f': 8,

                      'prompt': "",
                      'timestring': "",
                      'init_latent': None,
                      'init_sample': None,
                      'init_c': None,
                      }

        DeforumAnimArgs = {'animation_mode': data['animation_mode'],  # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
                       'max_frames': data['max_frames'],  # @param {type:"number"}
                       'border': data['border'],  # @param ['wrap', 'replicate'] {type:'string'}

                       # @markdown ####**Motion Parameters:**
                       'angle': data['angle'],  # @param {type:"string"}
                       'zoom': data['zoom'],  # @param {type:"string"}
                       'translation_x': data['translation_x'],  # @param {type:"string"}
                       'translation_y': data['translation_y'],  # @param {type:"string"}
                       'translation_z': data['translation_z'],  # @param {type:"string"}
                       'rotation_3d_x': data['rotation_3d_x'],  # @param {type:"string"}
                       'rotation_3d_y': data['rotation_3d_y'],  # @param {type:"string"}
                       'rotation_3d_z': data['rotation_3d_z'],  # @param {type:"string"}
                       'noise_schedule': data['noise_schedule'],  # @param {type:"string"}
                       'strength_schedule': data['strength_schedule'],  # @param {type:"string"}
                       'contrast_schedule': data['contrast_schedule'],  # @param {type:"string"}

                       # @markdown ####**Coherence:**
                       'color_coherence': data['color_coherence'],  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
                       'diffusion_cadence': data['diffusion_cadence'],  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

                       # @markdown ####**3D Depth Warping:**
                       'use_depth_warping': data['use_depth_warping'],  # @param {type:"boolean"}
                       'midas_weight': data['midas_weight'],  # @param {type:"number"}
                       'near_plane': data['near_plane'],
                       'far_plane': data['far_plane'],
                       'fov': data['fov'] , # @param {type:"number"}
                       'padding_mode': data['padding_mode'],  # @param ['border', 'reflection', 'zeros'] {type:'string'}
                       'sampling_mode': data['sampling_mode'],  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
                       'save_depth_maps': data['save_depth_maps'] , # @param {type:"boolean"}

                       # @markdown ####**Video Input:**
                       'video_init_path': data['video_init_path'],  # @param {type:"string"}
                       'extract_nth_frame': data['extract_nth_frame'] , # @param {type:"number"}

                       # @markdown ####**Interpolation:**
                       'interpolate_key_frames': data['interpolate_key_frames'] , # @param {type:"boolean"}
                       'interpolate_x_frames': data['interpolate_x_frames'] , # @param {type:"number"}

                       # @markdown ####**Resume Animation:**
                       'resume_from_timestring': data['resume_from_timestring'] , # @param {type:"boolean"}
                       'resume_timestring': data['resume_timestring']  # @param {type:"string"}

                       }

        args = SimpleNamespace(**DeforumArgs)
        anim_args = SimpleNamespace(**DeforumAnimArgs)

        return args,anim_args


    def run_batch(self,data):


        args,anim_args = self.get_args(data)

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

        models_path = os.path.join(os.getcwd(),'content', 'models')

        animation_prompts = {
            0: data['prompt'],
        }

        # dispatch to appropriate renderer
        if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
            generator.render_animation(args, anim_args, animation_prompts, models_path)

            image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
            mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
            max_frames = anim_args.max_frames
            video.produce_video(image_path, mp4_path, max_frames)



