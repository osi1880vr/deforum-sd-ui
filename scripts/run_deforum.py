import gc, os, time, subprocess, pathlib, json
import random
import torch

from types import SimpleNamespace

from deforum import generator
from torchvision.utils import make_grid
from einops import rearrange # , repeat
from PIL import Image
import numpy as np
from datetime import datetime

print("Local Path Variables:\n")

def next_seed(args):
    if args.seed_behavior == 'iter':
        args.seed += 1
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2 ** 32 - 1)
    return args.seed

def sanitize(prompt):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tmp = ''.join(filter(whitelist.__contains__, prompt))
    return tmp.replace(' ', '_')


def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path, time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    print(f"Saving animation frames to {args.outdir}")
    os.makedirs(out_path, exist_ok=True)
    return out_path


def render_input_video(args, anim_args):
    # create a folder for the video input frames to live in
    video_in_frame_path = os.path.join(args.outdir, 'inputframes')
    os.makedirs(video_in_frame_path, exist_ok=True)

    # save the video frames from input video
    print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}...")
    try:
        for f in pathlib.Path(video_in_frame_path).glob('*.jpg'):
            f.unlink()
    except:
        pass
    vf = r'select=not(mod(n\,' + str(anim_args.extract_nth_frame) + '))'
    subprocess.run([
        'ffmpeg', '-i', f'{anim_args.video_init_path}',
        '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2',
        '-loglevel', 'error', '-stats',
        os.path.join(video_in_frame_path, '%04d.jpg')
    ], stdout=subprocess.PIPE).stdout.decode('utf-8')

    # determine max frames from length of input frames
    anim_args.max_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.jpg')])

    args.use_init = True
    print(
        f"Loading {anim_args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}")
    generator.render_animation(args, anim_args)


def render_interpolation(self, args, anim_args):
    # animations use key framed prompts
    args.prompts = animation_prompts

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
        json.dump(s, f, ensure_ascii=False, indent=4)

    # Interpolation Settings
    args.n_samples = 1
    args.seed_behavior = 'fixed'  # force fix seed at the moment bc only 1 seed is available
    prompts_c_s = []  # cache all the text embeddings

    print(f"Preparing for interpolation of the following...")

    for i, prompt in animation_prompts.items():
        args.prompt = prompt

        # sample the diffusion model
        results = generator.generate(args, return_c=True)
        c, image = results[0], results[1]
        prompts_c_s.append(c)

        # display.clear_output(wait=True)
        # display.display(image)

        args.seed = next_seed(args)

    # display.clear_output(wait=True)
    print(f"Interpolation start...")

    frame_idx = 0

    if anim_args.interpolate_key_frames:
        for i in range(len(prompts_c_s) - 1):
            dist_frames = list(animation_prompts.items())[i + 1][0] - list(animation_prompts.items())[i][0]
            if dist_frames <= 0:
                print("key frames duplicated or reversed. interpolation skipped.")
                return
            else:
                for j in range(dist_frames):
                    # interpolate the text embedding
                    prompt1_c = prompts_c_s[i]
                    prompt2_c = prompts_c_s[i + 1]
                    args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1 / dist_frames))

                    # sample the diffusion model
                    results = generator.generate(args)
                    image = results[0]

                    filename = f"{args.timestring}_{frame_idx:05}.png"
                    image.save(os.path.join(args.outdir, filename))
                    frame_idx += 1

                    # display.clear_output(wait=True)
                    # display.display(image)

                    args.seed = next_seed(args)

    else:
        for i in range(len(prompts_c_s) - 1):
            for j in range(anim_args.interpolate_x_frames + 1):
                # interpolate the text embedding
                prompt1_c = prompts_c_s[i]
                prompt2_c = prompts_c_s[i + 1]
                args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1 / (anim_args.interpolate_x_frames + 1)))

                # sample the diffusion model
                results = generator.generate(args)
                image = results[0]

                filename = f"{args.timestring}_{frame_idx:05}.png"
                image.save(os.path.join(args.outdir, filename))
                frame_idx += 1

                # display.clear_output(wait=True)
                # display.display(image)

                args.seed = next_seed(args)

    # generate the last prompt
    args.init_c = prompts_c_s[-1]
    results = generator.generate(args)
    image = results[0]
    filename = f"{args.timestring}_{frame_idx:05}.png"
    image.save(os.path.join(args.outdir, filename))

    # display.clear_output(wait=True)
    # display.display(image)
    args.seed = next_seed(args)

    # clear init_c
    args.init_c = None


def render_image_batch(args):
    args.prompts = {k: f"{v:05d}" for v, k in enumerate(prompts)}
    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_settings or args.save_samples:
        print(f"Saving to {os.path.join(args.outdir, args.timestring)}_*")

    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)

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
                results = generator.generate(args)
                for image in results:
                    if args.make_grid:
                        all_images.append(T.functional.pil_to_tensor(image))
                    if args.save_samples:
                        if args.filename_format == "{timestring}_{index}_{prompt}.png":
                            filename = f"{args.timestring}_{index:05}_{sanitize(prompt)[:160]}.png"
                        else:
                            filename = f"{args.timestring}_{index:05}_{args.seed}.png"
                        image.save(os.path.join(args.outdir, filename))
                    # if args.display_samples:
                    # display.display(image)
                    index += 1
                args.seed = next_seed(args)

        # print(len(all_images))
        if args.make_grid:
            grid = make_grid(all_images, nrow=int(len(all_images) / args.grid_rows))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            filename = f"{args.timestring}_{iprompt:05d}_grid_{args.seed}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))
            grid_image.save(os.path.join(args.outdir, filename))


def DeforumAnimArgs():
    # @markdown ####**Animation:**
    animation_mode = '3D'  # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 1000  # @param {type:"number"}
    border = 'wrap'  # @param ['wrap', 'replicate'] {type:'string'}

    # @markdown ####**Motion Parameters:**
    angle = "0:(0)"  # @param {type:"string"}
    zoom = "0:(1.04)"  # @param {type:"string"}
    translation_x = "0:(0)"  # @param {type:"string"}
    translation_y = "0:(0)"  # @param {type:"string"}
    translation_z = "0:(10)"  # @param {type:"string"}
    rotation_3d_x = "0:(0)"  # @param {type:"string"}
    rotation_3d_y = "0:(1)"  # @param {type:"string"}
    rotation_3d_z = "0:(1)"  # @param {type:"string"}
    noise_schedule = "0: (0.02)"  # @param {type:"string"}
    strength_schedule = "0: (0.65)"  # @param {type:"string"}
    contrast_schedule = "0: (1.0)"  # @param {type:"string"}

    # @markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB'  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = '1'  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

    # @markdown ####**3D Depth Warping:**
    use_depth_warping = True  # @param {type:"boolean"}
    midas_weight = 0.3  # @param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40  # @param {type:"number"}
    padding_mode = 'border'  # @param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False  # @param {type:"boolean"}

    # @markdown ####**Video Input:**
    video_init_path = '/content/video_in.mp4'  # @param {type:"string"}
    extract_nth_frame = 1  # @param {type:"number"}

    # @markdown ####**Interpolation:**
    interpolate_key_frames = False  # @param {type:"boolean"}
    interpolate_x_frames = 4  # @param {type:"number"}

    # @markdown ####**Resume Animation:**
    resume_from_timestring = False  # @param {type:"boolean"}
    resume_timestring = "20220829210106"  # @param {type:"string"}

    return locals()


def DeforumArgs():
    # @markdown **Image Settings**
    W = 256  # @param
    H = 256  # @param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    # @markdown **Sampling Settings**
    seed = -1  # @param
    sampler = 'dpm2_ancestral'  # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 50  # @param
    scale = 7  # @param
    ddim_eta = 0.0  # @param
    dynamic_threshold = None
    static_threshold = None

    # @markdown **Save & Display Settings**
    save_samples = True  # @param {type:"boolean"}
    save_settings = True  # @param {type:"boolean"}
    display_samples = True  # @param {type:"boolean"}

    # @markdown **Batch Settings**
    n_batch = 1  # @param
    batch_name = 'time'  # @param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png"  # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter"  # @param ["iter","fixed","random"]
    make_grid = False  # @param {type:"boolean"}
    grid_rows = 2  # @param
    outdir = ''

    # @markdown **Init Settings**
    use_init = False  # @param {type:"boolean"}
    strength = 0.0  # @param {type:"number"}
    strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used
    init_image = "init.jpg"  # @param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False  # @param {type:"boolean"}
    use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
    mask_file = "mask.jpg"  # @param {type:"string"}
    invert_mask = False  # @param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  # @param {type:"number"}
    mask_contrast_adjust = 1.0  # @param {type:"number"}

    n_samples = 1  # doesnt do anything
    precision = 'autocast'
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None

    return locals()


prompts = [
    "a beautiful forest by Asher Brown Durand, trending on Artstation",  # the first prompt I want
    "a beautiful portrait of a woman by Artgerm, trending on Artstation",  # the second prompt I want
    # "the third prompt I don't want it I commented it with an",
]

animation_prompts = {
    0: "year 2100 soviet cyberpunk buildings by Escher 8k hd hyperreality detailed architecture unnatural angles intertwined",
    40: "Concept art of scifi rainforest city in the negev dessert. tall glass building covered in plants. cinematic. epic framing, beautiful, highly detailed art station, behance, realistic",
    100: "Beautiful Woman dissolving into colorful liquid oil paint, wind, cinematic lighting, photo realistic, by karol bak",
    180: "lucy in the sky with diamonds, trending on Artstation",
}

output_path = "./content/output"  # @param {type:"string"}
os.makedirs(output_path, exist_ok=True)
print(f"output_path: {output_path}")

args = SimpleNamespace(**DeforumArgs())
anim_args = SimpleNamespace(**DeforumAnimArgs())

now = datetime.now()  # current date and time
batch_name = now.strftime("%H_%M_%S")
args.batch_name = batch_name
args.outdir = get_output_folder(output_path, batch_name)

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

models_path = "./content/models"  # @param {type:"string"}

# dispatch to appropriate renderer
if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
    generator.render_animation(args, anim_args, animation_prompts, models_path)
elif anim_args.animation_mode == 'Video Input':
    render_input_video(args, anim_args)
elif anim_args.animation_mode == 'Interpolation':
    render_interpolation(args, anim_args)
else:
    generator.render_image_batch(args)
