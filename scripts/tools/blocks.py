from barfi import Block
from scripts.tools.deforum_runner import runner
import streamlit as st
import random
import numpy as np
import cv2
from scripts.tools.modelloader import load_models
from scripts.tools.sd_utils import img2img
from scripts.tools.sd_utils import *
from scripts.tools.nsp.nsp_pantry import parser
from scripts.tools.node_func import *

from gfpgan import GFPGANer

import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import PIL
import torch
def_runner = runner()

#Noodle Prompt emitter node

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

def img_reshape(img):
    #img = Image.open('./images/'+img).convert('RGB')
    #img = img.resize((300,300))
    img = np.asarray(img)
    return img

def image_grid(array, ncols=4):
    index, height, width, channels = array.shape
    nrows = index//ncols

    img_grid = (array.reshape(nrows, ncols, height, width, channels)
                .swapaxes(1,2)
                .reshape(height*nrows, width*ncols))

    return img_grid

def var_runner(img):
    outputimgs = variations(img, outdir='output', var_samples=4, var_plms='k_lms', v_cfg_scale=7.5, v_steps=20, v_W=512, v_H=512, v_ddim_eta=0, v_GFPGAN=False, v_bg_upsampling=False, v_upscale=1)
    return outputimgs
var_block = Block(name='Variations')
var_block.add_input(name='image')
var_block.add_output(name='variations')
def var_func(self):
    img = self.get_interface(name='image')
    outputimgs = var_runner(img)
    self.set_interface(name='variations', value=outputimgs)
var_block.add_compute(var_func)


grid_block = Block(name='grid')
grid_block.add_input(name='trigger')
grid_block.add_output('grid')
def grid_block_func(self):
    all_images = []
    for image in st.session_state["currentImages"]:
        all_images.append(T.functional.pil_to_tensor(image))
    grid = make_grid(all_images, nrow=int(len(all_images) / 4))
    grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
    grid_image = Image.fromarray(grid.astype(np.uint8))
    self.set_interface(name='grid', value=grid_image)


grid_block.add_compute(grid_block_func)

#Open Image
open_block = Block(name='open path')
open_block.add_option(name='path', type='input')
open_block.add_output(name='Image')
def open_func(self):
    image = PIL.Image.open(self.get_option(name='path'))
    self.set_interface(name='Image', value=image)
open_block.add_compute(open_func)


save_block = Block(name='save to path')
save_block.add_input(name='Image Input')
save_block.add_option(name='path', type='input', value=st.session_state['defaults'].general.outdir)
save_block.add_option(name='name', type='input', value=f'{str(random.randint(10000, 99999))}.png')
save_block.add_output(name='Image')
save_block.add_output(name='Path')
def save_func(self):
    image = self.get_interface(name='Image Input')
    path = os.path.join(self.get_option(name='path'), self.get_option(name='name'))
    image.save(path)
    self.set_interface(name='Image', value=image)
    self.set_interface(name='Path', value=path)
save_block.add_compute(save_func)

save_all_block = Block(name='save cache to path')
save_all_block.add_option(name='empty_memory', type='checkbox')
save_all_block.add_option(name='path', type='input', value=st.session_state['defaults'].general.outdir)
save_all_block.add_option(name='name', type='input', value=f'{str(random.randint(10000, 99999))}')
save_all_block.add_output(name='output_list')
def save_all_func(self):
    images = st.session_state["currentImages"]
    a = 0
    path = self.get_option(name='path')
    os.makedirs(path, exist_ok=True)
    for i in images:

        a = a + 1
        counter = f'00{a}'
        counter = counter[:3]
        name = self.get_option(name="name")
        name = f'{name}_{counter}.png'
        spath = os.path.join(path, name)
        i.save(spath)
    if self.get_option(name='empty_memory') == True and self.get_option(name='keep_temp') == True:
        st.session_state["templist"] = st.session_state["currentImages"]
        st.session_state["currentImages"] = []
        self.set_interface(name='output_list', value=st.session_state["templist"])
    elif self.get_option(name='empty_memory') == True and self.get_option(name='keep_temp') == False:
        self.set_interface(name='output_list', value=st.session_state["currentImages"])
        st.session_state["currentImages"] = []

    #path = os.path.join(self.get_option(name='path'), self.get_option(name='name'))
    #image.save(path)
    #self.set_interface(name='Image', value=image)
save_all_block.add_compute(save_all_func)

if 'currentImages' not in st.session_state:
    st.session_state['currentImages'] = []




#Get list item
list_out = Block(name='get image from cache')
list_out.add_option(name='images', type='select', items=[''])
list_out.add_option(name='index', type='integer', value=0)
list_out.add_output(name='image')
list_out.add_output(name='info')
def list_out_func(self):
    int1 = self.get_option(name='index')
    self.set_interface(name='image', value=st.session_state["currentImages"][int1])
    #self.set_interface(name='image', value='img')
list_out.add_compute(list_out_func)


#SD Custom Blocks:
#GFPGAN Block
upscale_block = Block(name='gfpgan upscale')
upscale_block.add_input(name='Input_Image')
upscale_block.add_option(name='Upscale', type='integer', value=1)
upscale_block.add_output(name='Restored_Img')
def upscale_func(self):
    upscale = int(self.get_option(name='Upscale'))
    data = self.get_interface(name='Input_Image')
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(st.session_state['defaults'].general.GFPGAN_dir,
                              model_name + '.pth')
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None
    )
    #torch_gc()

    if isinstance(data, list):
        data = data[0]
    image=np.array(data)
    input_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned=False, only_center_face=False, paste_back=True)
    image = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    gfpgan_image = PIL.Image.fromarray(image)
    self.set_interface(name='Restored_Img', value=gfpgan_image)
upscale_block.add_compute(upscale_func)

def img2img_runner():
    print(st.session_state["img2img"]["new_img"])
    print(st.session_state["img2img"]["steps"])
    print(st.session_state["img2img"]["seed"])
    print(st.session_state["img2img"]["cfg_scale"])
    print(st.session_state["img2img"]["prompt"])
    print(st.session_state["img2img"]["variant_amount"])
    print(st.session_state["img2img"]["denoising_strength"])
    output_images, seed, info, stats = img2img()
    return output_images


#img2img Block
img2img_block = Block(name='img2img node')
img2img_block.add_input(name='2ImageIn')
img2img_block.add_input(name='Var Amount')
img2img_block.add_input(name='CFG Scale')
img2img_block.add_input(name='Steps')
img2img_block.add_input(name='Sampler')
img2img_block.add_input(name='Seed')
img2img_block.add_input(name='Prompt')
img2img_block.add_option(name='seedInfo', type='display', value='Prompt:')
img2img_block.add_option(name='2Prompt', type='input', value='')
img2img_block.add_option(name='Sampler', type='select', items=["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", "k_heun", "PLMS", "DDIM"], value="k_dpm_2_a")
img2img_block.add_option(name='Variant', type='display', value='Variant Amount:')
img2img_block.add_option(name='Var Amount', type='number', value = 0.5)
img2img_block.add_option(name='CFG_Info', type='display', value='CFG Scale:')
img2img_block.add_option(name='CFG Scale', type='number', value = 7.5)
img2img_block.add_option(name='Steps_Info', type='display', value='Steps:')
img2img_block.add_option(name='steps', type='integer', value=20)
img2img_block.add_option(name='Seed info', type='display', value='Seed:')
img2img_block.add_option(name='seed', type='integer', value=-1)
img2img_block.add_output(name='Var AmountOut')
img2img_block.add_output(name='CFG ScaleOut')
img2img_block.add_output(name='StepsOut')
img2img_block.add_output(name='SamplerOut')
img2img_block.add_output(name='SeedOut')
img2img_block.add_output(name='PromptOut')
img2img_block.add_output(name='2Image')
def img2img_func(self):
    if self.get_interface(name='Var Amount') != None:
        var_amount = self.get_interface(name='Var Amount')
    else:
        var_amount = self.get_option(name='Var Amount')
    if self.get_interface(name='CFG Scale') != None:
        cfg_scale = self.get_interface(name='CFG Scale')
    else:
        cfg_scale = self.get_option(name='CFG Scale')
    if self.get_interface(name='Steps') != None:
        steps = self.get_interface(name='Steps')
    else:
        steps = self.get_option(name='steps')
    st.session_state["sampling_steps"] = steps
    if self.get_interface(name='Sampler') != None:
        samplern = self.get_interface(name='Sampler')
    else:
        samplern = self.get_option(name='Sampler')
    if self.get_interface(name='Prompt') != None:
        prompt2 = self.get_interface(name='Prompt')
    else:
        prompt2 = self.get_option(name='2Prompt')
    if self.get_interface(name='Seed') != None:
        seed = self.get_interface(name='Seed')
    else:
        seed = self.get_option(name='seed')

    init_img = self.get_interface(name='2ImageIn')
    print(init_img)
    if isinstance(init_img, list):
        init_img = init_img[0]
    print("ok")
    init_img = init_img.convert('RGBA')
    print("ok")

    output_images, seed, info, stats = img2img(prompt = prompt2,
                                               init_info = init_img,
                                               init_info_mask = None,
                                               mask_mode = 0,
                                               mask_blur_strength = 0,
                                               mask_restore = False,
                                               ddim_steps = steps,
                                               sampler_name = samplern,
                                               n_iter = 1,
                                               cfg_scale = cfg_scale,
                                               denoising_strength = 0.8,
                                               seed = seed,
                                               noise_mode = 0,
                                               find_noise_steps = 100,
                                               height = 512,
                                               width = 512,
                                               resize_mode = 0,
                                               fp="./outputs",
                                               variant_amount = var_amount, variant_seed = seed - 1, ddim_eta = 0.0,
                                               write_info_files = False, RealESRGAN_model = "RealESRGAN_x4plus_anime_6B",
                                               separate_prompts = False, normalize_prompt_weights = False,
                                               save_individual_images = True, save_grid = False, group_by_prompt = False,
                                               save_as_jpg = False, use_GFPGAN = False, use_RealESRGAN = False, loopback = True
                                               )

    self.set_interface(name='2Image', value=output_images[0])
    self.set_interface(name='Var AmountOut', value=var_amount)
    self.set_interface(name='CFG ScaleOut', value=cfg_scale)
    self.set_interface(name='StepsOut', value=steps)
    self.set_interface(name='SamplerOut', value=samplern)
    self.set_interface(name='SeedOut', value=seed)
    self.set_interface(name='PromptOut', value=prompt2)

img2img_block.add_compute(img2img_func)



#Dream Block - test
#If an input is not connected, its value is none.
dream_block = Block(name='txt2img node')
dream_block.add_input(name='PromptIn')
dream_block.add_input(name='SeedIn')
dream_block.add_input(name='CFG ScaleIn')
dream_block.add_option(name='active', type='checkbox', value=True)
dream_block.add_option(name='seedInfo', type='display', value='SEED:')
dream_block.add_option(name='Seed', type='integer', value=-1)
dream_block.add_option(name='Steps', type='integer', value=40)
dream_block.add_option(name='CFGScale', type='number', value=12.5)

dream_block.add_option(name='promptInfo', type='display', value='PROMPT:')
dream_block.add_option(name='Prompt', type='input', value='')
dream_block.add_option(name='Sampler', type='select', items=["ddim", "plms", "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"], value='klms')

dream_block.add_output(name='PromptOut')
dream_block.add_output(name='ImageOut')
dream_block.add_output(name='SeedOut')

def dream_func(self):
    if self.get_option(name='active') == True:
        st.session_state["generation_mode"] = "txt2img"
        if self.get_interface(name='PromptIn') != None:
            prompt = self.get_interface(name='PromptIn')
        else:
            prompt = self.get_option(name='Prompt')

        if self.get_interface(name='SeedIn') != None:
            seed = self.get_interface(name='SeedIn')
        else:
            seed = self.get_option(name='Seed')

        st.session_state["txt2img"]["seed"] = seed
        st.session_state["txt2img"]["prompt"] = prompt
        st.session_state["txt2img"]['sampler'] = self.get_option(name='Sampler')

        st.session_state["txt2img"]['steps'] = self.get_option(name='Steps')

        st.session_state["txt2img"]['scale'] = self.get_option(name='CFGScale')

        st.session_state["txt2img"]["keyframes"] = None
        #st.session_state["txt2img"]["iterations"] = 1
        def_runner.run_txt2img()
        self.set_interface(name='ImageOut', value=st.session_state["node_pipe"])
    self.set_interface(name='PromptOut', value=prompt)
    self.set_interface(name='SeedOut', value=seed)

dream_block.add_compute(dream_func)

#Number Input
num_block = Block(name='int')
num_block.add_output(name='number')
num_block.add_option(name='number', type='integer')
def num_func(self):
    self.set_interface(name='number', value=self.get_option(name='number'))
num_block.add_compute(num_func)



math_block = Block(name="math")
math_block.add_input(name='X')
math_block.add_input(name='Y')
math_block.add_option(name='Option', type='select', items=['+', '-', '*', '/'], value='+')
math_block.add_output(name='Result')
def math_func(self):
    op = self.get_option(name='Option')
    if op == '+':
        a = x + y
    elif op == '-':
        a = x - y
    elif op == '*':
        a = x * y
    elif op == '/':
        a = x / y
    self.set_interface(name='Result', value=a)
math_block.add_compute(math_func)


#Image Preview Block
img_preview = Block(name='image cache/preview')
img_preview.add_input(name='iimage')
img_preview.add_output(name='image_out')
def img_prev_func(self):
    st.session_state["node_preview_img_object"] = None
    if self.get_interface(name='iimage') != None:
        iimg = self.get_interface(name='iimage')
        #st.session_state["node_preview_img_object"] = iimg
        if "currentImages" not in st.session_state:
            st.session_state["currentImages"] = []
        if isinstance(iimg, list):
            for i in iimg:
                st.session_state["currentImages"].append(i)
        else:
            st.session_state["currentImages"].append(iimg)
        self.set_interface(name='image_out', value=iimg)
    #return st.session_state["node_preview_image"]
img_preview.add_compute(img_prev_func)


#PIL Blocks
#PIL.Image.effect_mandelbrot(size, extent, quality)
#

#Mandelbrot block
mandel_block = Block(name='mandelbrot')
mandel_block.add_option(name='xa', type='slider', min=-2.5, max=2.5, value=-2)
mandel_block.add_option(name='xb', type='slider', min=-2.5, max=2.5, value=1.0)
mandel_block.add_option(name='ya', type='slider', min=-2.5, max=2.5, value=-1.5)
mandel_block.add_option(name='yb', type='slider', min=-2.5, max=2.5, value=1.5)
mandel_block.add_output(name='mandel')
def mandel_func(self):
    # Mandelbrot fractal
    # FB - 201003151
    # drawing area (xa < xb and ya < yb)
    xa = self.get_option(name='xa')
    xb = self.get_option(name='xb')
    ya = self.get_option(name='ya')
    yb = self.get_option(name='yb')
    maxIt = 256 # iterations
    # image size
    imgx = 512
    imgy = 512
    mimage = PIL.Image.new("RGB", (imgx, imgy))

    for y in range(imgy):
        cy = y * (yb - ya) / (imgy - 1)  + ya
        for x in range(imgx):
            cx = x * (xb - xa) / (imgx - 1) + xa
            c = complex(cx, cy)
            z = 0
            for i in range(maxIt):
                if abs(z) > 2.0: break
                z = z * z + c
            r = i % 4 * 64
            g = i % 8 * 32
            b = i % 16 * 16
            mimage.putpixel((x, y), b * 65536 + g * 256 + r)
    self.set_interface(name='mandel', value=mimage)


mandel_block.add_compute(mandel_func)


#Random Julia fractal block
julia_block = Block(name='julia fractal')
julia_block.add_output(name='julia')
def julia_func(self):
    # Julia fractal
    # FB - 201003151
    from PIL import Image
    # drawing area (xa < xb and ya < yb)
    xa = -2.0
    xb = 1.0
    ya = -1.5
    yb = 1.5
    maxIt = 256 # iterations
    # image size
    imgx = 512
    imgy = 512
    image = PIL.Image.new("RGB", (imgx, imgy))
    # Julia set to draw
    c = complex(random.random() * 2.0 - 1.0, random.random() - 0.5)

    for y in range(imgy):
        zy = y * (yb - ya) / (imgy - 1)  + ya
        for x in range(imgx):
            zx = x * (xb - xa) / (imgx - 1) + xa
            z = complex(zx, zy)
            for i in range(maxIt):
                if abs(z) > 2.0: break
                z = z * z + c
            r = i % 4 * 64
            g = i % 8 * 32
            b = i % 16 * 16
            image.putpixel((x, y), b * 65536 + g * 256 + r)
    self.set_interface(name='julia', value=image)
julia_block.add_compute(julia_func)

#Blend Block
def blend_img(im1, im2, alpha):
    bimg = PIL.Image.blend(im1, im2, alpha)
    return bimg
blend_block = Block(name='blend')
blend_block.add_input(name='bImage_1')
blend_block.add_input(name='bImage_2')
blend_block.add_option(name='alpha', type='slider', min=0, max=1, value=0.5)
blend_block.add_output(name='blend_ImageOut')
def blend_func(self):
    im1 = self.get_interface(name='bImage_1').convert('RGB')
    im2 = self.get_interface(name='bImage_2').convert('RGB')
    alpha = self.get_option(name='alpha')
    bimg = PIL.Image.blend(im1, im2, alpha)
    self.set_interface(name='blend_ImageOut', value=bimg)
blend_block.add_compute(blend_func)

#Adjust Block

adj_block = Block(name='adjust image')
adj_block.add_input(name='image')
adj_block.add_option(name='brightness', type='slider', min=0, max=10, value=1)
adj_block.add_option(name='contrast', type='slider', min=0, max=10, value=1)
adj_block.add_option(name='sharpness', type='slider', min=0, max=10, value=1)
adj_block.add_output(name='adjImage')
def adjust_func(self):
    im = self.get_interface(name='image')
    enhancer = PIL.ImageEnhance.Contrast(im)
    factor = self.get_option(name='contrast') #gives original image
    im_output = enhancer.enhance(factor)
    enhancer = PIL.ImageEnhance.Brightness(im_output)
    factor = self.get_option(name='brightness') #gives original image
    im_output = enhancer.enhance(factor)
    enhancer = PIL.ImageEnhance.Sharpness(im_output)
    factor = self.get_option(name='sharpness') #gives original image
    im_output = enhancer.enhance(factor)
    self.set_interface(name='adjImage', value=im_output)
adj_block.add_compute(adjust_func)


#Invert Block
invert_block = Block(name='invert image')
invert_block.add_input(name='iImage_1')
invert_block.add_output(name='invert_ImageOut')
def invert_func(self):
    im1 = self.get_interface(name='iImage_1')
    iimg = PIL.ImageOps.invert(im1)
    self.set_interface(name='invert_ImageOut', value=iimg)
invert_block.add_compute(invert_func)


#Image Filter
gaussian_block = Block(name='gaussian blur')
gaussian_block.add_input(name='Input')
gaussian_block.add_option(name='Radius', type='integer', value=2)
gaussian_block.add_output(name='Output')
def gaussian_func(self):
    img = self.get_interface(name='Input')
    radius = self.get_option(name='Radius')
    #mode = self.get_option(name='Mode')
    #print(mode)
    img = img.filter(PIL.ImageFilter.GaussianBlur(radius=radius))
    self.set_interface(name='Output', value = img)
gaussian_block.add_compute(gaussian_func)


#Convert Block
imgfilter_block = Block(name='basic filters')
imgfilter_block.add_input(name='Input')
imgfilter_block.add_option(name='Mode', type='select', items=["BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "EDGE_ENHANCE_MORE", "EMBOSS", "FIND_EDGES", "SHARPEN", "SMOOTH", "SMOOTH_MORE"], value="BLUR")
imgfilter_block.add_output(name='Output')
def imgfilter_func(self):
    img = self.get_interface(name='Input')
    if self.get_option(name='Mode') == 'BLUR':
        img = img.filter(PIL.ImageFilter.BLUR)
    elif self.get_option(name='Mode') == 'CONTOUR':
        img = img.filter(PIL.ImageFilter.CONTOUR)
    elif self.get_option(name='Mode') == 'DETAIL':
        img = img.filter(PIL.ImageFilter.DETAIL)
    elif self.get_option(name='Mode') == 'EDGE_ENHANCE':
        img = img.filter(PIL.ImageFilter.EDGE_ENHANCE)
    elif self.get_option(name='Mode') == 'EDGE_ENHANCE_MORE':
        img = img.filter(PIL.ImageFilter.EDGE_ENHANCE_MORE)
    elif self.get_option(name='Mode') == 'EMBOSS':
        img = img.filter(PIL.ImageFilter.EMBOSS)
    elif self.get_option(name='Mode') == 'FIND_EDGES':
        img = img.filter(PIL.ImageFilter.FIND_EDGES)
    elif self.get_option(name='Mode') == 'SHARPEN':
        img = img.filter(PIL.ImageFilter.SHARPEN)
    elif self.get_option(name='Mode') == 'SMOOTH':
        img = img.filter(PIL.ImageFilter.SMOOTH)
    elif self.get_option(name='Mode') == 'SMOOTH_MORE':
        img = img.filter(PIL.ImageFilter.SMOOTH_MORE)
    self.set_interface(name='Output', value = img)
imgfilter_block.add_compute(imgfilter_func)


#Convert Block
convert_block = Block(name='greyscale')
convert_block.add_input(name='Input')
convert_block.add_option(name='Mode', type='select', items=["L", "P", "RGB", "RGBA", "CMYK"], value='P')
convert_block.add_output(name='Output')
def convert_func(self):
    img = self.get_interface(name='Input')
    #mode = self.get_option(name='Mode')
    #print(mode)
    img = PIL.ImageOps.grayscale(img)
    img = img.convert(self.get_option(name='Mode'))
    self.set_interface(name='Output', value=img)
convert_block.add_compute(convert_func)



#Debug Block
debug_block = Block(name='debug')
debug_block.add_input(name='Input')
debug_block.add_option(name='Test', type='display', value='init')
debug_block.add_output(name='Output')
def debug_func(self):
    data = self.get_interface(name='Input')
    self.set_interface(name='Test', value='Success')
    self.set_interface(name='Output', value=data)
    print(f'Input Type: {type(data)}')
    print(f'Input Content: {data}')
debug_block.add_compute(debug_func)


#Duplicator_Block
dup_block = Block(name='duplicate')
dup_block.add_input(name='Input')
dup_block.add_output(name='Output-1')
dup_block.add_output(name='Output-2')
def dup_func(self):
    data = self.get_interface(name='Input')
    self.set_interface(name='Output-1', value=data)
    self.set_interface(name='Output-2', value=data)
dup_block.add_compute(dup_func)



"""
#Original Blocks as demo below:
feed = Block(name='Feed')
feed.add_output()
def feed_func(self):
    self.set_interface(name='Output 1', value=4)
feed.add_compute(feed_func)

splitter = Block(name='Splitter')
splitter.add_input()
splitter.add_output()
splitter.add_output()
def splitter_func(self):
    in_1 = self.get_interface(name='Input 1')
    value = (in_1/2)
    self.set_interface(name='Output 1', value=value)
    self.set_interface(name='Output 2', value=value)
splitter.add_compute(splitter_func)

mixer = Block(name='Mixer')
mixer.add_input()
mixer.add_input()
mixer.add_output()
def mixer_func(self):
    in_1 = self.get_interface(name='Input 1')
    in_2 = self.get_interface(name='Input 2')
    value = (in_1 + in_2)
    self.set_interface(name='Output 1', value=value)
mixer.add_compute(mixer_func)

result = Block(name='Result')
result.add_input()
def result_func(self):
    in_1 = self.get_interface(name='Input 1')
result.add_compute(result_func)

textblock = Block(name='Text')
textblock.add_output()
def tx_func(self):
    self.set_interface(name='Output 1', value="This should appear")
textblock.add_compute(tx_func)

file_block = Block(name='File Selection')
file_block.add_option(name='display-option', type='display', value='Enter the path of the file to open.')
file_block.add_option(name='file-path-input', type='input')
file_block.add_output(name='File Path')
def file_block_func(self):
    file_path = self.get_option(name='file-path-input')
    self.set_interface(name='File Path', value=file_path)
file_block.add_compute(file_block_func)

import csv

select_file_block = Block(name='Select File')
select_file_block.add_option(name='display-option', type='display', value='Select the file to load data.')
select_file_block.add_option(name='select-file', type='select', items=['file_1.csv', 'file_2.csv'], value='file_1')
select_file_block.add_output(name='File Data')
def select_file_block_func(self):
    file_path = self.get_option(name='select-file')
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    self.set_interface(name='File Data', value=data[0])
select_file_block.add_compute(select_file_block_func)

load_file_block = Block(name='Load File')
load_file_block.add_option(name='display-option', type='display', value='Enter the name of the file to load its data.')
load_file_block.add_option(name='file-path-input', type='input')
load_file_block.add_output(name='File Data')
def load_file_block_func(self):
    file_path = self.get_option(name='file-path-input')
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    self.set_interface(name='File Data', value=data[0])
load_file_block.add_compute(load_file_block_func)


slider_block = Block(name='Slider')

# Add the input and output interfaces
slider_block.add_input()
slider_block.add_output()

# Add an optional display text to the block
slider_block.add_option(name='display-option', type='display', value='This is a Block with Slider option.')

# Add the interface options to the Block
slider_block.add_option(name='slider-option-1', type='slider', min=0, max=10, value=2.5)

def slider_block_func(self):
    # Implement your computation function here
    # Use the values from the input and input-options (checbox, slider, input-text..) with the
    # get_interface() and get_option() method

    # Get the value of the input interface
    input_1_value = self.get_interface(name='Input 1')

    # Get the value of the option
    slider_1_value = self.get_option(name='slider-option-1')

    # Implement your logic using the values
    # Here
    # And obtain the value to set to the output interface
    # output_1_value = ...

    # Set the value of the output interface
    output_1_value = 0
    self.set_interface(name='Output 1', value=output_1_value)

# Add the compute function to the block
slider_block.add_compute(slider_block_func)


select_block = Block(name='Select')

# Add the input and output interfaces
select_block.add_input()
select_block.add_output()

# Add an optional display text to the block
select_block.add_option(name='display-option', type='display', value='This is a Block with Select option.')

# Add the interface options to the Block
select_block.add_option(name='select-option', type='select', items=['Select A', 'Select B', 'Select C'], value='Select A')

def select_block_func(self):
    # Implement your computation function here
    # Use the values from the input and input-options (checbox, slider, input-text..) with the
    # get_interface() and get_option() method

    # Get the value of the input interface
    input_1_value = self.get_interface(name='Input 1')

    # Get the value of the option
    select_1_value = self.get_option(name='select-option-1')

    # Implement your logic using the values
    # Here
    # And obtain the value to set to the output interface
    # output_1_value = ...

    # Set the value of the output interface
    output_1_value = 0
    self.set_interface(name='Output 1', value=output_1_value)

# Add the compute function to the block
select_block.add_compute(select_block_func)



number_block = Block(name='Number')

# Add the input and output interfaces
number_block.add_input()
number_block.add_output()

# Add an optional display text to the block
number_block.add_option(name='display-option', type='display', value='This is a Block with Number option.')

# Add the interface options to the Bloc
number_block.add_option(name='number-option-1', type='number')

def number_block_func(self):
    # Implement your computation function here
    # Use the values from the input and input-options (checbox, slider, input-text..) with the
    # get_interface() and get_option() method

    # Get the value of the input interface
    input_1_value = self.get_interface(name='Input 1')

    # Get the value of the option
    number_1_value = self.get_option(name='number-option-1')

    # Implement your logic using the values
    # Here
    # And obtain the value to set to the output interface
    # output_1_value = ...

    # Set the value of the output interface
    output_1_value = 0
    self.set_interface(name='Output 1', value=output_1_value)

# Add the compute function to the block
number_block.add_compute(number_block_func)

integer_block = Block(name='Integer')

# Add the input and output interfaces
integer_block.add_input()
integer_block.add_output()

# Add an optional display text to the block
integer_block.add_option(name='display-option', type='display', value='This is a Block with Integer option.')

# Add the interface options to the Block
integer_block.add_option(name='integer-option-1', type='integer')

def integer_block_func(self):
    # Implement your computation function here
    # Use the values from the input and input-options (checbox, slider, input-text..) with the
    # get_interface() and get_option() method

    # Get the value of the input interface
    input_1_value = self.get_interface(name='Input 1')

    # Get the value of the option
    integer_1_value = self.get_option(name='integer-option-1')

    # Implement your logic using the values
    # Here
    # And obtain the value to set to the output interface
    # output_1_value = ...

    # Set the value of the output interface
    output_1_value = 0
    self.set_interface(name='Output 1', value=output_1_value)

# Add the compute function to the block
integer_block.add_compute(integer_block_func)
"""

default_blocks_category = {'generators': [dream_block, img2img_block, mandel_block, julia_block, var_block], 'image functions': [img_preview, adj_block, upscale_block, convert_block, blend_block, invert_block, gaussian_block, imgfilter_block, grid_block], 'math':[num_block, math_block], 'file':[open_block, save_block, save_all_block, list_out], 'test functions':[debug_block]}






