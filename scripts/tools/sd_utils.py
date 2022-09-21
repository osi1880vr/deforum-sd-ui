# base webui import and utils.
from webui_streamlit import st
import os, sys, re, random, datetime, time, math, gc

# streamlit imports


# other imports
import platform

if "Linux" in platform.platform():
	sys.path.extend([
		'/content/src/taming-transformers',
		'/content/src/clip',
		'/content/deforum-sd-ui-colab',
		'/content/src/k-diffusion',
		'/content/src/pytorch3d-lite',
		'/content/src/AdaBins',
		'/content/src/MiDaS',
		'/content/src/soup',
		'/content/src/Real-ESRGAN'
	])

import warnings
import json

import base64
import os, sys, re, random, datetime, time, math
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from scipy import integrate
import torch
from torchdiffeq import odeint
import k_diffusion as K
import math
import mimetypes
import numpy as np
import pynvml
import threading
import torch
from torch import autocast
from torchvision import transforms
import torch.nn as nn
from omegaconf import OmegaConf
import yaml
from pathlib import Path
from contextlib import nullcontext
from einops import rearrange
from ldm.util import instantiate_from_config
from retry import retry
from slugify import slugify
import skimage
import piexif
import piexif.helper
from tqdm import trange


#from scripts.tools.nsp.nsp_pantry import parser

from tools.modelloader import load_models

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# Temp imports
# terminology_database = nspterminology

# test = nsp_parse("_adj-beauty_")
# print(test)

# Note: nsp_parse() can accept dict, list and str input.
# wget -q --show-progress --no-cache 'https://raw.githubusercontent.com/WASasquatch/noodle-soup-prompts/main/nsp_pantry.py'
# Update: You do not use import nspterminology anymore. NSP will download the terminology database json on first run.
# end of imports
# ---------------------------------------------------------------------------------------------------------------

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass

# remove some annoying deprecation warnings that show every now and then.
warnings.filterwarnings("ignore", category=DeprecationWarning)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

# should and will be moved to a settings menu in the UI at some point
grid_format = [s.lower() for s in st.session_state["defaults"].general.grid_format.split(':')]
grid_lossless = False
grid_quality = 100
if grid_format[0] == 'png':
	grid_ext = 'png'
	grid_format = 'png'
elif grid_format[0] in ['jpg', 'jpeg']:
	grid_quality = int(grid_format[1]) if len(grid_format) > 1 else 100
	grid_ext = 'jpg'
	grid_format = 'jpeg'
elif grid_format[0] == 'webp':
	grid_quality = int(grid_format[1]) if len(grid_format) > 1 else 100
	grid_ext = 'webp'
	grid_format = 'webp'
	if grid_quality < 0:  # e.g. webp:-100 for lossless mode
		grid_lossless = True
		grid_quality = abs(grid_quality)

# should and will be moved to a settings menu in the UI at some point
save_format = [s.lower() for s in st.session_state["defaults"].general.save_format.split(':')]
save_lossless = False
save_quality = 100
if save_format[0] == 'png':
	save_ext = 'png'
	save_format = 'png'
elif save_format[0] in ['jpg', 'jpeg']:
	save_quality = int(save_format[1]) if len(save_format) > 1 else 100
	save_ext = 'jpg'
	save_format = 'jpeg'
elif save_format[0] == 'webp':
	save_quality = int(save_format[1]) if len(save_format) > 1 else 100
	save_ext = 'webp'
	save_format = 'webp'
	if save_quality < 0:  # e.g. webp:-100 for lossless mode
		save_lossless = True
		save_quality = abs(save_quality)

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(st.session_state["defaults"].general.gpu)


def torch_gc():
	gc.collect()
	torch.cuda.empty_cache()
	torch.cuda.ipc_collect()

"""
def img2img(prompt: str = '', init_info: any = None, init_info_mask: any = None, mask_mode: int = 0,
			mask_blur_strength: int = 3,
			mask_restore: bool = False, ddim_steps: int = 50, sampler_name: str = 'DDIM',
			n_iter: int = 1, cfg_scale: float = 7.5, denoising_strength: float = 0.8,
			seed: int = -1, noise_mode: int = 0, find_noise_steps: str = "", height: int = 512, width: int = 512,
			resize_mode: int = 0, fp=None,
			variant_amount: float = None, variant_seed: int = None, ddim_eta: float = 0.0,
			write_info_files: bool = True, RealESRGAN_model: str = "RealESRGAN_x4plus_anime_6B",
			separate_prompts: bool = False, normalize_prompt_weights: bool = True,
			save_individual_images: bool = True, save_grid: bool = True, group_by_prompt: bool = True,
			save_as_jpg: bool = True, use_GFPGAN: bool = True, use_RealESRGAN: bool = True, loopback: bool = False,
			random_seed_loopback: bool = False
			):
"""
def img2img():
	load_models()

	outpath = st.session_state['defaults'].general.outdir_img2img or st.session_state[
		'defaults'].general.outdir or "outputs/img2img-samples"

	seed = seed_to_int(st.session_state["img2img"]["seed"])

	batch_size = 1
	n_iter = st.session_state["img2img"]["batch_count"]


	if st.session_state["img2img"]["sampler_name"] == 'PLMS':
		sampler = PLMSSampler(st.session_state["model"])
	elif st.session_state["img2img"]["sampler_name"] == 'DDIM':
		sampler = DDIMSampler(st.session_state["model"])
	elif st.session_state["img2img"]["sampler_name"] == 'k_dpm_2_a':
		sampler = KDiffusionSampler(st.session_state["model"], 'dpm_2_ancestral')
	elif st.session_state["img2img"]["sampler_name"] == 'k_dpm_2':
		sampler = KDiffusionSampler(st.session_state["model"], 'dpm_2')
	elif st.session_state["img2img"]["sampler_name"] == 'k_euler_a':
		sampler = KDiffusionSampler(st.session_state["model"], 'euler_ancestral')
	elif st.session_state["img2img"]["sampler_name"] == 'k_euler':
		sampler = KDiffusionSampler(st.session_state["model"], 'euler')
	elif st.session_state["img2img"]["sampler_name"] == 'k_heun':
		sampler = KDiffusionSampler(st.session_state["model"], 'heun')
	elif st.session_state["img2img"]["sampler_name"] == 'k_lms':
		sampler = KDiffusionSampler(st.session_state["model"], 'lms')
	else:
		raise Exception("Unknown sampler: " + st.session_state["img2img"]["sampler_name"])

	def process_init_mask(init_mask: Image):
		if init_mask.mode == "RGBA":
			init_mask = init_mask.convert('RGBA')
			background = Image.new('RGBA', init_mask.size, (0, 0, 0))
			init_mask = Image.alpha_composite(background, init_mask)
			init_mask = init_mask.convert('RGB')
		return init_mask

	init_img = st.session_state["img2img"]["init_info"]
	init_mask = None
	if st.session_state["img2img"]["mask_mode"] == 0:
		if st.session_state["img2img"]["init_info_mask"]:
			init_mask = process_init_mask(st.session_state["img2img"]["init_info_mask"])
	elif st.session_state["img2img"]["mask_mode"] == 1:
		if st.session_state["img2img"]["init_info_mask"]:
			init_mask = process_init_mask(st.session_state["img2img"]["init_info_mask"])
			init_mask = ImageOps.invert(init_mask)
	elif st.session_state["img2img"]["mask_mode"] == 2:
		init_img_transparency = init_img.split()[-1].convert('L')  # .point(lambda x: 255 if x > 0 else 0, mode='1')
		init_mask = init_img_transparency
		init_mask = init_mask.convert("RGB")
		init_mask = resize_image(st.session_state["img2img"]["resize_mode"],
								 init_mask,
								 st.session_state["img2img"]["width"],
								 st.session_state["img2img"]["height"])
		init_mask = init_mask.convert("RGB")

	assert 0. <= st.session_state["img2img"]["denoising_strength"] <= 1., 'can only work with strength in [0.0, 1.0]'

	ddim_steps=st.session_state["img2img"]["steps"]

	t_enc = int(st.session_state["img2img"]["denoising_strength"] * ddim_steps)

	if init_mask is not None and (st.session_state["img2img"]["noise_mode"] == 2 or st.session_state["img2img"]["noise_mode"] == 3) and init_img is not None:
		noise_q = 0.99
		color_variation = 0.0
		mask_blend_factor = 1.0

		np_init = (np.asarray(init_img.convert("RGB")) / 255.0).astype(np.float64)  # annoyingly complex mask fixing
		np_mask_rgb = 1. - (np.asarray(ImageOps.invert(init_mask).convert("RGB")) / 255.0).astype(np.float64)
		np_mask_rgb -= np.min(np_mask_rgb)
		np_mask_rgb /= np.max(np_mask_rgb)
		np_mask_rgb = 1. - np_mask_rgb
		np_mask_rgb_hardened = 1. - (np_mask_rgb < 0.99).astype(np.float64)
		blurred = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16., channel_axis=2, truncate=32.)
		blurred2 = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16., channel_axis=2, truncate=32.)
		# np_mask_rgb_dilated = np_mask_rgb + blurred  # fixup mask todo: derive magic constants
		# np_mask_rgb = np_mask_rgb + blurred
		np_mask_rgb_dilated = np.clip((np_mask_rgb + blurred2) * 0.7071, 0., 1.)
		np_mask_rgb = np.clip((np_mask_rgb + blurred) * 0.7071, 0., 1.)

		noise_rgb = get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
		blend_mask_rgb = np.clip(np_mask_rgb_dilated, 0., 1.) ** (mask_blend_factor)
		noised = noise_rgb[:]
		blend_mask_rgb **= (2.)
		noised = np_init[:] * (1. - blend_mask_rgb) + noised * blend_mask_rgb

		np_mask_grey = np.sum(np_mask_rgb, axis=2) / 3.
		ref_mask = np_mask_grey < 1e-3

		all_mask = np.ones((st.session_state["img2img"]["height"], st.session_state["img2img"]["width"]), dtype=bool)
		noised[all_mask, :] = skimage.exposure.match_histograms(noised[all_mask, :] ** 1., noised[ref_mask, :],
																channel_axis=1)

		init_img = Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")
		st.session_state["editor_image"].image(init_img)  # debug

	def init():
		image = init_img.convert('RGB')
		image = np.array(image).astype(np.float32) / 255.0
		image = image[None].transpose(0, 3, 1, 2)
		image = torch.from_numpy(image)

		mask_channel = None
		if init_mask:
			alpha = resize_image(st.session_state["img2img"]["resize_mode"],
								 init_mask,
								 st.session_state["img2img"]["width"] // 8,
								 st.session_state["img2img"]["height"] // 8)
			mask_channel = alpha.split()[-1]

		mask = None
		if mask_channel is not None:
			mask = np.array(mask_channel).astype(np.float32) / 255.0
			mask = (1 - mask)
			mask = np.tile(mask, (4, 1, 1))
			mask = mask[None].transpose(0, 1, 2, 3)
			mask = torch.from_numpy(mask).to(st.session_state["device"])

		if st.session_state['defaults'].general.optimized:
			st.session_state.modelFS.to(st.session_state["device"])

		init_image = 2. * image - 1.
		init_image = init_image.to(st.session_state["device"])
		init_latent = (st.session_state["model"] if not st.session_state[
			'defaults'].general.optimized else st.session_state.modelFS).get_first_stage_encoding((st.session_state[
																									   "model"] if not
		st.session_state['defaults'].general.optimized else st.session_state["img2img"]["modelFS"]).encode_first_stage(
			init_image))  # move to latent space

		if st.session_state['defaults'].general.optimized:
			mem = torch.cuda.memory_allocated() / 1e6
			st.session_state.modelFS.to("cpu")
			while (torch.cuda.memory_allocated() / 1e6 >= mem):
				time.sleep(1)

		return init_latent, mask,

	def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
		t_enc_steps = t_enc
		obliterate = False
		if ddim_steps == t_enc_steps:
			t_enc_steps = t_enc_steps - 1
			obliterate = True

		if sampler_name != 'DDIM':
			x0, z_mask = init_data

			sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
			noise = x * sigmas[ddim_steps - t_enc_steps - 1]

			xi = x0 + noise

			# Obliterate masked image
			if z_mask is not None and obliterate:
				random = torch.randn(z_mask.shape, device=xi.device)
				xi = (z_mask * noise) + ((1 - z_mask) * xi)

			sigma_sched = sigmas[ddim_steps - t_enc_steps - 1:]
			model_wrap_cfg = CFGMaskedDenoiser(sampler.model_wrap)
			samples_ddim = K.sampling.__dict__[f'sample_{sampler.get_sampler_name()}'](model_wrap_cfg, xi, sigma_sched,
																					   extra_args={'cond': conditioning,
																								   'uncond': unconditional_conditioning,
																								   'cond_scale': st.session_state["img2img"]["cfg_scale"],
																								   'mask': z_mask,
																								   'x0': x0, 'xi': xi},
																					   disable=False,
																					   callback=generation_callback)
		else:

			x0, z_mask = init_data

			sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
			z_enc = sampler.stochastic_encode(x0,
											  torch.tensor([t_enc_steps] * batch_size).to(st.session_state["device"]))

			# Obliterate masked image
			if z_mask is not None and obliterate:
				random = torch.randn(z_mask.shape, device=z_enc.device)
				z_enc = (z_mask * random) + ((1 - z_mask) * z_enc)

			# decode it
			samples_ddim = sampler.decode(z_enc, conditioning, t_enc_steps,
										  unconditional_guidance_scale=st.session_state["img2img"]["cfg_scale"],
										  unconditional_conditioning=unconditional_conditioning,
										  z_mask=z_mask, x0=x0)
		return samples_ddim

	if st.session_state["img2img"]["loopback"]:
		output_images, info = None, None
		history = []
		initial_seed = None

		do_color_correction = False
		try:
			from skimage import exposure
			do_color_correction = True
		except:
			print("Install scikit-image to perform color correction on loopback")

		for i in range(st.session_state["img2img"]["batch_count"]):
			#if do_color_correction and i == 0:
			#	correction_target = cv2.cvtColor(np.asarray(init_img.copy()), cv2.COLOR_RGB2LAB)

			output_images, seed, info, stats = process_images(
				outpath=outpath,
				func_init=init,
				func_sample=sample,
				seed=seed,
				batch_size=1,
				n_iter=1,
				init_mask=init_mask,
				init_img=init_img
			)

			if initial_seed is None:
				initial_seed = seed

			input_image = init_img
			init_img = output_images[0]

			if do_color_correction and st.session_state["img2img"]["correction_target"] is not None:
				init_img = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
					cv2.cvtColor(
						np.asarray(init_img),
						cv2.COLOR_RGB2LAB
					),
					st.session_state["img2img"]["correction_target"],
					channel_axis=2
				), cv2.COLOR_LAB2RGB).astype("uint8"))
				if st.session_state["img2img"]["mask_restore"] is True and init_mask is not None:
					color_mask = init_mask.filter(ImageFilter.GaussianBlur(st.session_state['img2img']['mask_blur_strength']))
					color_mask = color_mask.convert('L')
					source_image = input_image.convert('RGB')
					target_image = init_img.convert('RGB')

					init_img = Image.composite(source_image, target_image, color_mask)

			if not st.session_state["img2img"]["random_seed_loopback"]:
				seed = seed + 1
			else:
				seed = seed_to_int(None)

			denoising_strength = max(st.session_state["img2img"]["denoising_strength"] * 0.95, 0.1)
			history.append(init_img)

		output_images = history
		seed = initial_seed

	else:
		output_images, seed, info, stats = process_images(
			outpath=outpath,
			func_init=init,
			func_sample=sample,
			seed=seed,
			batch_size=batch_size,
			init_img=init_img,
			init_mask=init_mask

		)

	del sampler

	return output_images, seed, info, stats


# used OK 20.09.22
def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")

	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model


# used OK 20.09.22
def load_sd_from_config(ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	return sd


# used OK 20.09.22
class MemUsageMonitor(threading.Thread):
	stop_flag = False
	max_usage = 0
	total = -1

	def __init__(self, name):
		threading.Thread.__init__(self)
		self.name = name

	def run(self):
		try:
			pynvml.nvmlInit()
		except:
			print(f"[{self.name}] Unable to initialize NVIDIA management. No memory stats. \n")
			return
		print(f"[{self.name}] Recording max memory usage...\n")
		# Missing context
		# handle = pynvml.nvmlDeviceGetHandleByIndex(st.session_state['defaults'].general.gpu)
		handle = pynvml.nvmlDeviceGetHandleByIndex(0)
		self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
		while not self.stop_flag:
			m = pynvml.nvmlDeviceGetMemoryInfo(handle)
			self.max_usage = max(self.max_usage, m.used)
			# print(self.max_usage)
			time.sleep(0.1)
		print(f"[{self.name}] Stopped recording.\n")
		pynvml.nvmlShutdown()

	def read(self):
		return self.max_usage, self.total

	def stop(self):
		self.stop_flag = True

	def read_and_stop(self):
		self.stop_flag = True
		return self.max_usage, self.total


# used OK 20.09.22
class CFGMaskedDenoiser(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.inner_model = model

	def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
		x_in = x
		x_in = torch.cat([x_in] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_in = torch.cat([uncond, cond])
		uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
		denoised = uncond + (cond - uncond) * cond_scale

		if mask is not None:
			assert x0 is not None
			img_orig = x0
			mask_inv = 1. - mask
			denoised = (img_orig * mask_inv) + (mask * denoised)

		return denoised


# used OK 20.09.22
class CFGDenoiser(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.inner_model = model

	def forward(self, x, sigma, uncond, cond, cond_scale):
		x_in = torch.cat([x] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_in = torch.cat([uncond, cond])
		uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
		return uncond + (cond - uncond) * cond_scale


# used OK 20.09.22
def append_zero(x):
	return torch.cat([x, x.new_zeros([1])])


# used OK 20.09.22
def append_dims(x, target_dims):
	"""Appends dimensions to the end of a tensor until it has target_dims dimensions."""
	dims_to_append = target_dims - x.ndim
	if dims_to_append < 0:
		raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
	return x[(...,) + (None,) * dims_to_append]


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
	"""Constructs the noise schedule of Karras et al. (2022)."""
	ramp = torch.linspace(0, 1, n)
	min_inv_rho = sigma_min ** (1 / rho)
	max_inv_rho = sigma_max ** (1 / rho)
	sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
	return append_zero(sigmas).to(device)


class log:
	f = lambda color: lambda string: print(color + string + "\33[0m")
	black = f("\33[30m")
	red = f("\33[31m")
	green = f("\33[32m")
	yellow = f("\33[33m")
	blue = f("\33[34m")
	megenta = f("\33[35m")
	cyan = f("\33[36m")
	white = f("\33[37m")


# if terminology_database:
#	log.green("Loaded terminology database from the pantry.")
#	print("\x1B[3mMmm. Noodle Soup.\x1B[0m")
# else:
#	log.red("Unable to load terminology database")

#
# helper fft routines that keep ortho normalization and auto-shift before and after fft
def _fft2(data):
	if data.ndim > 2:  # has channels
		out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
		for c in range(data.shape[2]):
			c_data = data[:, :, c]
			out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
			out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
	else:  # one channel
		out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
		out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
		out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

	return out_fft


def _ifft2(data):
	if data.ndim > 2:  # has channels
		out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
		for c in range(data.shape[2]):
			c_data = data[:, :, c]
			out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
			out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
	else:  # one channel
		out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
		out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
		out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

	return out_ifft


def _get_gaussian_window(width, height, std=3.14, mode=0):
	window_scale_x = float(width / min(width, height))
	window_scale_y = float(height / min(width, height))

	window = np.zeros((width, height))
	x = (np.arange(width) / width * 2. - 1.) * window_scale_x
	for y in range(height):
		fy = (y / height * 2. - 1.) * window_scale_y
		if mode == 0:
			window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
		else:
			window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (
						std / 3.14)  # hey wait a minute that's not gaussian

	return window


def _get_masked_window_rgb(np_mask_grey, hardness=1.):
	np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
	if hardness != 1.:
		hardened = np_mask_grey[:] ** hardness
	else:
		hardened = np_mask_grey[:]
	for c in range(3):
		np_mask_rgb[:, :, c] = hardened[:]
	return np_mask_rgb


def get_matched_noise(_np_src_image, np_mask_rgb, noise_q, color_variation):
	"""
	 Explanation:
	 Getting good results in/out-painting with stable diffusion can be challenging.
	 Although there are simpler effective solutions for in-painting, out-painting can be especially challenging because there is no color data
	 in the masked area to help prompt the generator. Ideally, even for in-painting we'd like work effectively without that data as well.
	 Provided here is my take on a potential solution to this problem.

	 By taking a fourier transform of the masked src img we get a function that tells us the presence and orientation of each feature scale in the unmasked src.
	 Shaping the init/seed noise for in/outpainting to the same distribution of feature scales, orientations, and positions increases output coherence
	 by helping keep features aligned. This technique is applicable to any continuous generation task such as audio or video, each of which can
	 be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased. For multi-channel data such as color
	 or stereo sound the "color tone" or histogram of the seed noise can be matched to improve quality (using scikit-image currently)
	 This method is quite robust and has the added benefit of being fast independently of the size of the out-painted area.
	 The effects of this method include things like helping the generator integrate the pre-existing view distance and camera angle.

	 Carefully managing color and brightness with histogram matching is also essential to achieving good coherence.

	 noise_q controls the exponent in the fall-off of the distribution can be any positive number, lower values means higher detail (range > 0, default 1.)
	 color_variation controls how much freedom is allowed for the colors/palette of the out-painted area (range 0..1, default 0.01)
	 This code is provided as is under the Unlicense (https://unlicense.org/)
	 Although you have no obligation to do so, if you found this code helpful please find it in your heart to credit me [parlance-zz].

	 Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
	 This code is part of a new branch of a discord bot I am working on integrating with diffusers (https://github.com/parlance-zz/g-diffuser-bot)

	"""

	global DEBUG_MODE
	global TMP_ROOT_PATH

	width = _np_src_image.shape[0]
	height = _np_src_image.shape[1]
	num_channels = _np_src_image.shape[2]

	np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
	np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
	np_src_grey = (np.sum(np_src_image, axis=2) / 3.)
	all_mask = np.ones((width, height), dtype=bool)
	img_mask = np_mask_grey > 1e-6
	ref_mask = np_mask_grey < 1e-3

	windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
	windowed_image /= np.max(windowed_image)
	windowed_image += np.average(
		_np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color
	# windowed_image += np.average(_np_src_image) * (np_mask_rgb * (1.- np_mask_rgb)) / (1.-np.average(np_mask_rgb)) # compensate for darkening across the mask transition area
	# _save_debug_img(windowed_image, "windowed_src_img")

	src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
	src_dist = np.absolute(src_fft)
	src_phase = src_fft / src_dist
	# _save_debug_img(src_dist, "windowed_src_dist")

	noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
	noise_rgb = np.random.random_sample((width, height, num_channels))
	noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
	noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
	for c in range(num_channels):
		noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

	noise_fft = _fft2(noise_rgb)
	for c in range(num_channels):
		noise_fft[:, :, c] *= noise_window
	noise_rgb = np.real(_ifft2(noise_fft))
	shaped_noise_fft = _fft2(noise_rgb)
	shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (
				src_dist ** noise_q) * src_phase  # perform the actual shaping

	brightness_variation = 0.  # color_variation # todo: temporarily tieing brightness variation to color variation for now
	contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

	# scikit-image is used for histogram matching, very convenient!
	shaped_noise = np.real(_ifft2(shaped_noise_fft))
	shaped_noise -= np.min(shaped_noise)
	shaped_noise /= np.max(shaped_noise)
	shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1.,
																  contrast_adjusted_np_src[ref_mask, :], channel_axis=1)
	shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb
	# _save_debug_img(shaped_noise, "shaped_noise")

	matched_noise = np.zeros((width, height, num_channels))
	matched_noise = shaped_noise[:]
	# matched_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
	# matched_noise = _np_src_image[:] * (1. - np_mask_rgb) + matched_noise * np_mask_rgb

	# _save_debug_img(matched_noise, "matched_noise")

	"""
	todo:
	color_variation doesnt have to be a single number, the overall color tone of the out-painted area could be param controlled
	"""

	return np.clip(matched_noise, 0., 1.)


#
def find_noise_for_image(model, device, init_image, prompt, steps=200, cond_scale=2.0, verbose=False, normalize=False,
						 generation_callback=None):
	image = np.array(init_image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	image = 2. * image - 1.
	image = image.to(device)
	x = model.get_first_stage_encoding(model.encode_first_stage(image))

	uncond = model.get_learned_conditioning([''])
	cond = model.get_learned_conditioning([prompt])

	s_in = x.new_ones([x.shape[0]])
	dnw = K.external.CompVisDenoiser(model)
	sigmas = dnw.get_sigmas(steps).flip(0)

	if verbose:
		print(sigmas)

	for i in trange(1, len(sigmas)):
		x_in = torch.cat([x] * 2)
		sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
		cond_in = torch.cat([uncond, cond])

		c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

		if i == 1:
			t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
		else:
			t = dnw.sigma_to_t(sigma_in)

		eps = model.apply_model(x_in * c_in, t, cond=cond_in)
		denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

		denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

		if i == 1:
			d = (x - denoised) / (2 * sigmas[i])
		else:
			d = (x - denoised) / sigmas[i - 1]

		if generation_callback is not None:
			generation_callback(x, i)

		dt = sigmas[i] - sigmas[i - 1]
		x = x + d * dt

	return x / sigmas[-1]


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
	"""Constructs an exponential noise schedule."""
	sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
	return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
	"""Constructs a continuous VP noise schedule."""
	t = torch.linspace(1, eps_s, n, device=device)
	sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
	return append_zero(sigmas)


def to_d(x, sigma, denoised):
	"""Converts a denoiser output to a Karras ODE derivative."""
	return (x - denoised) / append_dims(sigma, x.ndim)


def linear_multistep_coeff(order, t, i, j):
	if order - 1 > i:
		raise ValueError(f'Order {order} too high for step {i}')

	def fn(tau):
		prod = 1.
		for k in range(order):
			if j == k:
				continue
			prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
		return prod

	return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


# used OK 20.09.22
class KDiffusionSampler:
	def __init__(self, m, sampler):
		self.model = m
		self.model_wrap = K.external.CompVisDenoiser(m)
		self.schedule = sampler

	def get_sampler_name(self):
		return self.schedule

	def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale,
			   unconditional_conditioning, eta, x_T, img_callback=None, log_every_t=None):
		sigmas = self.model_wrap.get_sigmas(S)
		x = x_T * sigmas[0]
		model_wrap_cfg = CFGDenoiser(self.model_wrap)
		samples_ddim = None
		samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas,
																	  extra_args={'cond': conditioning,
																				  'uncond': unconditional_conditioning,
																				  'cond_scale': unconditional_guidance_scale},
																	  disable=False, callback=generation_callback)
		#
		return samples_ddim, None


def create_random_tensors(shape, seeds):
	xs = []
	for seed in seeds:
		torch.manual_seed(seed)

		# randn results depend on device; gpu and cpu get different results for same seed;
		# the way I see it, it's better to do this on CPU, so that everyone gets same result;
		# but the original script had it like this so i do not dare change it for now because
		# it will break everyone's seeds.
		xs.append(torch.randn(shape, device=st.session_state['defaults'].general.gpu))
	x = torch.stack(xs)
	return x


def torch_gc():
	torch.cuda.empty_cache()
	torch.cuda.ipc_collect()


def load_GFPGAN():
	model_name = 'GFPGANv1.3'
	model_path = os.path.join(st.session_state['defaults'].general.GFPGAN_dir, 'experiments/pretrained_models',
							  model_name + '.pth')
	if not os.path.isfile(model_path):
		raise Exception("GFPGAN model not found at path " + model_path)

	sys.path.append(os.path.abspath(st.session_state['defaults'].general.GFPGAN_dir))
	from gfpgan import GFPGANer

	if st.session_state['defaults'].general.gfpgan_cpu or st.session_state['defaults'].general.extra_models_cpu:
		instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None,
							device=torch.device('cpu'))
	elif st.session_state['defaults'].general.extra_models_gpu:
		instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None,
							device=torch.device(f"cuda:{st.session_state['defaults'].general.gfpgan_gpu}"))
	else:
		instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None,
							device=torch.device(f"cuda:{st.session_state['defaults'].general.gpu}"))
	return instance


def load_RealESRGAN(model_name: str):
	from basicsr.archs.rrdbnet_arch import RRDBNet
	RealESRGAN_models = {
		'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
		'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32,
											  scale=4)
	}

	model_path = os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, 'experiments/pretrained_models',
							  model_name + '.pth')
	if not os.path.exists(
			os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, "experiments", "pretrained_models",
						 f"{model_name}.pth")):
		raise Exception(model_name + ".pth not found at path " + model_path)

	sys.path.append(os.path.abspath(st.session_state['defaults'].general.RealESRGAN_dir))
	from realesrgan import RealESRGANer

	if st.session_state['defaults'].general.esrgan_cpu or st.session_state['defaults'].general.extra_models_cpu:
		instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0,
								half=False)  # cpu does not support half
		instance.device = torch.device('cpu')
		instance.model.to('cpu')
	elif st.session_state['defaults'].general.extra_models_gpu:
		instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0,
								half=not st.session_state['defaults'].general.no_half,
								device=torch.device(f"cuda:{st.session_state['defaults'].general.esrgan_gpu}"))
	else:
		instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0,
								half=not st.session_state['defaults'].general.no_half,
								device=torch.device(f"cuda:{st.session_state['defaults'].general.gpu}"))
	instance.model.name = model_name

	return instance


#
def load_LDSR(checking=False):
	model_name = 'model'
	yaml_name = 'project'
	model_path = os.path.join(st.session_state['defaults'].general.LDSR_dir, 'experiments/pretrained_models',
							  model_name + '.ckpt')
	yaml_path = os.path.join(st.session_state['defaults'].general.LDSR_dir, 'experiments/pretrained_models',
							 yaml_name + '.yaml')
	if not os.path.isfile(model_path):
		raise Exception("LDSR model not found at path " + model_path)
	if not os.path.isfile(yaml_path):
		raise Exception("LDSR model not found at path " + yaml_path)
	if checking == True:
		return True

	sys.path.append(os.path.abspath(st.session_state['defaults'].general.LDSR_dir))
	from LDSR import LDSR
	LDSRObject = LDSR(model_path, yaml_path)
	return LDSRObject


#
LDSR = None


def try_loading_LDSR(model_name: str, checking=False):
	global LDSR
	if os.path.exists(st.session_state['defaults'].general.LDSR_dir):
		try:
			LDSR = load_LDSR(checking=True)  # TODO: Should try to load both models before giving up
			if checking == True:
				print("Found LDSR")
				return True
			print("Latent Diffusion Super Sampling (LDSR) model loaded")
		except Exception:
			import traceback
			print("Error loading LDSR:", file=sys.stderr)
			print(traceback.format_exc(), file=sys.stderr)
	else:
		print("LDSR not found at path, please make sure you have cloned the LDSR repo to ./src/latent-diffusion/")


# try_loading_LDSR('model',checking=True)


# Loads Stable Diffusion model by name
def load_sd_model(model_name: str) -> [any, any, any, any, any]:
	ckpt_path = st.session_state.defaults.general.default_model_path
	if model_name != st.session_state.defaults.general.default_model:
		ckpt_path = os.path.join("models", "custom", f"{model_name}.ckpt")

	if st.session_state.defaults.general.optimized:
		config = OmegaConf.load(st.session_state.defaults.general.optimized_config)

		sd = load_sd_from_config(ckpt_path)
		li, lo = [], []
		for key, v_ in sd.items():
			sp = key.split('.')
			if (sp[0]) == 'model':
				if 'input_blocks' in sp:
					li.append(key)
				elif 'middle_block' in sp:
					li.append(key)
				elif 'time_embed' in sp:
					li.append(key)
				else:
					lo.append(key)
		for key in li:
			sd['model1.' + key[6:]] = sd.pop(key)
		for key in lo:
			sd['model2.' + key[6:]] = sd.pop(key)

		device = torch.device(f"cuda:{st.session_state.defaults.general.gpu}") \
			if torch.cuda.is_available() else torch.device("cpu")

		model = instantiate_from_config(config.modelUNet)
		_, _ = model.load_state_dict(sd, strict=False)
		model.cuda()
		model.eval()
		model.turbo = st.session_state.defaults.general.optimized_turbo

		modelCS = instantiate_from_config(config.modelCondStage)
		_, _ = modelCS.load_state_dict(sd, strict=False)
		modelCS.cond_stage_model.device = device
		modelCS.eval()

		modelFS = instantiate_from_config(config.modelFirstStage)
		_, _ = modelFS.load_state_dict(sd, strict=False)
		modelFS.eval()

		del sd

		if not st.session_state.defaults.general.no_half:
			model = model.half()
			modelCS = modelCS.half()
			modelFS = modelFS.half()

		return config, device, model, modelCS, modelFS
	else:
		config = OmegaConf.load(st.session_state.defaults.general.default_model_config)
		model = load_model_from_config(config, ckpt_path)

		device = torch.device(f"cuda:{st.session_state.defaults.general.gpu}") \
			if torch.cuda.is_available() else torch.device("cpu")
		model = (model if st.session_state.defaults.general.no_half
				 else model.half()).to(device)

		return config, device, model, None, None


# @codedealer: No usages
def ModelLoader(models, load=False, unload=False, imgproc_realesrgan_model_name='RealESRGAN_x4plus'):
	# get global variables
	global_vars = globals()
	# check if m is in globals
	if unload:
		for m in models:
			if m in global_vars:
				# if it is, delete it
				del global_vars[m]
				if st.session_state['defaults'].general.optimized:
					if m == 'model':
						del global_vars[m + 'FS']
						del global_vars[m + 'CS']
				if m == 'model':
					m = 'Stable Diffusion'
				print('Unloaded ' + m)
	if load:
		for m in models:
			if m not in global_vars or m in global_vars and type(global_vars[m]) == bool:
				# if it isn't, load it
				if m == 'GFPGAN':
					global_vars[m] = load_GFPGAN()
				elif m == 'model':
					sdLoader = load_sd_from_config()
					global_vars[m] = sdLoader[0]
					if st.session_state['defaults'].general.optimized:
						global_vars[m + 'CS'] = sdLoader[1]
						global_vars[m + 'FS'] = sdLoader[2]
				elif m == 'RealESRGAN':
					global_vars[m] = load_RealESRGAN(imgproc_realesrgan_model_name)
				elif m == 'LDSR':
					global_vars[m] = load_LDSR()
				if m == 'model':
					m = 'Stable Diffusion'
				print('Loaded ' + m)
	torch_gc()


#
@retry(tries=5)
def generation_callback(img, i=0):
	try:
		if i == 0:
			if img['i']: i = img['i']
	except TypeError:
		pass

	if i % int(st.session_state.update_preview_frequency) == 0 and st.session_state.update_preview:
		# print (img)
		# print (type(img))
		# The following lines will convert the tensor we got on img to an actual image we can render on the UI.
		# It can probably be done in a better way for someone who knows what they're doing. I don't.
		# print (img,isinstance(img, torch.Tensor))
		if isinstance(img, torch.Tensor):
			x_samples_ddim = (st.session_state["model"] if not st.session_state[
				'defaults'].general.optimized else st.session_state.modelFS).decode_first_stage(img)
		else:
			# When using the k Diffusion samplers they return a dict instead of a tensor that look like this:
			# {'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised}
			x_samples_ddim = (st.session_state["model"] if not st.session_state[
				'defaults'].general.optimized else st.session_state.modelFS).decode_first_stage(img["denoised"])

		x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

		pil_image = transforms.ToPILImage()(x_samples_ddim.squeeze_(0))

		# update image on the UI so we can see the progress
		st.session_state[st.session_state["generation_mode"]]["preview_image"].image(pil_image)

	# Show a progress bar so we can keep track of the progress even when the image progress is not been shown,
	# Dont worry, it doesnt affect the performance.


	if st.session_state["generation_mode"] == "txt2img":
		percent = int(
			100 * float(i + 1 if i + 1 < st.session_state.sampling_steps else st.session_state.sampling_steps) / float(
				st.session_state.sampling_steps))
		st.session_state[st.session_state["generation_mode"]]["progress_bar_text"].text(
			f"Running step: {i + 1 if i + 1 < st.session_state.sampling_steps else st.session_state.sampling_steps}/{st.session_state.sampling_steps} {percent if percent < 100 else 100}%")
	else:
		if st.session_state["generation_mode"] == "img2img":
			round_sampling_steps = round(st.session_state.sampling_steps * st.session_state["denoising_strength"])
			percent = int(100 * float(i + 1 if i + 1 < round_sampling_steps else round_sampling_steps) / float(
				round_sampling_steps))
			st.session_state[st.session_state["generation_mode"]]["progress_bar_text"].text(
				f"""Running step: {i + 1 if i + 1 < round_sampling_steps else round_sampling_steps}/{round_sampling_steps} {percent if percent < 100 else 100}%""")
		else:
			if st.session_state["generation_mode"] == "txt2vid":
				percent = int(100 * float(
					i + 1 if i + 1 < st.session_state.sampling_steps else st.session_state.sampling_steps) / float(
					st.session_state.sampling_steps))
				st.session_state[st.session_state["generation_mode"]]["progress_bar_text"].text(
					f"Running step: {i + 1 if i + 1 < st.session_state.sampling_steps else st.session_state.sampling_steps}/{st.session_state.sampling_steps}"
					f"{percent if percent < 100 else 100}%")

	st.session_state[st.session_state["generation_mode"]]["progress_bar"].progress(percent if percent < 100 else 100)


prompt_parser = re.compile("""
    (?P<prompt>                # capture group for 'prompt'
    [^:]+                      # match one or more non ':' characters
    )                          # end 'prompt'
    (?:                        # non-capture group
    :+                         # match one or more ':' characters
    (?P<weight>                # capture group for 'weight'
    -?\\d+(?:\\.\\d+)?            # match positive or negative decimal number
    )?                         # end weight capture group, make optional
    \\s*                        # strip spaces after weight
    |                          # OR
    $                          # else, if no ':' then match end of line
    )                          # end non-capture group
""", re.VERBOSE)


# grabs all text up to the first occurrence of ':' as sub-prompt
# takes the value following ':' as weight
# if ':' has no value defined, defaults to 1.0
# repeats until no text remaining
def split_weighted_subprompts(input_string, normalize=True):
	parsed_prompts = [(match.group("prompt"), float(match.group("weight") or 1)) for match in
					  re.finditer(prompt_parser, input_string)]
	if not normalize:
		return parsed_prompts
	# this probably still doesn't handle negative weights very well
	weight_sum = sum(map(lambda x: x[1], parsed_prompts))
	return [(x[0], x[1] / weight_sum) for x in parsed_prompts]


def slerp(device, t, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD=0.9995):
	v0 = v0.detach().cpu().numpy()
	v1 = v1.detach().cpu().numpy()

	dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
	if np.abs(dot) > DOT_THRESHOLD:
		v2 = (1 - t) * v0 + t * v1
	else:
		theta_0 = np.arccos(dot)
		sin_theta_0 = np.sin(theta_0)
		theta_t = theta_0 * t
		sin_theta_t = np.sin(theta_t)
		s0 = np.sin(theta_0 - theta_t) / sin_theta_0
		s1 = sin_theta_t / sin_theta_0
		v2 = s0 * v0 + s1 * v1

	v2 = torch.from_numpy(v2).to(device)

	return v2


#
def optimize_update_preview_frequency(current_chunk_speed, previous_chunk_speed_list, update_preview_frequency,
									  update_preview_frequency_list):
	"""Find the optimal update_preview_frequency value maximizing
	performance while minimizing the time between updates."""
	from statistics import mean

	previous_chunk_avg_speed = mean(previous_chunk_speed_list)

	previous_chunk_speed_list.append(current_chunk_speed)
	current_chunk_avg_speed = mean(previous_chunk_speed_list)

	if current_chunk_avg_speed >= previous_chunk_avg_speed:
		# print(f"{current_chunk_speed} >= {previous_chunk_speed}")
		update_preview_frequency_list.append(update_preview_frequency + 1)
	else:
		# print(f"{current_chunk_speed} <= {previous_chunk_speed}")
		update_preview_frequency_list.append(update_preview_frequency - 1)

	update_preview_frequency = round(mean(update_preview_frequency_list))

	return current_chunk_speed, previous_chunk_speed_list, update_preview_frequency, update_preview_frequency_list


def get_font(fontsize):
	fonts = ["arial.ttf", "DejaVuSans.ttf"]
	for font_name in fonts:
		try:
			return ImageFont.truetype(font_name, fontsize)
		except OSError:
			pass

	# ImageFont.load_default() is practically unusable as it only supports
	# latin1, so raise an exception instead if no usable font was found
	raise Exception(f"No usable font found (tried {', '.join(fonts)})")


def load_embeddings(fp):
	if fp is not None and hasattr(st.session_state["model"], "embedding_manager"):
		st.session_state["model"].embedding_manager.load(fp['name'])


def image_grid(imgs, batch_size, force_n_rows=None, captions=None):
	# print (len(imgs))
	if force_n_rows is not None:
		rows = force_n_rows
	elif st.session_state['defaults'].general.n_rows > 0:
		rows = st.session_state['defaults'].general.n_rows
	elif st.session_state['defaults'].general.n_rows == 0:
		rows = batch_size
	else:
		rows = math.sqrt(len(imgs))
		rows = round(rows)

	cols = math.ceil(len(imgs) / rows)

	w, h = imgs[0].size
	grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

	fnt = get_font(30)

	for i, img in enumerate(imgs):
		grid.paste(img, box=(i % cols * w, i // cols * h))
		if captions and i < len(captions):
			d = ImageDraw.Draw(grid)
			size = d.textbbox((0, 0), captions[i], font=fnt, stroke_width=2, align="center")
			d.multiline_text((i % cols * w + w / 2, i // cols * h + h - size[3]), captions[i], font=fnt,
							 fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), anchor="mm", align="center")

	return grid


def seed_to_int(s):
	if type(s) is int:
		return s
	if s is None or s == '':
		return random.randint(0, 2 ** 32 - 1)

	if type(s) is list:
		seed_list = []
		for seed in s:
			if seed is None or seed == '':
				seed_list.append(random.randint(0, 2 ** 32 - 1))
			else:
				seed_list = s

		return seed_list

	n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2 ** 32 - 1))
	while n >= 2 ** 32:
		n = n >> 32
	return n


#
def draw_prompt_matrix(im, width, height, all_prompts):
	def wrap(text, d, font, line_length):
		lines = ['']
		for word in text.split():
			line = f'{lines[-1]} {word}'.strip()
			if d.textlength(line, font=font) <= line_length:
				lines[-1] = line
			else:
				lines.append(word)
		return '\n'.join(lines)

	def draw_texts(pos, x, y, texts, sizes):
		for i, (text, size) in enumerate(zip(texts, sizes)):
			active = pos & (1 << i) != 0

			if not active:
				text = '\u0336'.join(text) + '\u0336'

			d.multiline_text((x, y + size[1] / 2), text, font=fnt, fill=color_active if active else color_inactive,
							 anchor="mm", align="center")

			y += size[1] + line_spacing

	fontsize = (width + height) // 25
	line_spacing = fontsize // 2
	fnt = get_font(fontsize)
	color_active = (0, 0, 0)
	color_inactive = (153, 153, 153)

	pad_top = height // 4
	pad_left = width * 3 // 4 if len(all_prompts) > 2 else 0

	cols = im.width // width
	rows = im.height // height

	prompts = all_prompts[1:]

	result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
	result.paste(im, (pad_left, pad_top))

	d = ImageDraw.Draw(result)

	boundary = math.ceil(len(prompts) / 2)
	prompts_horiz = [wrap(x, d, fnt, width) for x in prompts[:boundary]]
	prompts_vert = [wrap(x, d, fnt, pad_left) for x in prompts[boundary:]]

	sizes_hor = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_horiz]]
	sizes_ver = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_vert]]
	hor_text_height = sum([x[1] + line_spacing for x in sizes_hor]) - line_spacing
	ver_text_height = sum([x[1] + line_spacing for x in sizes_ver]) - line_spacing

	for col in range(cols):
		x = pad_left + width * col + width / 2
		y = pad_top / 2 - hor_text_height / 2

		draw_texts(col, x, y, prompts_horiz, sizes_hor)

	for row in range(rows):
		x = pad_left / 2
		y = pad_top + height * row + height / 2 - ver_text_height / 2

		draw_texts(row, x, y, prompts_vert, sizes_ver)

	return result


def check_prompt_length(prompt, comments):
	"""this function tests if prompt is too long, and if so, adds a message to comments"""

	tokenizer = (st.session_state["model"] if not st.session_state[
		'defaults'].general.optimized else st.session_state.modelCS).cond_stage_model.tokenizer
	max_length = (st.session_state["model"] if not st.session_state[
		'defaults'].general.optimized else st.session_state.modelCS).cond_stage_model.max_length

	info = (st.session_state["model"] if not st.session_state[
		'defaults'].general.optimized else st.session_state.modelCS).cond_stage_model.tokenizer([prompt],
																								truncation=True,
																								max_length=max_length,
																								return_overflowing_tokens=True,
																								padding="max_length",
																								return_tensors="pt")
	ovf = info['overflowing_tokens'][0]
	overflowing_count = ovf.shape[0]
	if overflowing_count == 0:
		return

	vocab = {v: k for k, v in tokenizer.get_vocab().items()}
	overflowing_words = [vocab.get(int(x), "") for x in ovf]
	overflowing_text = tokenizer.convert_tokens_to_string(''.join(overflowing_words))

	comments.append(
		f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

"""
def save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
				normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback,
				uses_random_seed_loopback,
				save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode,
				save_individual_images):"""
def save_sample(image, i):

	filename_i = os.path.join(st.session_state[st.session_state['generation_mode']]['sample_path_i'], st.session_state[st.session_state['generation_mode']]['filename'])

	if st.session_state['defaults'].general.save_metadata or st.session_state[st.session_state['generation_mode']]['write_info_files']:
		# toggles differ for txt2img vs. img2img:
		st.session_state[st.session_state['generation_mode']]['offset'] = 0 if st.session_state[st.session_state['generation_mode']]['init_img'] is None else 2
		toggles = []
		if st.session_state[st.session_state['generation_mode']]['prompt_matrix']:
			toggles.append(0)
		if st.session_state[st.session_state['generation_mode']]['normalize_prompt_weights']:
			toggles.append(1)
		if st.session_state[st.session_state['generation_mode']]['init_img'] is not None:
			if st.session_state[st.session_state['generation_mode']]['uses_loopback']:
				toggles.append(2)
			if st.session_state[st.session_state['generation_mode']]['uses_random_seed_loopback']:
				toggles.append(3)
		if st.session_state[st.session_state['generation_mode']]['save_individual_images']:
			toggles.append(2 + st.session_state[st.session_state['generation_mode']]['offset'])
		if st.session_state[st.session_state['generation_mode']]['save_grid']:
			toggles.append(3 + st.session_state[st.session_state['generation_mode']]['offset'])
		if st.session_state[st.session_state['generation_mode']]['sort_samples']:
			toggles.append(4 + st.session_state[st.session_state['generation_mode']]['offset'])
		if st.session_state[st.session_state['generation_mode']]['write_info_files']:
			toggles.append(5 + st.session_state[st.session_state['generation_mode']]['offset'])
		if st.session_state[st.session_state['generation_mode']]['use_GFPGAN']:
			toggles.append(6 + st.session_state[st.session_state['generation_mode']]['offset'])
		metadata = \
			dict(
				target="txt2img" if st.session_state[st.session_state['generation_mode']]['init_img'] is None else "img2img",
				prompt=st.session_state[st.session_state['generation_mode']]['prompts'][i],
				ddim_steps=st.session_state[st.session_state['generation_mode']]['steps'],
				toggles=toggles,
				sampler_name=st.session_state[st.session_state['generation_mode']]['sampler_name'],
				ddim_eta=st.session_state[st.session_state['generation_mode']]['ddim_eta'],
				n_iter=st.session_state[st.session_state['generation_mode']]['n_iter'],
				batch_size=st.session_state[st.session_state['generation_mode']]['batch_size'],
				cfg_scale=st.session_state[st.session_state['generation_mode']]['cfg_scale'],
				seed=st.session_state[st.session_state['generation_mode']]['seeds'][i],
				width=st.session_state[st.session_state['generation_mode']]['width'],
				height=st.session_state[st.session_state['generation_mode']]['height'],
				normalize_prompt_weights=st.session_state[st.session_state['generation_mode']]['normalize_prompt_weights'])
		# Not yet any use for these, but they bloat up the files:
		# info_dict["init_img"] = init_img
		# info_dict["init_mask"] = init_mask
		if st.session_state[st.session_state['generation_mode']]['init_img'] is not None:
			metadata["denoising_strength"] = str(st.session_state[st.session_state['generation_mode']]['denoising_strength'])
			metadata["resize_mode"] = st.session_state[st.session_state['generation_mode']]['resize_mode']

	if st.session_state[st.session_state['generation_mode']]['write_info_files']:
		with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
			yaml.dump(metadata, f, allow_unicode=True, width=10000)

	if st.session_state['defaults'].general.save_metadata:
		# metadata = {
		# 	"SD:prompt": prompts[i],
		# 	"SD:seed": str(seeds[i]),
		# 	"SD:width": str(width),
		# 	"SD:height": str(height),
		# 	"SD:steps": str(steps),
		# 	"SD:cfg_scale": str(cfg_scale),
		# 	"SD:normalize_prompt_weights": str(normalize_prompt_weights),
		# }
		metadata = {"SD:" + k: v for (k, v) in metadata.items()}

		if save_ext == "png":
			mdata = PngInfo()
			for key in metadata:
				mdata.add_text(key, str(metadata[key]))
			image.save(f"{filename_i}.png", pnginfo=mdata)
		else:
			if jpg_sample:
				image.save(f"{filename_i}.jpg", quality=save_quality,
						   optimize=True)
			elif save_ext == "webp":
				image.save(f"{filename_i}.{save_ext}", f"webp", quality=save_quality,
						   lossless=save_lossless)
			else:
				# not sure what file format this is
				image.save(f"{filename_i}.{save_ext}", f"{save_ext}")
			try:
				exif_dict = piexif.load(f"{filename_i}.{save_ext}")
			except:
				exif_dict = {"Exif": dict()}
			exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
				json.dumps(metadata), encoding="unicode")
			piexif.insert(piexif.dump(exif_dict), f"{filename_i}.{save_ext}")

	# render the image on the frontend
	st.session_state["preview_image"].image(image)


def get_next_sequence_number(path, prefix=''):
	"""
	Determines and returns the next sequence number to use when saving an
	image in the specified directory.

	If a prefix is given, only consider files whose names start with that
	prefix, and strip the prefix from filenames before extracting their
	sequence number.

	The sequence starts at 0.
	"""
	result = -1
	for p in Path(path).iterdir():
		if p.name.endswith(('.png', '.jpg')) and p.name.startswith(prefix):
			tmp = p.name[len(prefix):]
			try:
				result = max(int(tmp.split('-')[0]), result)
			except ValueError:
				pass
	return result + 1


def oxlamon_matrix(prompt, seed, n_iter, batch_size):
	pattern = re.compile(r'(,\s){2,}')

	class PromptItem:
		def __init__(self, text, parts, item):
			self.text = text
			self.parts = parts
			if item:
				self.parts.append(item)

	def clean(txt):
		return re.sub(pattern, ', ', txt)

	def getrowcount(txt):
		for data in re.finditer(".*?\\((.*?)\\).*", txt):
			if data:
				return len(data.group(1).split("|"))
			break
		return None

	def repliter(txt):
		for data in re.finditer(".*?\\((.*?)\\).*", txt):
			if data:
				r = data.span(1)
				for item in data.group(1).split("|"):
					yield (clean(txt[:r[0] - 1] + item.strip() + txt[r[1] + 1:]), item.strip())
			break

	def iterlist(items):
		outitems = []
		for item in items:
			for newitem, newpart in repliter(item.text):
				outitems.append(PromptItem(newitem, item.parts.copy(), newpart))

		return outitems

	def getmatrix(prompt):
		dataitems = [PromptItem(prompt[1:].strip(), [], None)]
		while True:
			newdataitems = iterlist(dataitems)
			if len(newdataitems) == 0:
				return dataitems
			dataitems = newdataitems

	def classToArrays(items, seed, n_iter):
		texts = []
		parts = []
		seeds = []

		for item in items:
			itemseed = seed
			for i in range(n_iter):
				texts.append(item.text)
				parts.append(f"Seed: {itemseed}\n" + "\n".join(item.parts))
				seeds.append(itemseed)
				itemseed += 1

		return seeds, texts, parts

	all_seeds, all_prompts, prompt_matrix_parts = classToArrays(getmatrix(prompt), seed, n_iter)
	n_iter = math.ceil(len(all_prompts) / batch_size)

	needrows = getrowcount(prompt)
	if needrows:
		xrows = math.sqrt(len(all_prompts))
		xrows = round(xrows)
		# if columns is to much
		cols = math.ceil(len(all_prompts) / xrows)
		if cols > needrows * 4:
			needrows *= 2

	return all_seeds, n_iter, prompt_matrix_parts, all_prompts, needrows


# used OK 20.09.22
"""
def process_images(
		outpath, func_init, func_sample, prompt, seed, sampler_name, save_grid, batch_size,
		n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
		fp=None, ddim_eta=0.0, normalize_prompt_weights=True, init_img=None, init_mask=None,
		mask_blur_strength=3, mask_restore=False, denoising_strength=0.75, noise_mode=0, find_noise_steps=1,
		resize_mode=None, uses_loopback=False,
		uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, jpg_sample=False,
		variant_amount=0.0, variant_seed=None, save_individual_images: bool = True):
"""


def process_images(
		outpath,
		func_init,
		func_sample,
		seed,
		batch_size,
		init_img,
		init_mask,
		n_iter
):
	"""this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
	assert st.session_state[st.session_state["generation_mode"]]['prompt'] is not None
	torch_gc()
	# start time after garbage collection (or before?)
	start_time = time.time()

	# We will use this date here later for the folder name, need to start_time if not need
	run_start_dt = datetime.datetime.now()

	mem_mon = MemUsageMonitor('MemMon')
	mem_mon.start()

	if hasattr(st.session_state["model"], "embedding_manager"):
		load_embeddings(st.session_state[st.session_state["generation_mode"]]['fp'])

	os.makedirs(outpath, exist_ok=True)

	sample_path = os.path.join(outpath, "samples")
	os.makedirs(sample_path, exist_ok=True)

	prompt = st.session_state[st.session_state["generation_mode"]]['prompt']
	if not ("|" in st.session_state[st.session_state["generation_mode"]]['prompt']) and st.session_state[st.session_state["generation_mode"]]['prompt'].startswith("@"):
		prompt = st.session_state[st.session_state["generation_mode"]]['prompt'][1:]

	comments = []

	prompt_matrix_parts = []
	simple_templating = False
	add_original_image = not (st.session_state[st.session_state["generation_mode"]]['use_RealESRGAN']
							  or st.session_state[st.session_state["generation_mode"]]['use_GFPGAN'])

	if st.session_state[st.session_state["generation_mode"]]['prompt_matrix']:
		if prompt.startswith("@"):
			simple_templating = True
			add_original_image = not (st.session_state[st.session_state["generation_mode"]]['use_RealESRGAN']
									  or st.session_state[st.session_state["generation_mode"]]['use_GFPGAN'])
			all_seeds, n_iter, prompt_matrix_parts, all_prompts, frows = oxlamon_matrix(prompt,
																						seed,
																						n_iter,
																						batch_size)
		else:
			all_prompts = []
			prompt_matrix_parts = prompt.split("|")
			combination_count = 2 ** (len(prompt_matrix_parts) - 1)
			for combination_num in range(combination_count):
				current = prompt_matrix_parts[0]

				for n, text in enumerate(prompt_matrix_parts[1:]):
					if combination_num & (2 ** n) > 0:
						current += ("" if text.strip().startswith(",") else ", ") + text

				all_prompts.append(current)

			n_iter = math.ceil(len(all_prompts) / batch_size)
			all_seeds = len(all_prompts) * [seed]

		print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
	else:

		if not st.session_state['defaults'].general.no_verify_input:
			try:
				check_prompt_length(prompt, comments)
			except:
				import traceback
				print("Error verifying input:", file=sys.stderr)
				print(traceback.format_exc(), file=sys.stderr)

		all_prompts = batch_size * n_iter * [prompt]
		all_seeds = [seed + x for x in range(len(all_prompts))]

	precision_scope = autocast if st.session_state['defaults'].general.precision == "autocast" else nullcontext
	output_images = []
	grid_captions = []
	stats = []
	with torch.no_grad(), precision_scope("cuda"), (
			st.session_state["model"].ema_scope() if not st.session_state['defaults'].general.optimized else nullcontext()):
		init_data = func_init()
		tic = time.time()

		# if variant_amount > 0.0 create noise from base seed
		base_x = None
		if st.session_state[st.session_state["generation_mode"]]['variant_amount'] > 0.0:
			target_seed_randomizer = seed_to_int('')  # random seed
			torch.manual_seed(seed)  # this has to be the single starting seed (not per-iteration)
			base_x = create_random_tensors([opt_C,
											st.session_state[st.session_state["generation_mode"]]['height'] // opt_f,
											st.session_state[st.session_state["generation_mode"]]['width'] // opt_f],
										   seeds=[seed])
			# we don't want all_seeds to be sequential from starting seed with variants,
			# since that makes the same variants each time,
			# so we add target_seed_randomizer as a random offset
			for si in range(len(all_seeds)):
				all_seeds[si] += target_seed_randomizer

		for n in range(st.session_state[st.session_state["generation_mode"]]['n_iter']):
			print(f"Iteration: {n + 1}/{st.session_state[st.session_state['generation_mode']]['n_iter']}")
			prompts = all_prompts[n * st.session_state[st.session_state['generation_mode']]['batch_size']:(n + 1) * batch_size]
			captions = prompt_matrix_parts[n * batch_size:(n + 1) * batch_size]
			seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

			print(prompt)

			if st.session_state['defaults'].general.optimized:
				st.session_state.modelCS.to(st.session_state['defaults'].general.gpu)

			uc = (st.session_state["model"] if not st.session_state[
				'defaults'].general.optimized else st.session_state.modelCS).get_learned_conditioning(
				len(prompts) * [""])

			if isinstance(prompts, tuple):
				prompts = list(prompts)

			# split the prompt if it has : for weighting
			# TODO for speed it might help to have this occur when all_prompts filled??
			weighted_subprompts = split_weighted_subprompts(prompts[0], st.session_state[st.session_state['generation_mode']]['normalize_prompt_weights'])

			# sub-prompt weighting used if more than 1
			if len(weighted_subprompts) > 1:
				c = torch.zeros_like(uc)  # i dont know if this is correct.. but it works
				for i in range(0, len(weighted_subprompts)):
					# note if alpha negative, it functions same as torch.sub
					c = torch.add(c, (st.session_state["model"] if not st.session_state[
						'defaults'].general.optimized else st.session_state.modelCS).get_learned_conditioning(
						weighted_subprompts[i][0]), alpha=weighted_subprompts[i][1])
			else:  # just behave like usual
				c = (st.session_state["model"] if not st.session_state[
					'defaults'].general.optimized else st.session_state.modelCS).get_learned_conditioning(prompts)

			shape = [opt_C,
					 st.session_state[st.session_state['generation_mode']]['height'] // opt_f,
					 st.session_state[st.session_state['generation_mode']]['width'] // opt_f]

			if st.session_state['defaults'].general.optimized:
				mem = torch.cuda.memory_allocated() / 1e6
				st.session_state.modelCS.to("cpu")
				while (torch.cuda.memory_allocated() / 1e6 >= mem):
					time.sleep(1)

			if st.session_state[st.session_state['generation_mode']]['noise_mode'] == 1 or st.session_state[st.session_state['generation_mode']]['noise_mode'] == 3:
				# TODO params for find_noise_to_image
				x = torch.cat(batch_size * [find_noise_for_image.find_noise_for_image(
					st.session_state["model"], st.session_state["device"],
					init_img.convert('RGB'), '', st.session_state[st.session_state['generation_mode']]['find_noise_steps'], 0.0, normalize=True,
					generation_callback=generation_callback,
				)], dim=0)
			else:
				# we manually generate all input noises because each one should have a specific seed
				x = create_random_tensors(shape, seeds=seeds)

			if st.session_state[st.session_state['generation_mode']]['variant_amount'] > 0.0:  # we are making variants
				# using variant_seed as sneaky toggle,
				# when not None or '' use the variant_seed
				# otherwise use seeds
				if st.session_state[st.session_state['generation_mode']]['variant_seed'] != None and st.session_state[st.session_state['generation_mode']]['variant_seed'] != '':
					specified_variant_seed = seed_to_int(st.session_state[st.session_state['generation_mode']]['variant_seed'])
					torch.manual_seed(specified_variant_seed)
					seeds = [specified_variant_seed]
				# finally, slerp base_x noise to target_x noise for creating a variant
				x = slerp(st.session_state['defaults'].general.gpu, max(0.0, min(1.0, st.session_state[st.session_state['generation_mode']]['variant_amount'])), base_x, x)

			samples_ddim = func_sample(init_data=init_data,
									   x=x,
									   conditioning=c,
									   unconditional_conditioning=uc,
									   sampler_name=st.session_state[st.session_state['generation_mode']]['sampler_name'])

			if st.session_state['defaults'].general.optimized:
				st.session_state.modelFS.to(st.session_state['defaults'].general.gpu)

			x_samples_ddim = (st.session_state["model"] if not st.session_state[
				'defaults'].general.optimized else st.session_state.modelFS).decode_first_stage(samples_ddim)
			x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

			for i, x_sample in enumerate(x_samples_ddim):
				sanitized_prompt = slugify(prompts[i])

				if st.session_state[st.session_state['generation_mode']]['sort_samples']:
					full_path = os.path.join(os.getcwd(), sample_path, sanitized_prompt)

					sanitized_prompt = sanitized_prompt[:220 - len(full_path)]
					sample_path_i = os.path.join(sample_path, sanitized_prompt)

					# print(f"output folder length: {len(os.path.join(os.getcwd(), sample_path_i))}")
					# print(os.path.join(os.getcwd(), sample_path_i))

					os.makedirs(sample_path_i, exist_ok=True)
					base_count = get_next_sequence_number(sample_path_i)
					filename = f"{base_count:05}-{st.session_state[st.session_state['generation_mode']]['steps']}_{st.session_state[st.session_state['generation_mode']]['sampler_name']}_{seeds[i]}"
				else:
					full_path = os.path.join(os.getcwd(), sample_path)
					sample_path_i = sample_path
					base_count = get_next_sequence_number(sample_path_i)
					filename = f"{base_count:05}-{st.session_state[st.session_state['generation_mode']]['steps']}_{st.session_state[st.session_state['generation_mode']]['sampler_name']}_{seeds[i]}_{sanitized_prompt}"[
							   :220 - len(full_path)]  # same as before

				x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
				x_sample = x_sample.astype(np.uint8)
				image = Image.fromarray(x_sample)
				original_sample = x_sample
				original_filename = filename

				if st.session_state[st.session_state['generation_mode']]['use_GFPGAN'] \
						and st.session_state["GFPGAN"] is not None \
						and not st.session_state[st.session_state['generation_mode']]['use_RealESRGAN']:
					# skip_save = True # #287 >_>
					torch_gc()
					cropped_faces, restored_faces, restored_img = st.session_state["GFPGAN"].enhance(
						x_sample[:, :, ::-1], has_aligned=False, only_center_face=False, paste_back=True)
					gfpgan_sample = restored_img[:, :, ::-1]
					gfpgan_image = Image.fromarray(gfpgan_sample)
					gfpgan_filename = original_filename + '-gfpgan'

					save_sample(gfpgan_image, i)

					output_images.append(gfpgan_image)  # 287
					if simple_templating:
						grid_captions.append(captions[i] + "\ngfpgan")

				if st.session_state[st.session_state['generation_mode']]['use_RealESRGAN'] \
						and st.session_state['RealESRGAN'] is not None \
						and not st.session_state[st.session_state['generation_mode']]['use_GFPGAN']:
					# skip_save = True # #287 >_>
					torch_gc()

					if st.session_state["RealESRGAN"].model.name != st.session_state[st.session_state['generation_mode']]['realesrgan_model_name']:
						# try_loading_RealESRGAN(realesrgan_model_name)
						load_models(use_GFPGAN=st.session_state[st.session_state['generation_mode']]['use_GFPGAN'],
									use_RealESRGAN=st.session_state[st.session_state['generation_mode']]['use_RealESRGAN'],
									RealESRGAN_model=st.session_state[st.session_state['generation_mode']]['realesrgan_model_name'])

					output, img_mode = st.session_state['RealESRGAN'].enhance(x_sample[:, :, ::-1])
					esrgan_filename = original_filename + '-esrgan4x'
					esrgan_sample = output[:, :, ::-1]
					esrgan_image = Image.fromarray(esrgan_sample)

					# save_sample(image, sample_path_i, original_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
					# normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
					# save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)

					save_sample(esrgan_image, i)

					output_images.append(esrgan_image)  # 287
					if simple_templating:
						grid_captions.append(captions[i] + "\nesrgan")

				if st.session_state[st.session_state['generation_mode']]['use_RealESRGAN'] \
						and st.session_state["RealESRGAN"] is not None and st.session_state[st.session_state['generation_mode']]['use_GFPGAN'] and st.session_state[
					"GFPGAN"] is not None:
					# skip_save = True # #287 >_>
					torch_gc()
					cropped_faces, restored_faces, restored_img = st.session_state["GFPGAN"].enhance(
						x_sample[:, :, ::-1], has_aligned=False, only_center_face=False, paste_back=True)
					gfpgan_sample = restored_img[:, :, ::-1]

					if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
						# try_loading_RealESRGAN(realesrgan_model_name)
						load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN,
									RealESRGAN_model=realesrgan_model_name)

					output, img_mode = st.session_state["RealESRGAN"].enhance(gfpgan_sample[:, :, ::-1])
					gfpgan_esrgan_filename = original_filename + '-gfpgan-esrgan4x'
					gfpgan_esrgan_sample = output[:, :, ::-1]
					gfpgan_esrgan_image = Image.fromarray(gfpgan_esrgan_sample)

					save_sample(gfpgan_esrgan_image, i)

					output_images.append(gfpgan_esrgan_image)  # 287

					if simple_templating:
						grid_captions.append(captions[i] + "\ngfpgan_esrgan")

				if st.session_state[st.session_state['generation_mode']]['mask_restore'] and init_mask:
					# init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
					init_mask = init_mask.filter(ImageFilter.GaussianBlur(st.session_state[st.session_state['generation_mode']]['mask_blur_strength']))
					init_mask = init_mask.convert('L')
					init_img = init_img.convert('RGB')
					image = image.convert('RGB')

					if st.session_state[st.session_state['generation_mode']]['use_RealESRGAN'] and st.session_state["RealESRGAN"] is not None:
						if st.session_state["RealESRGAN"].model.name != st.session_state['realesrgan_model_name']:
							# try_loading_RealESRGAN(realesrgan_model_name)
							load_models()

						output, img_mode = st.session_state["RealESRGAN"].enhance(np.array(init_img, dtype=np.uint8))
						init_img = Image.fromarray(output)
						init_img = init_img.convert('RGB')

						output, img_mode = st.session_state["RealESRGAN"].enhance(np.array(init_mask, dtype=np.uint8))
						init_mask = Image.fromarray(output)
						init_mask = init_mask.convert('L')

					image = Image.composite(init_img, image, init_mask)

				if st.session_state[st.session_state['generation_mode']]['save_individual_images']:
					save_sample(image,  i)

					if not st.session_state[st.session_state['generation_mode']]['use_GFPGAN'] \
							or not st.session_state[st.session_state['generation_mode']]['use_RealESRGAN']:
						output_images.append(image)

				# if add_original_image or not simple_templating:
				# output_images.append(image)
				# if simple_templating:
				# grid_captions.append( captions[i] )

				if st.session_state['defaults'].general.optimized:
					mem = torch.cuda.memory_allocated() / 1e6
					st.session_state.modelFS.to("cpu")
					while (torch.cuda.memory_allocated() / 1e6 >= mem):
						time.sleep(1)

		if st.session_state[st.session_state['generation_mode']]['prompt_matrix'] \
				or st.session_state[st.session_state['generation_mode']]['save_grid']:
			if st.session_state[st.session_state['generation_mode']]['prompt_matrix']:
				if simple_templating:
					grid = image_grid(output_images, st.session_state[st.session_state['generation_mode']]['n_iter'], force_n_rows=frows, captions=grid_captions)
				else:
					grid = image_grid(output_images, st.session_state[st.session_state['generation_mode']]['n_iter'], force_n_rows=1 << ((len(prompt_matrix_parts) - 1) // 2))
					try:
						grid = draw_prompt_matrix(grid, st.session_state[st.session_state['generation_mode']]['width'],
												  st.session_state[st.session_state['generation_mode']]['height'],
												  prompt_matrix_parts)
					except:
						import traceback
						print("Error creating prompt_matrix text:", file=sys.stderr)
						print(traceback.format_exc(), file=sys.stderr)
			else:
				grid = image_grid(output_images, batch_size)

			if grid and (batch_size > 1
						 or n_iter > 1):
				output_images.insert(0, grid)

			grid_count = get_next_sequence_number(st.session_state[st.session_state['generation_mode']]['outpath'], 'grid-')
			grid_file = f"grid-{grid_count:05}-{seed}_{slugify(prompts[i].replace(' ', '_')[:220 - len(full_path)])}.{grid_ext}"
			grid.save(os.path.join(st.session_state[st.session_state['generation_mode']]['outpath'], grid_file), grid_format, quality=grid_quality, lossless=grid_lossless,
					  optimize=True)

		toc = time.time()

	mem_max_used, mem_total = mem_mon.read_and_stop()
	time_diff = time.time() - start_time

	info = f"""
            {prompt}
            Steps: {st.session_state[st.session_state['generation_mode']]['steps']}, Sampler: {st.session_state[st.session_state['generation_mode']]['sampler_name']}, CFG scale: {st.session_state[st.session_state['generation_mode']]['cfg_scale']}, Seed: {seed}{', Denoising strength: ' + str(st.session_state[st.session_state['generation_mode']]['denoising_strength']) if init_img is not None else ''}{', GFPGAN' if st.session_state[st.session_state['generation_mode']]['use_GFPGAN'] and st.session_state["GFPGAN"] is not None else ''}{', ' + st.session_state[st.session_state['generation_mode']]['realesrgan_model_name'] if st.session_state[st.session_state['generation_mode']]['use_RealESRGAN'] and st.session_state["RealESRGAN"] is not None else ''}{', Prompt Matrix Mode.' if st.session_state[st.session_state['generation_mode']]['prompt_matrix'] else ''}""".strip()
	stats = f'''
            Took {round(time_diff, 2)}s total ({round(time_diff / (len(all_prompts)), 2)}s per image)
            Peak memory usage: {-(mem_max_used // -1_048_576)} MiB / {-(mem_total // -1_048_576)} MiB / {round(mem_max_used / mem_total * 100, 3)}%'''

	for comment in comments:
		info += "\n\n" + comment

	# mem_mon.stop()
	# del mem_mon
	torch_gc()

	return output_images, seed, info, stats


def resize_image(resize_mode, im, width, height):
	LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
	if resize_mode == 0:
		res = im.resize((width, height), resample=LANCZOS)
	elif resize_mode == 1:
		ratio = width / height
		src_ratio = im.width / im.height

		src_w = width if ratio > src_ratio else im.width * height // im.height
		src_h = height if ratio <= src_ratio else im.height * width // im.width

		resized = im.resize((src_w, src_h), resample=LANCZOS)
		res = Image.new("RGBA", (width, height))
		res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
	else:
		ratio = width / height
		src_ratio = im.width / im.height

		src_w = width if ratio < src_ratio else im.width * height // im.height
		src_h = height if ratio >= src_ratio else im.height * width // im.width

		resized = im.resize((src_w, src_h), resample=LANCZOS)
		res = Image.new("RGBA", (width, height))
		res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

		if ratio < src_ratio:
			fill_height = height // 2 - src_h // 2
			res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
			res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
					  box=(0, fill_height + src_h))
		elif ratio > src_ratio:
			fill_width = width // 2 - src_w // 2
			res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
			res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
					  box=(fill_width + src_w, 0))

	return res

#
