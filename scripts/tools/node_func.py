from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from torchvision import transforms
from PIL import Image
from einops import rearrange, repeat
import os
import gc
import k_diffusion as K
import streamlit as st
import torch.nn as nn
from streamlit import StopException
from torch import autocast
import numpy as np




def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

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


def load_var_model_from_config(config_var, ckpt_var, device, verbose=False, half_precision=True):
    #model.to("cpu")
    torch_gc()
    print(f"Loading model from {ckpt_var}")
    pl_sd = torch.load(ckpt_var, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if "model" in st.session_state:
        del st.session_state["model"]
    if "model_var" not in st.session_state:
        st.session_state["model_var"] = instantiate_from_config(config_var.model)
    else:
        print("Variation model already loaded.")
    m, u = st.session_state["model_var"].load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    #model.to("cpu")
    torch_gc()
    st.session_state["model_var"].half().to(device)
    st.session_state["model_var"].eval()


def variations(input_im, outdir, var_samples, var_plms, v_cfg_scale, v_steps, v_W, v_H, v_ddim_eta, v_GFPGAN, v_bg_upsampling, v_upscale):
    #im_path="data/example_conditioning/superresolution/sample_0.jpg",
    ckpt_var="/gdrive/MyDrive/sd-clip-vit-l14-img-embed_ema_only.ckpt"
    config_var="./configs/stable-diffusion/sd-image-condition-finetune.yaml"
    outpath=outdir
    scale=v_cfg_scale
    h=v_H
    w=v_W
    n_samples=var_samples
    precision="autocast"
    if var_plms == True:
        plms=True
    ddim_steps=v_steps
    ddim_eta=v_ddim_eta
    device_idx=0


    if var_plms == 'k_dpm_2_a':
        sampler_name = 'dpm_2_ancestral'
    elif var_plms == 'k_dpm_2':
        sampler_name = 'dpm_2'
    elif var_plms == 'k_euler_a':
        sampler_name = 'euler_ancestral'
    elif var_plms == 'k_euler':
        sampler_name = 'euler'
    elif var_plms == 'k_heun':
        sampler_name = 'heun'
    elif var_plms == 'k_lms':
        sampler_name = 'lms'


    device = 'cuda'

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im*2-1
    #input_im = load_im(im_path).to(device)
    config_var = OmegaConf.load(config_var)
    if "model_var" not in st.session_state:

        st.session_state["model_var"] = load_var_model_from_config(config_var, ckpt_var, device)
    model_wrap = K.external.CompVisDenoiser(st.session_state["model_var"])
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
    sigmas = model_wrap.get_sigmas(ddim_steps)
    model_wrap_cfg = CFGDenoiser(model_wrap)

    #if plms:
    #    sampler = PLMSSampler(model_var)
    #    ddim_eta = 0.0
    #else:
    #    sampler = DDIMSampler(model_var)

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    paths = list()
    x_samples_ddim = sample_model(input_im, st.session_state["model_var"], sampler_name, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, sigmas, model_wrap_cfg)
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
        paths.append(f"{sample_path}/{base_count:05}.png")

        base_count += 1
    return paths

def sample_model(input_im, model_var, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, sigmas, model_wrap_cfg):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model_var.ema_scope():

                c = model_var.get_learned_conditioning(input_im).tile(n_samples,1,1)

                if scale != 1.0:
                    uc = torch.zeros_like(c)
                else:
                    uc = None

                shape = [4, h // 8, w // 8]


                t_enc = int(0.5 * ddim_steps)



                x = torch.randn([n_samples, *shape], device='cuda:0') * sigmas[0]

                #x.to('cuda:0')
                #sigmas.to('cuda:0')
                #model_wrap_cfg.to('cuda:0')


                #x_samples_ddim = accelerator.gather(x_samples_ddim)
                samples_ddim = K.sampling.__dict__[f'sample_{sampler}'](model_wrap_cfg, x, sigmas, extra_args={'cond': c, 'uncond': uc, 'cond_scale': scale}, disable=False)

                x_samples_ddim = model_var.decode_first_stage(samples_ddim)
                return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
