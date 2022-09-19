# base webui import and utils.
import streamlit as st

import os, sys, re, random, datetime, time, math, gc
# streamlit imports
from streamlit import StopException

#other imports
import cv2
from PIL import Image, ImageOps
import torch
import k_diffusion as K
import numpy as np
import time
import torch
import skimage
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from tools.modelloader import load_models



def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '':
        return random.randint(0, 2**32 - 1)

    if type(s) is list:
        seed_list = []
        for seed in s:
            if seed is None or seed == '':
                seed_list.append(random.randint(0, 2**32 - 1))
            else:
                seed_list = s

        return seed_list

    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
    while n >= 2**32:
        n = n >> 32
    return n

class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler
    def get_sampler_name(self):
        return self.schedule
    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T, img_callback=None, log_every_t=None):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        samples_ddim = None
        samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas,
                                                                      extra_args={'cond': conditioning, 'uncond': unconditional_conditioning,
                                                                                  'cond_scale': unconditional_guidance_scale}, disable=False, callback=generation_callback)
        #
        return samples_ddim, None
def process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, save_grid, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
        ddim_eta=0.0, normalize_prompt_weights=True, init_img=None, init_mask=None,
        mask_blur_strength=3, mask_restore=False, denoising_strength=0.75, noise_mode=0, find_noise_steps=1, resize_mode=None, uses_loopback=False,
        uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, jpg_sample=False,
        variant_amount=0.0, variant_seed=None, save_individual_images: bool = True):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    assert prompt is not None
    torch_gc()
    # start time after garbage collection (or before?)
    start_time = time.time()

    # We will use this date here later for the folder name, need to start_time if not need
    run_start_dt = datetime.datetime.now()

    mem_mon = MemUsageMonitor('MemMon')
    mem_mon.start()

    if st.session_state.defaults.general.use_sd_concepts_library:

        prompt_tokens = re.findall('<([a-zA-Z0-9-]+)>', prompt)

        if prompt_tokens:
            # compviz
            tokenizer = (st.session_state["model"] if not st.session_state['defaults'].general.optimized else st.session_state.modelCS).cond_stage_model.tokenizer
            text_encoder = (st.session_state["model"] if not st.session_state['defaults'].general.optimized else st.session_state.modelCS).cond_stage_model.transformer

            # diffusers
            #tokenizer = pipe.tokenizer
            #text_encoder = pipe.text_encoder

            ext = ('pt', 'bin')

            if len(prompt_tokens) > 1:
                for token_name in prompt_tokens:
                    embedding_path = os.path.join(st.session_state['defaults'].general.sd_concepts_library_folder, token_name)
                    if os.path.exists(embedding_path):
                        for files in os.listdir(embedding_path):
                            if files.endswith(ext):
                                load_learned_embed_in_clip(f"{os.path.join(embedding_path, files)}", text_encoder, tokenizer, f"<{token_name}>")
            else:
                embedding_path = os.path.join(st.session_state['defaults'].general.sd_concepts_library_folder, prompt_tokens[0])
                if os.path.exists(embedding_path):
                    for files in os.listdir(embedding_path):
                        if files.endswith(ext):
                            load_learned_embed_in_clip(f"{os.path.join(embedding_path, files)}", text_encoder, tokenizer, f"<{prompt_tokens[0]}>")

                            #


    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    if not ("|" in prompt) and prompt.startswith("@"):
        prompt = prompt[1:]

    negprompt = ''
    if '###' in prompt:
        prompt, negprompt = prompt.split('###', 1)
        prompt = prompt.strip()
        negprompt = negprompt.strip()

    comments = []

    prompt_matrix_parts = []
    simple_templating = False
    add_original_image = not (use_RealESRGAN or use_GFPGAN)

    if prompt_matrix:
        if prompt.startswith("@"):
            simple_templating = True
            add_original_image = not (use_RealESRGAN or use_GFPGAN)
            all_seeds, n_iter, prompt_matrix_parts, all_prompts, frows = oxlamon_matrix(prompt, seed, n_iter, batch_size)
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
    with torch.no_grad(), precision_scope("cuda"), (st.session_state["model"].ema_scope() if not st.session_state['defaults'].general.optimized else nullcontext()):
        init_data = func_init()
        tic = time.time()


        # if variant_amount > 0.0 create noise from base seed
        base_x = None
        if variant_amount > 0.0:
            target_seed_randomizer = seed_to_int('') # random seed
            torch.manual_seed(seed) # this has to be the single starting seed (not per-iteration)
            base_x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=[seed])
            # we don't want all_seeds to be sequential from starting seed with variants,
            # since that makes the same variants each time,
            # so we add target_seed_randomizer as a random offset
            for si in range(len(all_seeds)):
                all_seeds[si] += target_seed_randomizer

        for n in range(n_iter):
            print(f"Iteration: {n+1}/{n_iter}")
            prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
            captions = prompt_matrix_parts[n * batch_size:(n + 1) * batch_size]
            seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

            print(prompt)

            if st.session_state['defaults'].general.optimized:
                st.session_state.modelCS.to(st.session_state['defaults'].general.gpu)

            uc = (st.session_state["model"] if not st.session_state['defaults'].general.optimized else st.session_state.modelCS).get_learned_conditioning(len(prompts) * [negprompt])

            if isinstance(prompts, tuple):
                prompts = list(prompts)

            # split the prompt if it has : for weighting
            # TODO for speed it might help to have this occur when all_prompts filled??
            weighted_subprompts = split_weighted_subprompts(prompts[0], normalize_prompt_weights)

            # sub-prompt weighting used if more than 1
            if len(weighted_subprompts) > 1:
                c = torch.zeros_like(uc) # i dont know if this is correct.. but it works
                for i in range(0, len(weighted_subprompts)):
                    # note if alpha negative, it functions same as torch.sub
                    c = torch.add(c, (st.session_state["model"] if not st.session_state['defaults'].general.optimized else st.session_state.modelCS).get_learned_conditioning(weighted_subprompts[i][0]), alpha=weighted_subprompts[i][1])
            else: # just behave like usual
                c = (st.session_state["model"] if not st.session_state['defaults'].general.optimized else st.session_state.modelCS).get_learned_conditioning(prompts)


            shape = [opt_C, height // opt_f, width // opt_f]

            if st.session_state['defaults'].general.optimized:
                mem = torch.cuda.memory_allocated()/1e6
                st.session_state.modelCS.to("cpu")
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)

            if noise_mode == 1 or noise_mode == 3:
                # TODO params for find_noise_to_image
                x = torch.cat(batch_size * [find_noise_for_image(
                    st.session_state["model"], st.session_state["device"],
                    init_img.convert('RGB'), '', find_noise_steps, 0.0, normalize=True,
                    generation_callback=generation_callback,
                )], dim=0)
            else:
                # we manually generate all input noises because each one should have a specific seed
                x = create_random_tensors(shape, seeds=seeds)

            if variant_amount > 0.0: # we are making variants
                # using variant_seed as sneaky toggle,
                # when not None or '' use the variant_seed
                # otherwise use seeds
                if variant_seed != None and variant_seed != '':
                    specified_variant_seed = seed_to_int(variant_seed)
                    torch.manual_seed(specified_variant_seed)
                    seeds = [specified_variant_seed]
                # finally, slerp base_x noise to target_x noise for creating a variant
                x = slerp(st.session_state['defaults'].general.gpu, max(0.0, min(1.0, variant_amount)), base_x, x)

            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

            if st.session_state['defaults'].general.optimized:
                st.session_state.modelFS.to(st.session_state['defaults'].general.gpu)

            x_samples_ddim = (st.session_state["model"] if not st.session_state['defaults'].general.optimized else st.session_state.modelFS).decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            run_images = []
            for i, x_sample in enumerate(x_samples_ddim):
                sanitized_prompt = slugify(prompts[i])

                percent = i / len(x_samples_ddim)
                st.session_state["progress_bar"].progress(percent if percent < 100 else 100)

                if sort_samples:
                    full_path = os.path.join(os.getcwd(), sample_path, sanitized_prompt)


                    sanitized_prompt = sanitized_prompt[:220-len(full_path)]
                    sample_path_i = os.path.join(sample_path, sanitized_prompt)

                    #print(f"output folder length: {len(os.path.join(os.getcwd(), sample_path_i))}")
                    #print(os.path.join(os.getcwd(), sample_path_i))

                    os.makedirs(sample_path_i, exist_ok=True)
                    base_count = get_next_sequence_number(sample_path_i)
                    filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}"
                else:
                    full_path = os.path.join(os.getcwd(), sample_path)
                    sample_path_i = sample_path
                    base_count = get_next_sequence_number(sample_path_i)
                    filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[:220-len(full_path)] #same as before

                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)
                original_sample = x_sample
                original_filename = filename

                st.session_state["preview_image"].image(image)

                if use_GFPGAN and st.session_state["GFPGAN"] is not None and not use_RealESRGAN:
                    st.session_state["progress_bar_text"].text("Running GFPGAN on image %d of %d..." % (i+1, len(x_samples_ddim)))
                    #skip_save = True # #287 >_>
                    torch_gc()
                    cropped_faces, restored_faces, restored_img = st.session_state["GFPGAN"].enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    gfpgan_sample = restored_img[:,:,::-1]
                    gfpgan_image = Image.fromarray(gfpgan_sample)
                    gfpgan_filename = original_filename + '-gfpgan'

                    save_sample(gfpgan_image, sample_path_i, gfpgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
                                normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback,
                                uses_random_seed_loopback, save_grid, sort_samples, sampler_name, ddim_eta,
                                n_iter, batch_size, i, denoising_strength, resize_mode, False, st.session_state["loaded_model"])

                    output_images.append(gfpgan_image) #287
                    run_images.append(gfpgan_image)

                    if simple_templating:
                        grid_captions.append( captions[i] + "\ngfpgan" )

                elif use_RealESRGAN and st.session_state["RealESRGAN"] is not None and not use_GFPGAN:
                    st.session_state["progress_bar_text"].text("Running RealESRGAN on image %d of %d..." % (i+1, len(x_samples_ddim)))
                    #skip_save = True # #287 >_>
                    torch_gc()

                    if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
                        #try_loading_RealESRGAN(realesrgan_model_name)
                        load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

                    output, img_mode = st.session_state["RealESRGAN"].enhance(x_sample[:,:,::-1])
                    esrgan_filename = original_filename + '-esrgan4x'
                    esrgan_sample = output[:,:,::-1]
                    esrgan_image = Image.fromarray(esrgan_sample)

                    #save_sample(image, sample_path_i, original_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
                    #normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
                    #save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)

                    save_sample(esrgan_image, sample_path_i, esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
                                normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, False, st.session_state["loaded_model"])

                    output_images.append(esrgan_image) #287
                    run_images.append(esrgan_image)

                    if simple_templating:
                        grid_captions.append( captions[i] + "\nesrgan" )

                elif use_RealESRGAN and st.session_state["RealESRGAN"] is not None and use_GFPGAN and st.session_state["GFPGAN"] is not None:
                    st.session_state["progress_bar_text"].text("Running GFPGAN+RealESRGAN on image %d of %d..." % (i+1, len(x_samples_ddim)))
                    #skip_save = True # #287 >_>
                    torch_gc()
                    cropped_faces, restored_faces, restored_img = st.session_state["GFPGAN"].enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    gfpgan_sample = restored_img[:,:,::-1]

                    if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
                        #try_loading_RealESRGAN(realesrgan_model_name)
                        load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

                    output, img_mode = st.session_state["RealESRGAN"].enhance(gfpgan_sample[:,:,::-1])
                    gfpgan_esrgan_filename = original_filename + '-gfpgan-esrgan4x'
                    gfpgan_esrgan_sample = output[:,:,::-1]
                    gfpgan_esrgan_image = Image.fromarray(gfpgan_esrgan_sample)

                    save_sample(gfpgan_esrgan_image, sample_path_i, gfpgan_esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
                                normalize_prompt_weights, False, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, False, st.session_state["loaded_model"])

                    output_images.append(gfpgan_esrgan_image) #287
                    run_images.append(gfpgan_esrgan_image)

                    if simple_templating:
                        grid_captions.append( captions[i] + "\ngfpgan_esrgan" )
                else:
                    output_images.append(image)
                    run_images.append(image)

                if mask_restore and init_mask:
                    #init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
                    init_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
                    init_mask = init_mask.convert('L')
                    init_img = init_img.convert('RGB')
                    image = image.convert('RGB')

                    if use_RealESRGAN and st.session_state["RealESRGAN"] is not None:
                        if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
                            #try_loading_RealESRGAN(realesrgan_model_name)
                            load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN, RealESRGAN_model=realesrgan_model_name)

                        output, img_mode = st.session_state["RealESRGAN"].enhance(np.array(init_img, dtype=np.uint8))
                        init_img = Image.fromarray(output)
                        init_img = init_img.convert('RGB')

                        output, img_mode = st.session_state["RealESRGAN"].enhance(np.array(init_mask, dtype=np.uint8))
                        init_mask = Image.fromarray(output)
                        init_mask = init_mask.convert('L')

                    image = Image.composite(init_img, image, init_mask)

                if save_individual_images:
                    save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
                                normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback,
                                save_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, save_individual_images, st.session_state["loaded_model"])

                    #if add_original_image or not simple_templating:
                    #output_images.append(image)
                    #if simple_templating:
                    #grid_captions.append( captions[i] )

                if st.session_state['defaults'].general.optimized:
                    mem = torch.cuda.memory_allocated()/1e6
                    st.session_state.modelFS.to("cpu")
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)

            if len(run_images) > 1:
                preview_image = image_grid(run_images, n_iter)
            else:
                preview_image = run_images[0]

            # Constrain the final preview image to 1440x900 so we're not sending huge amounts of data
            # to the browser
            preview_image = constrain_image(preview_image, 1440, 900)
            st.session_state["progress_bar_text"].text("Finished!")
            st.session_state["preview_image"].image(preview_image)

        if prompt_matrix or save_grid:
            if prompt_matrix:
                if simple_templating:
                    grid = image_grid(output_images, n_iter, force_n_rows=frows, captions=grid_captions)
                else:
                    grid = image_grid(output_images, n_iter, force_n_rows=1 << ((len(prompt_matrix_parts)-1)//2))
                    try:
                        grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                    except:
                        import traceback
                        print("Error creating prompt_matrix text:", file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)
            else:
                grid = image_grid(output_images, batch_size)

            if grid and (batch_size > 1  or n_iter > 1):
                output_images.insert(0, grid)

            grid_count = get_next_sequence_number(outpath, 'grid-')
            grid_file = f"grid-{grid_count:05}-{seed}_{slugify(prompts[i].replace(' ', '_')[:220-len(full_path)])}.{grid_ext}"
            grid.save(os.path.join(outpath, grid_file), grid_format, quality=grid_quality, lossless=grid_lossless, optimize=True)

        toc = time.time()

    mem_max_used, mem_total = mem_mon.read_and_stop()
    time_diff = time.time()-start_time

    info = f"""
            {prompt}
            Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', Denoising strength: '+str(denoising_strength) if init_img is not None else ''}{', GFPGAN' if use_GFPGAN and st.session_state["GFPGAN"] is not None else ''}{', '+realesrgan_model_name if use_RealESRGAN and st.session_state["RealESRGAN"] is not None else ''}{', Prompt Matrix Mode.' if prompt_matrix else ''}""".strip()
    stats = f'''
            Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
            Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    for comment in comments:
        info += "\n\n" + comment

    #mem_mon.stop()
    #del mem_mon
    torch_gc()

