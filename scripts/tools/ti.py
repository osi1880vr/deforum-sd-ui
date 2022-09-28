# base webui import and utils.
import streamlit as st
from streamlit_server_state import server_state, server_state_lock

import random, os, json
import itertools
import math


import PIL
from PIL import Image


import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from slugify import slugify
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from scripts.pipelines.stable_diffusion.no_check import NoCheck
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel#, PNDMScheduler
from diffusers.optimization import get_scheduler


logger = get_logger(__name__)


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

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

class TextualInversionDataset(Dataset):
    def __init__(
            self,
            data_root,
            tokenizer,
            learnable_property="object",  # [object, style]
            size=512,
            repeats=100,
            interpolation="bicubic",
            set="train",
            placeholder_token="*",
            center_crop=False,
            templates=None
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if file_path.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
        }[interpolation]

        self.templates = templates
        self.cache = {}
        self.tokenized_templates = [self.tokenizer(
            text.format(self.placeholder_token),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0] for text in self.templates]

    def __len__(self):
        return self._length

    def get_example(self, image_path, flipped):
        if image_path in self.cache:
            return self.cache[image_path]

        example = {}
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        image = transforms.RandomHorizontalFlip(p=1 if flipped else 0)(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["key"] = "-".join([image_path, "-", str(flipped)])
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        self.cache[image_path] = example
        return example

    def __getitem__(self, i):
        flipped = random.choice([False, True])
        example = self.get_example(self.image_paths[i % self.num_images], flipped)
        example["input_ids"] = random.choice(self.tokenized_templates)
        return example


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def save_resume_file(basepath, args, extra = {}):
    info = {"args": st.session_state['textual_inversion']["args"]}
    info["args"].update(extra)
    with open(f"{basepath}/resume.json", "w") as f:
        json.dump(info, f, indent=4)

class Checkpointer:
    def __init__(
            self,
            accelerator,
            vae,
            unet,
            tokenizer,
            placeholder_token,
            placeholder_token_id,
            templates,
            output_dir,
            random_sample_batches,
            sample_batch_size,
            stable_sample_batches,
            seed
    ):
        self.accelerator = accelerator
        self.vae = vae
        self.unet = unet
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.placeholder_token_id = placeholder_token_id
        self.templates = templates
        self.output_dir = output_dir
        self.seed = seed
        self.random_sample_batches = random_sample_batches
        self.sample_batch_size = sample_batch_size
        self.stable_sample_batches = stable_sample_batches

    @torch.no_grad()
    def checkpoint(self, step, text_encoder, save_samples=True, path=None):
        print("Saving checkpoint for step %d..." % step)
        with torch.autocast("cuda"):
            if path is None:
                checkpoints_path = f"{self.output_dir}/checkpoints"
                os.makedirs(checkpoints_path, exist_ok=True)

            unwrapped = self.accelerator.unwrap_model(text_encoder)

            # Save a checkpoint
            learned_embeds = unwrapped.get_input_embeddings().weight[self.placeholder_token_id]
            learned_embeds_dict = {self.placeholder_token: learned_embeds.detach().cpu()}

            filename = f"%s_%d.bin" % (slugify(self.placeholder_token), step)
            if path is not None:
                torch.save(learned_embeds_dict, path)
            else:
                torch.save(learned_embeds_dict, f"{checkpoints_path}/{filename}")
                torch.save(learned_embeds_dict, f"{checkpoints_path}/last.bin")
            del unwrapped
            del learned_embeds


    @torch.no_grad()
    def save_samples(self, step, text_encoder, height, width, guidance_scale, eta, num_inference_steps):
        samples_path = f"{self.output_dir}/samples"
        os.makedirs(samples_path, exist_ok=True)

        #if "checker" not in server_state['textual_inversion']:
        #with server_state_lock['textual_inversion']["checker"]:
        server_state['textual_inversion']["checker"] = NoCheck()

        #if "unwrapped" not in server_state['textual_inversion']:
        #	with server_state_lock['textual_inversion']["unwrapped"]:
        server_state['textual_inversion']["unwrapped"] = self.accelerator.unwrap_model(text_encoder)

        #if "pipeline" not in server_state['textual_inversion']:
        #	with server_state_lock['textual_inversion']["pipeline"]:
        # Save a sample image
        server_state['textual_inversion']["pipeline"] = StableDiffusionPipeline(
            text_encoder=server_state['textual_inversion']["unwrapped"],
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            ),
            safety_checker=NoCheck(),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        ).to("cuda")

        server_state['textual_inversion']["pipeline"].enable_attention_slicing()

        if self.stable_sample_batches > 0:
            stable_latents = torch.randn(
                (self.sample_batch_size, server_state['textual_inversion']["pipeline"].unet.in_channels, height // 8, width // 8),
                device=server_state['textual_inversion']["pipeline"].device,
                generator=torch.Generator(device=server_state['textual_inversion']["pipeline"].device).manual_seed(self.seed),
            )

            stable_prompts = [choice.format(self.placeholder_token) for choice in (self.templates * self.sample_batch_size)[:self.sample_batch_size]]

            # Generate and save stable samples
            for i in range(0, self.stable_sample_batches):
                samples = server_state['textual_inversion']["pipeline"](
                    prompt=stable_prompts,
                    height=384,
                    latents=stable_latents,
                    width=384,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_inference_steps=num_inference_steps,
                    output_type='pil'
                )["sample"]
                for idx, im in enumerate(samples):
                    filename = f"stable_sample_%d_%d_step_%d.png" % (i+1, idx+1, step)
                    im.save(f"{samples_path}/{filename}")
                del samples
            del stable_latents

        prompts = [choice.format(self.placeholder_token) for choice in random.choices(self.templates, k=self.sample_batch_size)]
        # Generate and save random samples
        for i in range(0, self.random_sample_batches):
            samples = server_state['textual_inversion']["pipeline"](
                prompt=prompts,
                height=384,
                width=384,
                guidance_scale=guidance_scale,
                eta=eta,
                num_inference_steps=num_inference_steps,
                output_type='pil'
            )["sample"]
            for idx, im in enumerate(samples):
                filename = f"step_%d_sample_%d_%d.png" % (step, i+1, idx+1)
                im.save(f"{samples_path}/{filename}")
            del samples

        del server_state['textual_inversion']["checker"]
        del server_state['textual_inversion']["unwrapped"]
        del server_state['textual_inversion']["pipeline"]
        torch.cuda.empty_cache()

#@retry(RuntimeError, tries=5)
def textual_inversion():
    print ("Running textual inversion.")
    if "pipeline" in server_state["textual_inversion"]:
        del server_state['textual_inversion']["checker"]
        del server_state['textual_inversion']["unwrapped"]
        del server_state['textual_inversion']["pipeline"]

    global_step_offset = 0
    if st.session_state['textual_inversion']['args']["resume_from"]:
        basepath = f"{st.session_state['textual_inversion']['args']['resume_from']}"
        print("Resuming state from %s" % st.session_state['textual_inversion']['args']['resume_from'])
        with open(f"{basepath}/resume.json", 'r') as f:
            state = json.load(f)
        global_step_offset = state["args"].get("global_step", 0)

        print("We've trained %d steps so far" % global_step_offset)
    else:
        basepath = f"{st.session_state['textual_inversion']['args']['output_dir']}/{slugify(st.session_state['textual_inversion']['args']['placeholder_token'])}"
        os.makedirs(basepath, exist_ok=True)


    accelerator = Accelerator(
        gradient_accumulation_steps=st.session_state['textual_inversion']['args']['gradient_accumulation_steps'],
        mixed_precision=st.session_state['textual_inversion']['args']['mixed_precision']
    )

    # If passed along, set the training seed.
    if st.session_state['textual_inversion']['args']['seed']:
        set_seed(st.session_state['textual_inversion']['args']['seed'])

    #if "tokenizer" not in server_state["textual_inversion"]:
    # Load the tokenizer and add the placeholder token as a additional special token
    #with server_state_lock['textual_inversion']["tokenizer"]:
    if st.session_state['textual_inversion']['args']['tokenizer_name']:
        server_state['textual_inversion']["tokenizer"] = CLIPTokenizer.from_pretrained(st.session_state['textual_inversion']['args']['tokenizer_name'])
    elif st.session_state['textual_inversion']['args']['pretrained_model_name_or_path']:
        server_state['textual_inversion']["tokenizer"] = CLIPTokenizer.from_pretrained(
            st.session_state['textual_inversion']['args']['pretrained_model_name_or_path'] + '/tokenizer'
        )

    # Add the placeholder token in tokenizer
    num_added_tokens = server_state['textual_inversion']["tokenizer"].add_tokens(st.session_state['textual_inversion']['args']['placeholder_token'])
    if num_added_tokens == 0:
        st.error(
            f"The tokenizer already contains the token {st.session_state['textual_inversion']['args']['placeholder_token']}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = server_state['textual_inversion']["tokenizer"].encode(st.session_state['textual_inversion']['args']['initializer_token'], add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        st.error("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = server_state['textual_inversion']["tokenizer"].convert_tokens_to_ids(st.session_state['textual_inversion']['args']['placeholder_token'])

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        st.session_state['textual_inversion']['args']['pretrained_model_name_or_path'] + '/text_encoder',
        )
    vae = AutoencoderKL.from_pretrained(
        st.session_state['textual_inversion']['args']['pretrained_model_name_or_path'] + '/vae',
        )
    unet = UNet2DConditionModel.from_pretrained(
        st.session_state['textual_inversion']['args']['pretrained_model_name_or_path'] + '/unet',
        )

    base_templates = imagenet_style_templates_small if st.session_state['textual_inversion']['args']['learnable_property'] == "style" else imagenet_templates_small
    if st.session_state['textual_inversion']['args']['custom_templates']:
        templates = st.session_state['textual_inversion']['args']['custom_templates'].split(";")
    else:
        templates = base_templates

    slice_size = unet.config.attention_head_dim // 2
    unet.set_attention_slice(slice_size)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(server_state['textual_inversion']["tokenizer"]))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data

    if "resume_checkpoint" in st.session_state['textual_inversion']['args']:
        if st.session_state['textual_inversion']['args']['resume_checkpoint'] is not None:
            token_embeds[placeholder_token_id] = torch.load(st.session_state['textual_inversion']['args']['resume_checkpoint'])[st.session_state['textual_inversion']['args']['placeholder_token']]
    else:
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    checkpointer = Checkpointer(
        accelerator=accelerator,
        vae=vae,
        unet=unet,
        tokenizer=server_state['textual_inversion']["tokenizer"],
        placeholder_token=st.session_state['textual_inversion']['args']['placeholder_token'],
        placeholder_token_id=placeholder_token_id,
        templates=templates,
        output_dir=basepath,
        sample_batch_size=st.session_state['textual_inversion']['args']['sample_batch_size'],
        random_sample_batches=st.session_state['textual_inversion']['args']['random_sample_batches'],
        stable_sample_batches=st.session_state['textual_inversion']['args']['stable_sample_batches'],
        seed=st.session_state['textual_inversion']['args']['seed']
    )

    if st.session_state['textual_inversion']['args']['scale_lr']:
        st.session_state['textual_inversion']['args']['learning_rate'] = (
                st.session_state['textual_inversion']['args']['learning_rate'] * st.session_state['textual_inversion'][
            'args']['gradient_accumulation_steps'] * st.session_state['textual_inversion']['args']['train_batch_size'] * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=st.session_state['textual_inversion']['args']['learning_rate'],
        betas=(st.session_state['textual_inversion']['args']['adam_beta1'], st.session_state['textual_inversion']['args']['adam_beta2']),
        weight_decay=st.session_state['textual_inversion']['args']['adam_weight_decay'],
        eps=st.session_state['textual_inversion']['args']['adam_epsilon'],
    )

    # TODO (patil-suraj): laod scheduler using args
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt"
    )

    train_dataset = TextualInversionDataset(
        data_root=st.session_state['textual_inversion']['args']['train_data_dir'],
        tokenizer=server_state['textual_inversion']["tokenizer"],
        size=st.session_state['textual_inversion']['args']['resolution'],
        placeholder_token=st.session_state['textual_inversion']['args']['placeholder_token'],
        repeats=st.session_state['textual_inversion']['args']['repeats'],
        learnable_property=st.session_state['textual_inversion']['args']['learnable_property'],
        center_crop=st.session_state['textual_inversion']['args']['center_crop'],
        set="train",
        templates=templates
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=st.session_state['textual_inversion']['args']['train_batch_size'], shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / st.session_state['textual_inversion']['args']['gradient_accumulation_steps'])
    if st.session_state['textual_inversion']['args']['max_train_steps'] is None:
        st.session_state['textual_inversion']['args']['max_train_steps'] = st.session_state['textual_inversion']['args']['num_train_epochs'] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        st.session_state['textual_inversion']['args']['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=st.session_state['textual_inversion']['args']['lr_warmup_steps'] * st.session_state['textual_inversion']['args']['gradient_accumulation_steps'],
        num_training_steps=st.session_state['textual_inversion']['args']['max_train_steps'] * st.session_state['textual_inversion']['args']['gradient_accumulation_steps'],
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae and unet to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    # Keep vae and unet in eval mode as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / st.session_state['textual_inversion']['args']['gradient_accumulation_steps'])
    if overrode_max_train_steps:
        st.session_state['textual_inversion']['args']['max_train_steps'] = st.session_state['textual_inversion']['args']['num_train_epochs'] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    st.session_state['textual_inversion']['args']['num_train_epochs'] = math.ceil(st.session_state['textual_inversion']['args']['max_train_steps'] / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=st.session_state['textual_inversion']['args'])

    # Train!
    total_batch_size = st.session_state['textual_inversion']['args']['train_batch_size'] * accelerator.num_processes * st.session_state[
        'textual_inversion']['args']['gradient_accumulation_steps']

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {st.session_state['textual_inversion']['args']['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {st.session_state['textual_inversion']['args']['train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {st.session_state['textual_inversion']['args']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {st.session_state['textual_inversion']['args']['max_train_steps']}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(st.session_state['textual_inversion']['args']['max_train_steps']), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    encoded_pixel_values_cache = {}

    try:
        for epoch in range(st.session_state['textual_inversion']['args']['num_train_epochs']):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    key = "|".join(batch["key"])
                    if encoded_pixel_values_cache.get(key, None) is None:
                        encoded_pixel_values_cache[key] = vae.encode(batch["pixel_values"]).latent_dist
                    latents = encoded_pixel_values_cache[key].sample().detach().half() * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(server_state['textual_inversion']["tokenizer"])) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % st.session_state['textual_inversion']['args']['checkpoint_frequency'] == 0 and global_step > 0 and accelerator.is_main_process:
                        checkpointer.checkpoint(global_step + global_step_offset, text_encoder)
                        save_resume_file(basepath, st.session_state['textual_inversion']['args'], {
                            "global_step": global_step + global_step_offset,
                            "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
                        })
                        checkpointer.save_samples(
                            global_step + global_step_offset,
                            text_encoder,
                            st.session_state['textual_inversion']['args']['resolution'], st.session_state['textual_inversion']['args'][
                                'resolution'], 7.5, 0.0, st.session_state['textual_inversion']['args']['sample_steps'])

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= st.session_state['textual_inversion']['args']['max_train_steps']:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            print("Finished! Saving final checkpoint and resume state.")
            checkpointer.checkpoint(
                global_step + global_step_offset,
                text_encoder,
                path=f"{basepath}/learned_embeds.bin"
            )

            save_resume_file(basepath, st.session_state['textual_inversion']['args'], {
                "global_step": global_step + global_step_offset,
                "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
            })

            accelerator.end_training()

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("Interrupted, saving checkpoint and resume state...")
            checkpointer.checkpoint(global_step + global_step_offset, text_encoder)
            save_resume_file(basepath, st.session_state['textual_inversion']['args'], {
                "global_step": global_step + global_step_offset,
                "resume_checkpoint": f"{basepath}/checkpoints/last.bin"
            })
        quit()
