# base webui import and utils.
import streamlit as st
from scripts.tools.sd_utils import *

# streamlit imports
from streamlit import StopException
from scripts.tools.node_func import *

from PIL import Image, ImageOps

#from scripts.tools.img2img import img2img

# Temp imports


# end of imports
# ---------------------------------------------------------------------------------------------------------------


try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass


class PluginInfo():
	plugname = "img2img"
	description = "Image to Image"
	isTab = True
	displayPriority = 3


def layoutFunc():
	with st.form("img2img-inputs"):
		st.session_state["generation_mode"] = "img2img"
		st.session_state["img2img"] = {}


		#for now
		st.session_state[st.session_state["generation_mode"]]['use_RealESRGAN'] = False
		st.session_state[st.session_state["generation_mode"]]['use_GFPGAN'] = False

		img2img_input_col, img2img_generate_col = st.columns([10, 1])
		with img2img_input_col:
			# prompt = st.text_area("Input Text","")
			st.session_state["img2img"]["prompt"] = st.text_input("Input Text", "", placeholder="A corgi wearing a top hat as an oil painting.")

		# Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
		img2img_generate_col.write("")
		img2img_generate_col.write("")
		generate_button = img2img_generate_col.form_submit_button("Generate")

		# creating the page layout using columns
		col1_img2img_layout, col2_img2img_layout, col3_img2img_layout = st.columns([1, 2, 2], gap="small")

		with col1_img2img_layout:
			# If we have custom models available on the "models/custom"
			# folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
			"""
      if st.session_state["CustomModel_available"]:
				st.session_state["custom_model"] = st.selectbox("Custom Model:", st.session_state["custom_models"],
									    index=st.session_state["custom_models"].index(st.session_state['defaults'].general.default_model),
							    help="Select the model you want to use. This option is only available if you have custom models \
							    on your 'models/custom' folder. The model name that will be shown here is the same as the name\
							    the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
							    will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4") 	
			else:
				st.session_state["custom_model"] = "Stable Diffusion v1.4"
			"""

			st.session_state["img2img"]["steps"] = st.slider("Sampling Steps",
															 value=st.session_state['defaults'].img2img.sampling_steps,
															 min_value=1, max_value=500)


			st.session_state["img2img"]["sampler_name"] = st.selectbox("Sampling method",
																	   ("k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", "k_heun", "PLMS", "DDIM"),
																	   help="Sampling method to use.")



			st.session_state["img2img"]["mask_mode"] = st.selectbox("Mask Mode",
																	("Mask", "Inverted mask", "Image alpha"),
																	help="Select how you want your image to be masked.\"Mask\" modifies the image where the mask is white.\n\
								 \"Inverted mask\" modifies the image where the mask is black. \"Image alpha\" modifies the image where the image is transparent."
																	)


			st.session_state["img2img"]["width"] = st.slider("Width:", min_value=64, max_value=1024, value=st.session_state['defaults'].img2img.width,
															 step=64)
			st.session_state["img2img"]["height"] = st.slider("Height:", min_value=64, max_value=1024,
															  value=st.session_state['defaults'].img2img.height, step=64)
			st.session_state["img2img"]["seed"] = st.text_input("Seed:", value=st.session_state['defaults'].img2img.seed,
																help=" The seed to use, if left blank a random seed will be generated.")

			st.session_state["img2img"]["noise_mode"] = st.selectbox(
				"Noise Mode", ("Seed", "Find Noise", "Matched Noise", "Find+Matched Noise"),
				help=""
			)

			st.session_state["img2img"]["find_noise_steps"] = st.slider("Find Noise Steps", value=100, min_value=1, max_value=500)
			st.session_state["img2img"]["batch_count"] = st.slider("Batch count.", min_value=1, max_value=100,
																   value=st.session_state['defaults'].img2img.batch_count, step=1,
																   help="How many iterations or batches of images to generate in total.")
			st.session_state["img2img"]["pathmode"] = st.selectbox('Path Structure', ("subfolders", "root"),
																   index=st.session_state[
																	   'defaults'].general.default_path_mode_index,
																   help="subfolders structure will create daily folders plus many subfolders, root will use your outdir as root",
																   key='pathmode-img2img')
			st.session_state["img2img"]["outdir"] = st.text_input("Output Folder",
																  value=st.session_state['defaults'].general.outdir,
																  help=" Output folder", key='outdir-img2img')

			#
			with st.expander("Advanced"):
				st.session_state["img2img"]["separate_prompts"] = st.checkbox("Create Prompt Matrix.",
																			  value=st.session_state['defaults'].img2img.separate_prompts,
																			  help="Separate multiple prompts using the `|` character, and get all combinations of them.")
				st.session_state["img2img"]["normalize_prompt_weights"] = st.checkbox("Normalize Prompt Weights.", value=st.session_state[
					'defaults'].img2img.normalize_prompt_weights,
																					  help="Ensure the sum of all weights add up to 1.0")
				st.session_state["img2img"]["loopback"] = st.checkbox("Loopback.", value=st.session_state['defaults'].img2img.loopback,
																	  help="Use images from previous batch when creating next batch.")
				random_seed_loopback = st.checkbox("Random loopback seed.", #wird nicht benutzt
												   value=st.session_state['defaults'].img2img.random_seed_loopback,
												   help="Random loopback seed")
				st.session_state["img2img"]["img2img_mask_restore"] = st.checkbox("Only modify regenerated parts of image",
																				  value=st.session_state['defaults'].img2img.mask_restore,
																				  help="Enable to restore the unmasked parts of the image with the input, may not blend as well but preserves detail")
				st.session_state["img2img"]["save_individual_images"] = st.checkbox("Save individual images.",
																					value=st.session_state['defaults'].img2img.save_individual_images,
																					help="Save each image generated before any filter or enhancement is applied.")
				st.session_state["img2img"]["save_grid"] = st.checkbox("Save grid", value=st.session_state['defaults'].img2img.save_grid,
																	   help="Save a grid with all the images generated into a single image.")
				st.session_state["img2img"]["group_by_prompt"] = st.checkbox("Group results by prompt",
																			 value=st.session_state['defaults'].img2img.group_by_prompt,
																			 help="Saves all the images with the same prompt into the same folder. \
									      When using a prompt matrix each prompt combination will have its own folder.")
				st.session_state["img2img"]["write_info_files"] = st.checkbox("Write Info file",
																			  value=st.session_state['defaults'].img2img.write_info_files,
																			  help="Save a file next to the image with informartion about the generation.")
				st.session_state["img2img"]["save_as_jpg"] = st.checkbox("Save samples as jpg", value=st.session_state['defaults'].img2img.save_as_jpg,
																		 help="Saves the images as jpg instead of png.")

				if st.session_state["GFPGAN_available"]:
					use_GFPGAN = st.checkbox("Use GFPGAN", value=st.session_state['defaults'].img2img.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation.\
							This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
				else:
					use_GFPGAN = False

				if st.session_state["RealESRGAN_available"]:
					st.session_state["use_RealESRGAN"] = st.checkbox("Use RealESRGAN", value=st.session_state['defaults'].img2img.use_RealESRGAN,
																	 help="Uses the RealESRGAN model to upscale the images after the generation.\
							This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
					st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)
				else:
					st.session_state["use_RealESRGAN"] = False
					st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"

				st.session_state["img2img"]["variant_amount"] = st.slider("Variant Amount:", value=st.session_state['defaults'].img2img.variant_amount,
																		  min_value=0.0, max_value=1.0, step=0.01)
				st.session_state["img2img"]["variant_seed"] = st.text_input("Variant Seed:", value=st.session_state['defaults'].img2img.variant_seed,
																			help="The seed to use when generating a variant, if left blank a random seed will be generated.")
				st.session_state["img2img"]["cfg_scale"] = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0,
																	 value=st.session_state['defaults'].img2img.cfg_scale, step=0.5,
																	 help="How strongly the image should follow the prompt.")
				batch_size = st.slider("Batch size", min_value=1, max_value=100, # wird nicht beutzt
									   value=st.session_state['defaults'].img2img.batch_size, step=1,
									   help="How many images are at once in a batch.\
									       It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish \
									       generation as more images are generated at once.\
									       Default: 1")

				st.session_state["img2img"]["denoising_strength"] = st.slider("Denoising Strength:", value=st.session_state[
					'defaults'].img2img.denoising_strength,
																			  min_value=0.01, max_value=1.0, step=0.01)

			with st.expander("Preview Settings"):
				st.session_state["img2img"]["update_preview"] = st.checkbox("Update Image Preview", value=st.session_state[
					'defaults'].img2img.update_preview,
																			help="If enabled the image preview will be updated during the generation instead of at the end. \
												 You can use the Update Preview \Frequency option bellow to customize how frequent it's updated. \
												 By default this is enabled and the frequency is set to 1 step.")

				st.session_state["img2img"]["update_preview_frequency"] = st.text_input("Update Image Preview Frequency",
																						value=st.session_state['defaults'].img2img.update_preview_frequency,
																						help="Frequency in steps at which the the preview image is updated. By default the frequency \
													  is set to 1 step.")

		with col2_img2img_layout:
			editor_tab, result_tab  = st.tabs(["Editor", "Result"])

			with editor_tab:

				st.session_state["img2img"]["editor_image"] = st.empty()

				st.form_submit_button("Refresh")

				masked_image_holder = st.empty()
				image_holder = st.empty()

				uploaded_images = st.file_uploader(
					"Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg"],
					help="Upload an image which will be used for the image to image generation.",
				)
				if uploaded_images:
					image = Image.open(uploaded_images).convert('RGBA')
					new_img = image.resize((int(st.session_state["img2img"]["width"] / 2), int(st.session_state["img2img"]["height"] / 2)))
					image_holder.image(new_img)

				mask_holder = st.empty()

				st.session_state["img2img"]["uploaded_masks"] = st.file_uploader(
					"Upload Mask", accept_multiple_files=False, type=["png", "jpg", "jpeg"],
					help="Upload an mask image which will be used for masking the image to image generation.",
				)
				if st.session_state["img2img"]["uploaded_masks"]:
					mask = Image.open(st.session_state["img2img"]["uploaded_masks"])
					if mask.mode == "RGBA":
						mask = mask.convert('RGBA')
						background = Image.new('RGBA', mask.size, (0, 0, 0))
						mask = Image.alpha_composite(background, mask)
					mask = mask.resize((st.session_state["img2img"]["width"], st.session_state["img2img"]["height"]))
					mask_holder.image(mask)

				if uploaded_images and st.session_state["img2img"]["uploaded_masks"]:
					if st.session_state["img2img"]["mask_mode"] != 2:
						final_img = new_img.copy()
						alpha_layer = mask.convert('L')
						strength = st.session_state["img2img"]["denoising_strength"]
						if st.session_state["img2img"]["mask_mode"] == 0:
							alpha_layer = ImageOps.invert(alpha_layer)
							alpha_layer = alpha_layer.point(lambda a: a * strength)
							alpha_layer = ImageOps.invert(alpha_layer)
						elif st.session_state["img2img"]["mask_mode"] == 1:
							alpha_layer = alpha_layer.point(lambda a: a * strength)
							alpha_layer = ImageOps.invert(alpha_layer)

						final_img.putalpha(alpha_layer)

						with masked_image_holder.container():
							st.text("Masked Image Preview")
							st.image(final_img)



			# create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
			with col3_img2img_layout:
				preview_image = st.empty()
				st.session_state["img2img"]["preview_image"] = preview_image

				# st.session_state["loading"] = st.empty()

				st.session_state["img2img"]["progress_bar_text"] = st.empty()
				st.session_state["img2img"]["progress_bar"] = st.empty()

				message = st.empty()



	# if uploaded_images:
	# image = Image.open(uploaded_images).convert('RGB')
	##img_array = np.array(image) # if you want to pass it to OpenCV
	# new_img = image.resize((width, height))
	# st.image(new_img, use_column_width=True)

	if generate_button:
		# print("Loading models")
		# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.

		st.session_state["img2img"]["init_info"] = None
		st.session_state[st.session_state['generation_mode']]['prompt_matrix'] = None

		if uploaded_images:
			image = Image.open(uploaded_images).convert('RGBA')
			st.session_state["img2img"]["new_img"] = image.resize((st.session_state["img2img"]["width"], st.session_state["img2img"]["height"]))
			# img_array = np.array(image) # if you want to pass it to OpenCV
			new_mask = None
			if st.session_state["img2img"]["uploaded_masks"]:
				mask = Image.open(st.session_state["img2img"]["uploaded_masks"]).convert('RGBA')
				new_mask = mask.resize((st.session_state["img2img"]["width"], st.session_state["img2img"]["height"]))

			try:

				output_images, seed, info, stats = img2img(prompt=st.session_state["img2img"]["prompt"],
														   init_info=st.session_state["img2img"]["new_img"],
														   init_info_mask=new_mask,
														   mask_mode=st.session_state["img2img"]["mask_mode"],
														   mask_restore=st.session_state["img2img"]["img2img_mask_restore"],
														   ddim_steps=st.session_state["img2img"]["steps"],
														   sampler_name=st.session_state["img2img"]["sampler_name"],
														   n_iter=st.session_state["img2img"]["batch_count"],
														   cfg_scale=st.session_state["img2img"]["cfg_scale"],
														   denoising_strength=st.session_state["img2img"]["denoising_strength"],
														   variant_seed=st.session_state["img2img"]["variant_seed"],
														   seed=st.session_state["img2img"]["seed"],
														   noise_mode=st.session_state["img2img"]["noise_mode"],
														   find_noise_steps=st.session_state["img2img"]["find_noise_steps"],
														   width=st.session_state["img2img"]["width"],
														   height=st.session_state["img2img"]["height"],
														   fp=st.session_state['defaults'].general.fp,
														   variant_amount=st.session_state["img2img"]["variant_amount"],
														   ddim_eta=0.0,
														   write_info_files=st.session_state["img2img"]["write_info_files"],
														   RealESRGAN_model=st.session_state["RealESRGAN_model"],
														   separate_prompts=st.session_state["img2img"]["separate_prompts"],
														   normalize_prompt_weights=st.session_state["img2img"]["normalize_prompt_weights"],
														   save_individual_images=st.session_state["img2img"]["save_individual_images"],
														   save_grid=st.session_state["img2img"]["save_grid"],
														   group_by_prompt=st.session_state["img2img"]["group_by_prompt"],
														   save_as_jpg=st.session_state["img2img"]["save_as_jpg"],
														   use_GFPGAN=st.session_state["img2img"]["use_GFPGAN"],
														   use_RealESRGAN=st.session_state["img2img"]["use_RealESRGAN"] if not st.session_state["img2img"]["loopback"] else False,
														   loopback=st.session_state["img2img"]["loopback"]
														   )

				# show a message when the generation is complete.
				message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")

			except (StopException, KeyError) as e:
				print(e)
				print(f"Received Streamlit StopException")

		# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
		# use the current col2 first tab to show the preview_img and update it as its generated.
		# preview_image.image(output_images, width=750)

		# on import run init
