# base webui import and utils.
import streamlit as st
from scripts.tools.sd_utils import *

# streamlit imports
from streamlit import StopException
from scripts.tools.img2var import *

from PIL import Image, ImageOps

from scripts.tools.html_gallery import *

#from scripts.tools.img2var import img2var

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
	plugname = "img2var"
	description = "Image to Variations"
	isTab = True
	displayPriority = 3


def layoutFunc():
	with st.form("img2var-inputs"):
		st.session_state["generation_mode"] = "img2var"
		st.session_state["img2var"] = {}
		img2var_latestimages = []


		#for now
		st.session_state[st.session_state["generation_mode"]]['use_RealESRGAN'] = False
		st.session_state[st.session_state["generation_mode"]]['use_GFPGAN'] = False

		img2var_input_col, img2var_generate_col = st.columns([10, 1])
		with img2var_input_col:
			st.text('just upload an Image, select the setup parameters and generate, no prompts here')

		# Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
		img2var_generate_col.write("")
		img2var_generate_col.write("")
		generate_button = img2var_generate_col.form_submit_button("Generate")

		# creating the page layout using columns
		col1_img2var_layout, col2_img2var_layout, col3_img2var_layout = st.columns([1, 2, 1], gap="large")

		with col1_img2var_layout:
			# If we have custom models available on the "models/custom"
			# folder then we show a menu to select which model we want to use, otherwise we use the main model for SD

			st.session_state["img2var"]["steps"] = st.slider("Sampling Steps",
															 value=st.session_state['defaults'].img2var.sampling_steps,
															 min_value=1, max_value=500)


			st.session_state["img2var"]["sampler_name"] = st.selectbox("Sampling method",
																	   ("k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", "k_heun", "PLMS", "DDIM"),
																	   help="Sampling method to use.")


			st.session_state["img2var"]["width"] = st.slider("Width:", min_value=64, max_value=1024, value=st.session_state['defaults'].img2var.width,
															 step=64)
			st.session_state["img2var"]["height"] = st.slider("Height:", min_value=64, max_value=1024,
															  value=st.session_state['defaults'].img2var.height, step=64)
			st.session_state["img2var"]["ddim_eta"] = st.slider("DDIM eta:", min_value=0.0, max_value=1.0,
												  value=st.session_state['defaults'].img2var.ddim_eta, step=0.1)

			st.session_state["img2var"]["var_samples"] = st.slider("Number of Variations", min_value=1, max_value=100,
																   value=st.session_state['defaults'].img2var.var_samples, step=1,
																   help="How many iterations or batches of images to generate in total.")


			#
			with st.expander("Advanced"):
				#st.session_state["img2var"]["save_individual_images"] = st.checkbox("Save individual images.",
				#																	value=st.session_state['defaults'].img2var.save_individual_images,
				#																	help="Save each image generated before any filter or enhancement is applied.")



				#st.session_state["img2var"]["save_grid"] = st.checkbox("Save grid", value=st.session_state['defaults'].img2var.save_grid,
				#													   help="Save a grid with all the images generated into a single image.")
				#st.session_state["img2var"]["save_as_jpg"] = st.checkbox("Save samples as jpg", value=st.session_state['defaults'].img2var.save_as_jpg,
				#														 help="Saves the images as jpg instead of png.")



				"""
				if st.session_state["RealESRGAN_available"]:
					st.session_state["use_RealESRGAN"] = st.checkbox("Use RealESRGAN", value=st.session_state['defaults'].img2var.use_RealESRGAN,
																	 help="Uses the RealESRGAN model to upscale the images after the generation.\
							This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
					st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)
				else:
					st.session_state["use_RealESRGAN"] = False
					st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"
				"""


		with col2_img2var_layout:
			editor_tab, result_tab  = st.tabs(["Editor", "Result"])

			with editor_tab:

				st.session_state["img2var"]["editor_image"] = st.empty()

				st.form_submit_button("Refresh")

				masked_image_holder = st.empty()
				image_holder = st.empty()

				uploaded_images = st.file_uploader(
					"Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg"],
					help="Upload an image which will be used for the image to image generation.",
				)
				if uploaded_images:
					image = Image.open(uploaded_images).convert('RGB')
					new_img = image.resize((int(st.session_state["img2var"]["width"] / 2), int(st.session_state["img2var"]["height"] / 2)))
					image_holder.image(new_img)


			with result_tab:

				placeholder = st.empty()

				#populate the 3 images per column

				#print(type(st.session_state["node_preview_img_object"]))
				with placeholder.container():
					image_results = st.container()
			
				# ---------------------------------------------------------

				if len(img2var_latestimages) > 0:

					gallery_html = get_gallery(images, st.session_state["img2var"]["width"], st.session_state["img2var"]["height"])

					print(gallery_html)
					result_tab.markdown(gallery_html, unsafe_allow_html=False)

				
				
			"""
						# create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
			with col3_img2var_layout:
				preview_image = st.empty()
				st.session_state["img2var"]["preview_image"] = preview_image

				# st.session_state["loading"] = st.empty()

				st.session_state["img2var"]["progress_bar_text"] = st.empty()
				st.session_state["img2var"]["progress_bar"] = st.empty()

				message = st.empty()
			
			"""


		with col3_img2var_layout:


			st.session_state["img2var"]["pathmode"] = st.selectbox('Path Structure', ("subfolders", "root"),
																   index=st.session_state[
																	   'defaults'].general.default_path_mode_index,
																   help="subfolders structure will create daily folders plus many subfolders, root will use your outdir as root",
																   key='pathmode-img2var')
			st.session_state["img2var"]["outdir"] = st.text_input("Output Folder",
																  value=st.session_state['defaults'].general.outdir,
																  help=" Output folder", key='outdir-img2var')

			st.session_state["img2var"]["bg_upsampling"] = st.checkbox("BG Upscaling.",
																	   value=st.session_state['defaults'].img2var.bg_upsampling,
																	   help="Save each image generated before any filter or enhancement is applied.")


			if st.session_state["GFPGAN_available"]:
				st.session_state["img2var"]['use_GFPGAN'] = st.checkbox("Use GFPGAN", value=st.session_state['defaults'].img2var.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation.\
								This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
			else:
				st.session_state["img2var"]['use_GFPGAN'] = False

			st.session_state["img2var"]["cfg_scale"] = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0,
																 value=st.session_state['defaults'].img2var.cfg_scale, step=0.5,
																 help="How strongly the image should follow the prompt.")




	# if uploaded_images:
	# image = Image.open(uploaded_images).convert('RGB')
	##img_array = np.array(image) # if you want to pass it to OpenCV
	# new_img = image.resize((width, height))
	# st.image(new_img, use_column_width=True)

	if generate_button:
		# print("Loading models")
		# load the models when we hit the generate button for the first time, it wont be loaded after that so dont worry.

		st.session_state["img2var"]["init_info"] = None
		st.session_state[st.session_state['generation_mode']]['prompt_matrix'] = None

		if uploaded_images:
			image = Image.open(uploaded_images).convert('RGB')
			st.session_state["img2var"]["new_img"] = image.resize((st.session_state["img2var"]["width"], st.session_state["img2var"]["height"]))
			# img_array = np.array(image) # if you want to pass it to OpenCV

			try:
				img2var_latestimages = get_variations(input_im=new_img,
										outdir = st.session_state["img2var"]["outdir"],
										var_samples = st.session_state["img2var"]["var_samples"] ,
										var_plms = st.session_state["img2var"]["sampler_name"],
										v_cfg_scale = st.session_state["img2var"]["cfg_scale"],
										v_steps = st.session_state["img2var"]["steps"],
										v_W = st.session_state["img2var"]["width"],
										v_H = st.session_state["img2var"]["height"],
										v_ddim_eta = st.session_state["img2var"]["ddim_eta"],
										v_GFPGAN = st.session_state["img2var"]['use_GFPGAN'],
										v_bg_upsampling = st.session_state["img2var"]["bg_upsampling"],
										v_upscale = 1 #hardcoded for now
										)

				print(img2var_latestimages)

				if len(img2var_latestimages) > 0:
					images = img2var_latestimages
					start = 1
					if (len(images) % 2) == 0:
						half = len(images)/2
					else:
						half = (len(images) + 0.5) / 2
					half = int(half)
					for i in img2var_latestimages:
						with image_results:
							st.write(f"Image Index: [ **{start}** ]")
							st.image(i)
							start = start + 1



				# show a message when the generation is complete.
				#message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")

			except (StopException, KeyError) as e:
				print(e)
				print(f"Received Streamlit StopException")

	# this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
	# use the current col2 first tab to show the preview_img and update it as its generated.
	# preview_image.image(output_images, width=750)

	# on import run init
