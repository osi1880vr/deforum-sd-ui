# base webui import and utils.
import streamlit as st
# from ui.sd_utils import *

from scripts.tools.deforum_runner import runner
from scripts.tools.nsp.nsp_pantry import parser

# streamlit imports
from streamlit import StopException
#from streamlit.runtime.in_memory_file_manager import in_memory_file_manager
from streamlit.elements import image as STImage

# other imports

import os
from io import BytesIO

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass


class PluginInfo():
	plugname = "txt2vid"
	description = "Text to Video"
	isTab = True
	displayPriority = 1


if os.path.exists(os.path.join(st.session_state['defaults'].general.GFPGAN_dir, "experiments", "pretrained_models",
							   "GFPGANv1.3.pth")):
	GFPGAN_available = True
else:
	GFPGAN_available = False

if os.path.exists(os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, "experiments", "pretrained_models",
							   f"{st.session_state['defaults'].txt2vid.RealESRGAN_model}.pth")):
	RealESRGAN_available = True
else:
	RealESRGAN_available = False


#
def layoutFunc():
	def_runner = runner()

	with st.form("txt2vid-deforum"):
		st.session_state["generation_mode"] = "txt2vid"
		st.session_state["txt2vid"] = {}

		# input_col1, generate_col1 = st.columns([10,1])
		# with input_col1:
		# prompt = st.text_area("Input Text","")

		# Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
		# generate_col1.write("")
		# generate_col1.write("")

		# creating the page layout using columns
		col1, col2, col3 = st.columns([1, 2, 1], gap="small")
		with col1:
			with st.expander("Basic Settings", expanded=True):
				generate_button = st.form_submit_button("Generate")

				st.session_state["txt2vid"]["prompt"] = st.text_area("Input Text", value='',
																	 placeholder="A corgi wearing a top hat.\nSecond Prompt")
				st.session_state["txt2vid"]["keyframes"] = st.text_area("Keyframes", "0", placeholder="0\n5\n10")

				st.session_state["txt2vid"]["max_frames"] = st.slider("Max Frames:", min_value=1, max_value=2048,
																	  value=st.session_state['defaults'].txt2vid.max_frames, step=1)

				st.session_state["txt2vid"]["W"] = st.slider("Width:", min_value=64, max_value=2048,
															 value=st.session_state['defaults'].txt2vid.W, step=64)
				st.session_state["txt2vid"]["H"] = st.slider("Height:", min_value=64, max_value=2048,
															 value=st.session_state['defaults'].txt2vid.H, step=64)
				st.session_state["txt2vid"]["scale"] = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0,
																 max_value=30.0, value=st.session_state['defaults'].txt2vid.scale,
																 step=1e-1, format="%.1f",
																 help="How strongly the image should follow the prompt.")

		# with st.expander(""):
		with col2:
			preview_tab, prompt_tab, rendering_tab, settings_tab = st.tabs(["Preview",
																			"Prompt help",
																			"Rendering",
																			"Settings"
																			])
			with prompt_tab:
				nsp = parser()
				nsp_keys = nsp.get_nsp_keys()

				inputprompt = st.multiselect('Topics', nsp_keys, key='prompts_ms_2')
				st.text_input(label="Prompt Sample", value=nsp.parse(inputprompt), key='prompt_helper')

				st.session_state["txt2vid"]["prompt_tmp"] = st.text_area("Park your samples here", value='')

			with preview_tab:
				# st.write("Image")
				# Image for testing
				# image = Image.open(requests.get("https://icon-library.com/images/image-placeholder-icon/image-placeholder-icon-13.jpg", stream=True).raw).convert('RGB')
				# new_image = image.resize((175, 240))
				# preview_image = st.image(image)

				# create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
				st.session_state["txt2vid"]["preview_image"] = st.empty()

				st.session_state["txt2vid"]["loading"] = st.empty()

				st.session_state["txt2vid"]["progress_bar_text"] = st.empty()
				st.session_state["txt2vid"]["progress_bar"] = st.empty()

				# generate_video = st.empty()
				if "mp4_path" not in st.session_state:
					st.session_state["txt2vid"]["preview_video"] = st.empty()
				else:
					st.session_state["txt2vid"]["preview_video"] = st.video(st.session_state["mp4_path"])

				message = st.empty()
			with rendering_tab:
				sampler_tab, sequence_tab, flip_sequence_tab, frames_tab = st.tabs(["Sampler",
																					"3D Animation Sequence",
																					"2D Flip Sequence",
																					"Frame Setup"
																					])

				with sampler_tab:

					st.session_state["txt2vid"]["steps"] = st.number_input('Sample Steps',
																		   value=st.session_state['defaults'].txt2vid.steps, step=1)
					st.session_state["txt2vid"]["sampler"] = st.selectbox(
						'Sampler',
						("ddim", "plms", "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"),
						help="DDIM and PLMS are for quick results, you can use low sample steps. for the rest go up with the steps maybe start at 50 and raise from there")

					st.session_state["txt2vid"]["sampling_mode"] = st.selectbox(
						'Sampling Mode',
						('bicubic', 'bilinear', 'nearest'))
					st.session_state["txt2vid"]["seed"] = st.text_input("Seed:", value=st.session_state['defaults'].txt2vid.seed,
																		help=" The seed to use, if left blank a random seed will be generated.")
					st.session_state["txt2vid"]["seed_behavior"] = st.selectbox(
						'Seed Behavior',
						("iter", "fixed", "random"))
				with sequence_tab:
					# col4, col5 = st.columns([1,1], gap="medium")
					st.session_state["txt2vid"]["angle"] = st.text_input("Angle:", value=st.session_state['defaults'].txt2vid.angle)
					st.session_state["txt2vid"]["zoom"] = st.text_input("Zoom:", value=st.session_state['defaults'].txt2vid.zoom)
					st.session_state["txt2vid"]["translation_x"] = st.text_input("X Translation:",
																				 value=st.session_state['defaults'].txt2vid.translation_x)
					st.session_state["txt2vid"]["translation_y"] = st.text_input("Y Translation:",
																				 value=st.session_state['defaults'].txt2vid.translation_y)
					st.session_state["txt2vid"]["translation_z"] = st.text_input("Z Translation:",
																				 value=st.session_state['defaults'].txt2vid.translation_z)
					st.session_state["txt2vid"]["rotation_3d_x"] = st.text_input("X 3D Rotaion:",
																				 value=st.session_state['defaults'].txt2vid.rotation_3d_x)
					st.session_state["txt2vid"]["rotation_3d_y"] = st.text_input("Y 3D Rotaion:",
																				 value=st.session_state['defaults'].txt2vid.rotation_3d_y)
					st.session_state["txt2vid"]["rotation_3d_z"] = st.text_input("Z 3D Rotaion:",
																				 value=st.session_state['defaults'].txt2vid.rotation_3d_z)
					st.session_state["txt2vid"]["noise_schedule"] = st.text_input("Noise Schedule:", value=st.session_state[
						'defaults'].txt2vid.noise_schedule)
					st.session_state["txt2vid"]["strength_schedule"] = st.text_input("Strength Schedule:", value=st.session_state[
						'defaults'].txt2vid.strength_schedule)
					st.session_state["txt2vid"]["contrast_schedule"] = st.text_input("Contrast Schedule:", value=st.session_state[
						'defaults'].txt2vid.contrast_schedule)
				with flip_sequence_tab:
					st.session_state["txt2vid"]["flip_2d_perspective"] = st.checkbox('Flip 2d Perspective', value=False)
					st.session_state["txt2vid"]["perspective_flip_theta"] = st.text_input("Flip Theta:", value=st.session_state[
						'defaults'].txt2vid.perspective_flip_theta)
					st.session_state["txt2vid"]["perspective_flip_phi"] = st.text_input("Flip Phi:", value=st.session_state[
						'defaults'].txt2vid.perspective_flip_phi)
					st.session_state["txt2vid"]["perspective_flip_gamma"] = st.text_input("Flip Gamma:", value=st.session_state[
						'defaults'].txt2vid.perspective_flip_gamma)
					st.session_state["txt2vid"]["perspective_flip_fv"] = st.text_input("Flip FV:", value=st.session_state[
						'defaults'].txt2vid.perspective_flip_fv)
				with frames_tab:
					basic_tab, mask_tab, init_tab = st.tabs(["Basics", "Mask", "Init Image"])
					with basic_tab:
						st.session_state["txt2vid"]["ddim_eta"] = st.number_input('DDIM ETA',
																				  value=st.session_state['defaults'].txt2vid.ddim_eta,
																				  step=1e-1, format="%.1f")

						st.session_state["txt2vid"]["make_grid"] = st.checkbox('Make Grid', value=False)
						st.session_state["txt2vid"]["grid_rows"] = st.number_input('Hight',
																				   value=st.session_state['defaults'].txt2vid.grid_rows,
																				   step=1)

					with mask_tab:
						st.session_state["txt2vid"]["use_mask"] = st.checkbox('Use Mask', value=False)
						st.session_state["txt2vid"]["use_alpha_as_mask"] = st.checkbox('Use Alpha as Mask', value=False)
						st.session_state["txt2vid"]["mask_file"] = st.text_input("Init Image:",
																				 value=st.session_state['defaults'].txt2vid.mask_file,
																				 help="The Mask to be used")
						st.session_state["txt2vid"]["invert_mask"] = st.checkbox('Invert Mask', value=False)
						st.session_state["txt2vid"]["mask_brightness_adjust"] = st.number_input('Brightness Adjust',
																								value=st.session_state[
																									'defaults'].txt2vid.mask_brightness_adjust,
																								step=1e-1, format="%.1f",
																								help="Adjust the brightness of the mask")
						st.session_state["txt2vid"]["mask_contrast_adjust"] = st.number_input('Contrast Adjust', value=st.session_state[
							'defaults'].txt2vid.mask_contrast_adjust, step=1e-1, format="%.1f",
																							  help="Adjust the contrast of the mask")
					with init_tab:
						st.session_state["txt2vid"]["use_init"] = st.checkbox('Use Init', value=False)
						st.session_state["txt2vid"]["strength"] = st.number_input('Strength',
																				  value=st.session_state['defaults'].txt2vid.strength,
																				  step=1e-1, format="%.1f")
						st.session_state["txt2vid"]["strength_0_no_init"] = st.checkbox('Strength 0', value=True,
																						help="Set the strength to 0 automatically when no init image is used")
						st.session_state["txt2vid"]["init_image"] = st.text_input("Init Image:",
																				  value=st.session_state['defaults'].txt2vid.init_image,
																				  help="The image to be used as init")
			with settings_tab:
				st.session_state["txt2vid"]["save_samples"] = st.checkbox('Save Samples', value=True)
				st.session_state["txt2vid"]["save_settings"] = st.checkbox('Save Settings', value=False)  # For now
				st.session_state["txt2vid"]["display_samples"] = st.checkbox('Display Samples', value=True)
				st.session_state["txt2vid"]["pathmode"] = st.selectbox('Path Structure', ("subfolders", "root"),
																	   index=st.session_state[
																		   'defaults'].general.default_path_mode_index,
																	   help="subfolders structure will create daily folders plus many subfolders, root will use your outdir as root",
																	   key='pathmode-txt2vid')
				st.session_state["txt2vid"]["outdir"] = st.text_input("Output Folder",
																	  value=st.session_state['defaults'].general.outdir,
																	  help=" Output folder", key='outdir-txt2vid')
				st.session_state["txt2vid"]["filename_format"] = st.selectbox(
					'Filename Format',
					("{timestring}_{index}_{seed}.png", "{timestring}_{index}_{prompt}.png"))

		with col3:
			# If we have custom models available on the "models/custom"
			# folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
			# if st.session_state["CustomModel_available"]:
			#    custom_model = st.selectbox("Custom Model:", st.session_state["defaults"].txt2vid.custom_models_list,
			#                                index=st.session_state["defaults"].txt2vid.custom_models_list.index(st.session_state["defaults"].txt2vid.default_model),
			#                                help="Select the model you want to use. This option is only available if you have custom models \
			#                        on your 'models/custom' folder. The model name that will be shown here is the same as the name\
			#                        the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
			#                    will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")
			# else:
			#    custom_model = "CompVis/stable-diffusion-v1-4"

			# st.session_state["weights_path"] = custom_model
			# else:
			# custom_model = "CompVis/stable-diffusion-v1-4"
			# st.session_state["weights_path"] = f"CompVis/{slugify(custom_model.lower())}"

			# basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

			# with basic_tab:
			# summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
			# help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")

			with st.expander("Animation", expanded=True):
				st.session_state["txt2vid"]["animation_mode"] = st.selectbox(
					'Animation Mode',
					('2D', '3D'))
				st.session_state["txt2vid"]["border"] = st.selectbox(
					'Border',
					('wrap', 'replicate'))
				st.session_state["txt2vid"]["color_coherence"] = st.selectbox(
					'Color Coherence',
					('Match Frame 0 LAB', 'None', 'Match Frame 0 HSV', 'Match Frame 0 RGB'))
				st.session_state["txt2vid"]["diffusion_cadence"] = st.selectbox(
					'Diffusion Cadence',
					('1', '2', '3', '4', '5', '6', '7', '8'))
				st.session_state["txt2vid"]["use_depth_warping"] = st.checkbox('Use Depth Warping', value=True)
				st.session_state["txt2vid"]["midas_weight"] = st.number_input('Midas Weight',
																			  value=st.session_state['defaults'].txt2vid.midas_weight,
																			  step=1e-1,
																			  format="%.1f")
				st.session_state["txt2vid"]["near_plane"] = st.number_input('Near Plane',
																			value=st.session_state['defaults'].txt2vid.near_plane,
																			step=1)
				st.session_state["txt2vid"]["far_plane"] = st.number_input('Far Plane',
																		   value=st.session_state['defaults'].txt2vid.far_plane,
																		   step=1)
				st.session_state["txt2vid"]["fov"] = st.number_input('FOV', value=st.session_state['defaults'].txt2vid.fov, step=1)
				st.session_state["txt2vid"]["padding_mode"] = st.selectbox(
					'Padding Mode',
					('border', 'reflection', 'zeros'))

				st.session_state["txt2vid"]["save_depth_maps"] = st.checkbox('Save Depth Maps', value=False)
				st.session_state["txt2vid"]["video_init_path"] = ''
				st.session_state["txt2vid"]["extract_nth_frame"] = st.number_input('Extract Nth Frame', value=st.session_state[
					'defaults'].txt2vid.extract_nth_frame, step=1)
				st.session_state["txt2vid"]["interpolate_key_frames"] = st.checkbox('Interpolate Key Frames', value=False)
				st.session_state["txt2vid"]["interpolate_x_frames"] = st.number_input('Number Frames to Interpolate',
																					  value=st.session_state[
																						  'defaults'].txt2vid.interpolate_x_frames,
																					  step=1)
				st.session_state["txt2vid"]["resume_from_timestring"] = st.checkbox('Resume From Timestring', value=False)
				st.session_state["txt2vid"]["resume_timestring"] = st.text_input("Resume Timestring:", value=st.session_state[
					'defaults'].txt2vid.resume_timestring, help="Some Video Path")

		if generate_button:
			st.session_state["txt2vid"]["iterations"] = 1
			st.session_state["txt2vid"]["batch_size"] = 1
		try:
			def_runner.run_batch()
		except (StopException, KeyError) as e:
			print(e)
			print(f"Received Streamlit StopException")

