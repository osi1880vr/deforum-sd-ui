# base webui import and utils.
from webui_streamlit import st
from ui.sd_utils import *

from ui.deforum_runner import runner

# streamlit imports
from streamlit import StopException
from streamlit.runtime.in_memory_file_manager import in_memory_file_manager
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


class plugin_info():
    plugname = "txt2img"
    description = "Text to Image"
    isTab = True
    displayPriority = 1

if os.path.exists(os.path.join(st.session_state['defaults'].general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
    GFPGAN_available = True
else:
    GFPGAN_available = False

if os.path.exists(os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, "experiments","pretrained_models", f"{st.session_state['defaults'].txt2vid.RealESRGAN_model}.pth")):
    RealESRGAN_available = True
else:
    RealESRGAN_available = False
#
def layout():
    def_runner = runner()
    with st.form("txt2vid-inputs"):
        st.session_state["generation_mode"] = "txt2vid"

        input_col1, generate_col1 = st.columns([10,1])
        with input_col1:
            #prompt = st.text_area("Input Text","")
            st.session_state["prompt"] = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")

            # Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
            generate_col1.write("")
            generate_col1.write("")
            generate_button = generate_col1.form_submit_button("Generate")

        # creating the page layout using columns
        col1, col2, col3 = st.columns([1,2,1], gap="large")
        with col1:
            st.session_state["max_frames"] = st.slider("Max Frames:", min_value=1, max_value=2048, value=st.session_state['defaults'].txt2vid.max_frames, step=1)

            st.session_state["W"] = st.slider("Width:", min_value=64, max_value=2048, value=st.session_state['defaults'].txt2vid.W, step=64)
            st.session_state["H"] = st.slider("Height:", min_value=64, max_value=2048, value=st.session_state['defaults'].txt2vid.H, step=64)
            st.session_state["scale"] = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=st.session_state['defaults'].txt2vid.scale, step=1e-1,format="%.1f", help="How strongly the image should follow the prompt.")

            st.session_state["seed"]  = st.text_input("Seed:", value=st.session_state['defaults'].txt2vid.seed, help=" The seed to use, if left blank a random seed will be generated.")

        with col2:
            preview_tab, gallery_tab = st.tabs(["Preview", "Gallery"])

            with preview_tab:
                #st.write("Image")
                #Image for testing
                #image = Image.open(requests.get("https://icon-library.com/images/image-placeholder-icon/image-placeholder-icon-13.jpg", stream=True).raw).convert('RGB')
                #new_image = image.resize((175, 240))
                #preview_image = st.image(image)

                # create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
                st.session_state["preview_image"] = st.empty()

                st.session_state["loading"] = st.empty()

                st.session_state["progress_bar_text"] = st.empty()
                st.session_state["progress_bar"] = st.empty()

                #generate_video = st.empty()
                st.session_state["preview_video"] = st.empty()

                message = st.empty()

        with col3:
            # If we have custom models available on the "models/custom"
            #folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
            #if st.session_state["CustomModel_available"]:
            #    custom_model = st.selectbox("Custom Model:", st.session_state["defaults"].txt2vid.custom_models_list,
            #                                index=st.session_state["defaults"].txt2vid.custom_models_list.index(st.session_state["defaults"].txt2vid.default_model),
            #                                help="Select the model you want to use. This option is only available if you have custom models \
            #                        on your 'models/custom' folder. The model name that will be shown here is the same as the name\
            #                        the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
            #                    will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")
            #else:
            #    custom_model = "CompVis/stable-diffusion-v1-4"

            #st.session_state["weights_path"] = custom_model
            #else:
            #custom_model = "CompVis/stable-diffusion-v1-4"
            #st.session_state["weights_path"] = f"CompVis/{slugify(custom_model.lower())}"

            st.session_state["steps"] = st.number_input('Sample Steps', value=st.session_state['defaults'].txt2vid.steps,step=1)
            st.session_state["sampler"] = st.selectbox(
                'Sampler',
                ("ddim", "plms","klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral" ), help="DDIM and PLMS are for quick results, you can use low sample steps. for the rest go up with the steps maybe start at 50 and raise from there")

            st.session_state["sampling_mode"] = st.selectbox(
                'Sampling Mode',
                ('bicubic', 'bilinear', 'nearest'))

            #basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

            #with basic_tab:
            #summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
            #help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")


            with st.expander("Animation"):
                st.session_state["animation_mode"] = st.selectbox(
                    'Animation Mode',
                    ('2D', '3D'))
                st.session_state["border"] = st.selectbox(
                    'Border',
                    ('wrap', 'replicate'))
                st.session_state["angle"] = st.text_input("Angle:", value=st.session_state['defaults'].txt2vid.angle)
                st.session_state["zoom"] = st.text_input("Zoom:", value=st.session_state['defaults'].txt2vid.zoom)
                st.session_state["translation_x"] = st.text_input("X Translation:", value=st.session_state['defaults'].txt2vid.translation_x)
                st.session_state["translation_y"] = st.text_input("Y Translation:", value=st.session_state['defaults'].txt2vid.translation_y)
                st.session_state["translation_z"] = st.text_input("Z Translation:", value=st.session_state['defaults'].txt2vid.translation_z)
                st.session_state["flip_2d_perspective"] = st.checkbox('Flip 2d Perspective', value=False)
                st.session_state["perspective_flip_theta"] = st.text_input("Flip Theta:", value=st.session_state['defaults'].txt2vid.perspective_flip_theta)
                st.session_state["perspective_flip_phi"] = st.text_input("Flip Phi:", value=st.session_state['defaults'].txt2vid.perspective_flip_phi)
                st.session_state["perspective_flip_gamma"] = st.text_input("Flip Gamma:", value=st.session_state['defaults'].txt2vid.perspective_flip_gamma)
                st.session_state["perspective_flip_fv"] = st.text_input("Flip FV:", value=st.session_state['defaults'].txt2vid.perspective_flip_fv)
                st.session_state["rotation_3d_x"] = st.text_input("X 3D Rotaion:", value=st.session_state['defaults'].txt2vid.rotation_3d_x)
                st.session_state["rotation_3d_y"] = st.text_input("Y 3D Rotaion:", value=st.session_state['defaults'].txt2vid.rotation_3d_y)
                st.session_state["rotation_3d_z"] = st.text_input("Z 3D Rotaion:", value=st.session_state['defaults'].txt2vid.rotation_3d_z)
                st.session_state["noise_schedule"] = st.text_input("Noise Schedule:", value=st.session_state['defaults'].txt2vid.noise_schedule)
                st.session_state["strength_schedule"] = st.text_input("Strength Schedule:", value=st.session_state['defaults'].txt2vid.strength_schedule)
                st.session_state["contrast_schedule"] = st.text_input("Contrast Schedule:", value=st.session_state['defaults'].txt2vid.contrast_schedule)
                st.session_state["color_coherence"] = st.selectbox(
                    'Color Coherence',
                    ('Match Frame 0 LAB', 'None', 'Match Frame 0 HSV',  'Match Frame 0 RGB'))
                st.session_state["diffusion_cadence"] = st.selectbox(
                    'Diffusion Cadence',
                    ('1','2','3','4','5','6','7','8'))
                st.session_state["use_depth_warping"] = st.checkbox('Use Depth Warping', value=True)
                st.session_state["midas_weight"] = st.number_input('Midas Weight', value=st.session_state['defaults'].txt2vid.midas_weight,step=1e-1,
                                                             format="%.1f")
                st.session_state["near_plane"] = st.number_input('Near Plane', value=st.session_state['defaults'].txt2vid.near_plane,step=1)
                st.session_state["far_plane"] = st.number_input('Far Plane', value=st.session_state['defaults'].txt2vid.far_plane,step=1)
                st.session_state["fov"] = st.number_input('FOV', value=st.session_state['defaults'].txt2vid.fov,step=1)
                st.session_state["padding_mode"] = st.selectbox(
                    'Padding Mode',
                    ('border', 'reflection', 'zeros'))

                st.session_state["save_depth_maps"] = st.checkbox('Save Depth Maps', value=False)
                st.session_state["video_init_path"] = st.text_input("Video Init Path:", value=st.session_state['defaults'].txt2vid.video_init_path, help="Some Video Path")
                st.session_state["extract_nth_frame"] = st.number_input('Extract Nth Frame', value=st.session_state['defaults'].txt2vid.extract_nth_frame,step=1)
                st.session_state["interpolate_key_frames"] = st.checkbox('Interpolate Key Frames', value=False)
                st.session_state["interpolate_x_frames"] = st.number_input('Number Frames to Interpolate', value=st.session_state['defaults'].txt2vid.interpolate_x_frames,step=1)
                st.session_state["resume_from_timestring"] = st.checkbox('Resume From Timestring', value=False)
                st.session_state["resume_timestring"] = st.text_input("Resume Timestring:", value=st.session_state['defaults'].txt2vid.resume_timestring, help="Some Video Path")


            with st.expander("Single Frame"):
                st.session_state["ddim_eta"] = st.number_input('DDIM ETA', value=st.session_state['defaults'].txt2vid.ddim_eta,step=1e-1,format="%.1f")
                st.session_state["save_samples"] = st.checkbox('Save Samples', value=True)
                st.session_state["save_settings"] = st.checkbox('Save Settings', value=False)#For now
                st.session_state["display_samples"] = st.checkbox('Display Samples', value=True)
                st.session_state["filename_format"] = st.selectbox(
                    'Filename Format',
                    ("{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"))
                st.session_state["seed_behavior"] = st.selectbox(
                    'Seed Behavior',
                    ("iter","fixed","random"))
                st.session_state["make_grid"] = st.checkbox('Make Grid', value=False)
                st.session_state["grid_rows"] = st.number_input('Hight', value=st.session_state['defaults'].txt2vid.grid_rows,step=1)
                st.session_state["use_init"] = st.checkbox('Use Init', value=False)
                st.session_state["strength"] = st.number_input('Strength', value=st.session_state['defaults'].txt2vid.strength,step=1e-1,format="%.1f")
                st.session_state["strength_0_no_init"] = st.checkbox('Strength 0', value=True, help="Set the strength to 0 automatically when no init image is used")
                st.session_state["init_image"] = st.text_input("Init Image:", value=st.session_state['defaults'].txt2vid.init_image, help="The image to be used as init")
                st.session_state["use_mask"] = st.checkbox('Use Mask', value=False)
                st.session_state["use_alpha_as_mask"] = st.checkbox('Use Alpha as Mask', value=False)
                st.session_state["mask_file"] = st.text_input("Init Image:", value=st.session_state['defaults'].txt2vid.mask_file, help="The Mask to be used")
                st.session_state["invert_mask"] = st.checkbox('Invert Mask', value=False)
                st.session_state["mask_brightness_adjust"] = st.number_input('Brightness Adjust', value=st.session_state['defaults'].txt2vid.mask_brightness_adjust,step=1e-1,format="%.1f", help="Adjust the brightness of the mask")
                st.session_state["mask_contrast_adjust"] = st.number_input('Contrast Adjust', value=st.session_state['defaults'].txt2vid.mask_contrast_adjust,step=1e-1,format="%.1f", help="Adjust the contrast of the mask")


















#W, H: map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
#dynamic_threshold = None
#static_threshold = None
#n_batch = 1  # @param
#batch_name = 'time'  # @param {type:"string"}
#outdir = ''

#n_samples = 1  # doesnt do anything
#precision = 'autocast'
#C = 4
#f = 8
#prompt = ""
#timestring = ""
#init_latent = None
#init_sample = None
#init_c = None

            with st.expander("Advanced"):
                st.session_state["separate_prompts"] = st.checkbox("Create Prompt Matrix.", value=st.session_state['defaults'].txt2vid.separate_prompts,
                                                                   help="Separate multiple prompts using the `|` character, and get all combinations of them.")
                st.session_state["normalize_prompt_weights"] = st.checkbox("Normalize Prompt Weights.",
                                                                           value=st.session_state['defaults'].txt2vid.normalize_prompt_weights, help="Ensure the sum of all weights add up to 1.0")
                st.session_state["save_individual_images"] = st.checkbox("Save individual images.",
                                                                         value=st.session_state['defaults'].txt2vid.save_individual_images, help="Save each image generated before any filter or enhancement is applied.")
                st.session_state["save_video"] = st.checkbox("Save video",value=st.session_state['defaults'].txt2vid.save_video, help="Save a video with all the images generated as frames at the end of the generation.")
                st.session_state["group_by_prompt"] = st.checkbox("Group results by prompt", value=st.session_state['defaults'].txt2vid.group_by_prompt,
                                                                  help="Saves all the images with the same prompt into the same folder. When using a prompt matrix each prompt combination will have its own folder.")
                st.session_state["write_info_files"] = st.checkbox("Write Info file", value=st.session_state['defaults'].txt2vid.write_info_files,
                                                                   help="Save a file next to the image with informartion about the generation.")
                st.session_state["dynamic_preview_frequency"] = st.checkbox("Dynamic Preview Frequency", value=st.session_state['defaults'].txt2vid.dynamic_preview_frequency,
                                                                            help="This option tries to find the best value at which we can update \
                                                       the preview image during generation while minimizing the impact it has in performance. Default: True")
                st.session_state["do_loop"] = st.checkbox("Do Loop", value=st.session_state['defaults'].txt2vid.do_loop,
                                                          help="Do loop")
                st.session_state["save_as_jpg"] = st.checkbox("Save samples as jpg", value=st.session_state['defaults'].txt2vid.save_as_jpg, help="Saves the images as jpg instead of png.")

                if GFPGAN_available:
                    st.session_state["use_GFPGAN"] = st.checkbox("Use GFPGAN", value=st.session_state['defaults'].txt2vid.use_GFPGAN, help="Uses the GFPGAN model to improve faces after the generation. This greatly improve the quality and consistency of faces but uses extra VRAM. Disable if you need the extra VRAM.")
                else:
                    st.session_state["use_GFPGAN"] = False

                if RealESRGAN_available:
                    st.session_state["use_RealESRGAN"] = st.checkbox("Use RealESRGAN", value=st.session_state['defaults'].txt2vid.use_RealESRGAN,
                                                                     help="Uses the RealESRGAN model to upscale the images after the generation. This greatly improve the quality and lets you have high resolution images but uses extra VRAM. Disable if you need the extra VRAM.")
                    st.session_state["RealESRGAN_model"] = st.selectbox("RealESRGAN model", ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], index=0)
                else:
                    st.session_state["use_RealESRGAN"] = False
                    st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"

                st.session_state["variant_amount"] = st.slider("Variant Amount:", value=st.session_state['defaults'].txt2vid.variant_amount, min_value=0.0, max_value=1.0, step=0.01)
                st.session_state["variant_seed"] = st.text_input("Variant Seed:", value=st.session_state['defaults'].txt2vid.seed, help="The seed to use when generating a variant, if left blank a random seed will be generated.")
                st.session_state["beta_start"] = st.slider("Beta Start:", value=st.session_state['defaults'].txt2vid.beta_start, min_value=0.0001, max_value=0.03, step=0.0001, format="%.4f")
                st.session_state["beta_end"] = st.slider("Beta End:", value=st.session_state['defaults'].txt2vid.beta_end, min_value=0.0001, max_value=0.03, step=0.0001, format="%.4f")

        if generate_button:
            def_runner.run_batch()






