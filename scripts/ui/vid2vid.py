# base webui import and utils.
import streamlit as st
#from ui.sd_utils import *

from scripts.tools.deforum_runner import runner
import tools.nsp.nsp_pantry
from tools.nsp.nsp_pantry import nsp_parse
from tools.nsp.nsp_pantry import get_nsp_keys

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


class PluginInfo():
    plugname = "vid2vid"
    description = "Video to Video"
    isTab = True
    displayPriority = 1

if os.path.exists(os.path.join(st.session_state['defaults'].general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
    GFPGAN_available = True
else:
    GFPGAN_available = False

if os.path.exists(os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, "experiments","pretrained_models", f"{st.session_state['defaults'].vid2vid.RealESRGAN_model}.pth")):
    RealESRGAN_available = True
else:
    RealESRGAN_available = False
#

import tkinter as tk
from tkinter import filedialog



def layoutFunc():
    def_runner = runner()

    #with st.form("vid2vid-inputs"):
    st.session_state["generation_mode"] = "vid2vid"

    #input_col1, generate_col1 = st.columns([10,1])
    #with input_col1:
    #prompt = st.text_area("Input Text","")

    # Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
    #generate_col1.write("")
    #generate_col1.write("")

    # creating the page layout using columns
    col1, col2, col3 = st.columns([1,2,1], gap="small")
    with col1:
        with st.expander("Noodle Soup", expanded=False):
            nsp_keys = get_nsp_keys()
            inputprompt = st.text_input('Raw Noodle', placeholder='_adj-beauty_', key='vid2vid')
            outputprompt = st.empty()
            noodle_btn = st.button(label='Cook!', key='Cook!-vid2vid')
            if noodle_btn:
                outputprompt = st.write(nsp_parse(inputprompt))

        with st.expander("Basic Settings", expanded=True):
            generate_button = st.button("Generate", key='Generate-vid2vid')

            st.session_state["prompt"] = st.text_area("Input Text","", placeholder="A corgi wearing a top hat.\nSecond Prompt", key='Input Text-vid2vid')
            st.session_state["keyframes"] = st.text_area("Keyframes","", placeholder="0\n5\n10",key='Keyframes-vid2vid')

            st.session_state["max_frames"] = st.slider("Max Frames:", min_value=1, max_value=2048, value=st.session_state['defaults'].vid2vid.max_frames, step=1, key='Max Frames-vid2vid')

            st.session_state["W"] = st.slider("Width:", min_value=64, max_value=2048, value=st.session_state['defaults'].vid2vid.W, step=64, key='Width-vid2vid')
            st.session_state["H"] = st.slider("Height:", min_value=64, max_value=2048, value=st.session_state['defaults'].vid2vid.H, step=64, key='Height-vid2vid')
            st.session_state["scale"] = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0, max_value=30.0, value=st.session_state['defaults'].vid2vid.scale, step=1e-1,format="%.1f", help="How strongly the image should follow the prompt.", key='Scale-vid2vid')

            st.session_state["seed"]  = st.text_input("Seed:", value=st.session_state['defaults'].vid2vid.seed, help=" The seed to use, if left blank a random seed will be generated.", key='Seed-vid2vid')
            st.session_state["steps"] = st.number_input('Sample Steps', value=st.session_state['defaults'].vid2vid.steps,step=1, key='Steps-vid2vid')
            st.session_state["sampler"] = st.selectbox(
                'Sampler',
                ("ddim", "plms","klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral" ),
                help="DDIM and PLMS are for quick results, you can use low sample steps. for the rest go up with the steps maybe start at 50 and raise from there",
                key='Sampler-vid2vid')

            st.session_state["sampling_mode"] = st.selectbox(
                'Sampling Mode',
                ('bicubic', 'bilinear', 'nearest'),
                key='sampling_mode-vid2vid')

        #with st.expander(""):
    with col2:
        preview_tab, sequence_tab, flip_sequence_tab = st.tabs(["Preview", "3D Animation Sequence", "2D Flip Sequence"])

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
            if "mp4_path" not in st.session_state:
                st.session_state["preview_video"] = st.empty()
            else:
                st.session_state["preview_video"] = st.video(st.session_state["mp4_path"])

            message = st.empty()
        with sequence_tab:
            #col4, col5 = st.columns([1,1], gap="medium")
            st.session_state["angle"] = st.text_input("Angle:", value=st.session_state['defaults'].vid2vid.angle, key='angle-vid2vid')
            st.session_state["zoom"] = st.text_input("Zoom:", value=st.session_state['defaults'].vid2vid.zoom, key='zoom-vid2vid')
            st.session_state["translation_x"] = st.text_input("X Translation:", value=st.session_state['defaults'].vid2vid.translation_x, key='translation_x-vid2vid')
            st.session_state["translation_y"] = st.text_input("Y Translation:", value=st.session_state['defaults'].vid2vid.translation_y, key='translation_y-vid2vid')
            st.session_state["translation_z"] = st.text_input("Z Translation:", value=st.session_state['defaults'].vid2vid.translation_z, key='translation_z-vid2vid')
            st.session_state["rotation_3d_x"] = st.text_input("X 3D Rotaion:", value=st.session_state['defaults'].vid2vid.rotation_3d_x, key='rotation_3d_x-vid2vid')
            st.session_state["rotation_3d_y"] = st.text_input("Y 3D Rotaion:", value=st.session_state['defaults'].vid2vid.rotation_3d_y, key='rotation_3d_y-vid2vid')
            st.session_state["rotation_3d_z"] = st.text_input("Z 3D Rotaion:", value=st.session_state['defaults'].vid2vid.rotation_3d_z, key='rotation_3d_z-vid2vid')
            st.session_state["noise_schedule"] = st.text_input("Noise Schedule:", value=st.session_state['defaults'].vid2vid.noise_schedule, key='noise_schedule-vid2vid')
            st.session_state["strength_schedule"] = st.text_input("Strength Schedule:", value=st.session_state['defaults'].vid2vid.strength_schedule, key='strength_schedule-vid2vid')
            st.session_state["contrast_schedule"] = st.text_input("Contrast Schedule:", value=st.session_state['defaults'].vid2vid.contrast_schedule, key='contrast_schedule-vid2vid')
        with flip_sequence_tab:
            st.session_state["flip_2d_perspective"] = st.checkbox('Flip 2d Perspective', value=False, key='flip_2d_perspective-vid2vid')
            st.session_state["perspective_flip_theta"] = st.text_input("Flip Theta:", value=st.session_state['defaults'].vid2vid.perspective_flip_theta, key='perspective_flip_theta-vid2vid')
            st.session_state["perspective_flip_phi"] = st.text_input("Flip Phi:", value=st.session_state['defaults'].vid2vid.perspective_flip_phi, key='perspective_flip_phi-vid2vid')
            st.session_state["perspective_flip_gamma"] = st.text_input("Flip Gamma:", value=st.session_state['defaults'].vid2vid.perspective_flip_gamma, key='perspective_flip_gamma-vid2vid')
            st.session_state["perspective_flip_fv"] = st.text_input("Flip FV:", value=st.session_state['defaults'].vid2vid.perspective_flip_fv, key='perspective_flip_fv-vid2vid')

    with col3:
        # If we have custom models available on the "models/custom"
        #folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
        #if st.session_state["CustomModel_available"]:
        #    custom_model = st.selectbox("Custom Model:", st.session_state["defaults"].vid2vid.custom_models_list,
        #                                index=st.session_state["defaults"].vid2vid.custom_models_list.index(st.session_state["defaults"].vid2vid.default_model),
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


        #basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

        #with basic_tab:
        #summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
        #help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")

        with st.expander("Animation", expanded=True):
            videopath = st.button('Choose Path')
            st.session_state["video_init_path"] = st.text_input("Video Init Path:", value=st.session_state['defaults'].vid2vid.video_init_path, help="Input Video Path", key='video_init_path-vid2vid')

            st.session_state["animation_mode"] = 'Video Input'
            st.session_state["border"] = st.selectbox(
                'Border',
                ('wrap', 'replicate'),
                key='border-vid2vid')
            st.session_state["color_coherence"] = st.selectbox(
                'Color Coherence',
                ('Match Frame 0 LAB', 'None', 'Match Frame 0 HSV',  'Match Frame 0 RGB'),
                key='color_coherence-vid2vid')
            st.session_state["diffusion_cadence"] = st.selectbox(
                'Diffusion Cadence',
                ('1','2','3','4','5','6','7','8'),
                key='diffusion_cadence-vid2vid')
            st.session_state["use_depth_warping"] = st.checkbox('Use Depth Warping', value=True, key='use_depth_warping-vid2vid')
            st.session_state["midas_weight"] = st.number_input('Midas Weight',
                                                               value=st.session_state['defaults'].vid2vid.midas_weight,
                                                               step=1e-1,
                                                               format="%.1f",
                                                               key='midas_weight-vid2vid')
            st.session_state["near_plane"] = st.number_input('Near Plane', value=st.session_state['defaults'].vid2vid.near_plane,step=1, key='near_plane-vid2vid')
            st.session_state["far_plane"] = st.number_input('Far Plane', value=st.session_state['defaults'].vid2vid.far_plane,step=1, key='far_plane-vid2vid')
            st.session_state["fov"] = st.number_input('FOV', value=st.session_state['defaults'].vid2vid.fov,step=1, key='fov-vid2vid')
            st.session_state["padding_mode"] = st.selectbox(
                'Padding Mode',
                ('border', 'reflection', 'zeros'),
                key='padding_mode-vid2vid')

            st.session_state["save_depth_maps"] = st.checkbox('Save Depth Maps', value=False, key='save_depth_maps-vid2vid')
            st.session_state["extract_nth_frame"] = st.number_input('Extract Nth Frame', value=st.session_state['defaults'].vid2vid.extract_nth_frame,step=1, key='extract_nth_frame-vid2vid')
            st.session_state["interpolate_key_frames"] = st.checkbox('Interpolate Key Frames', value=False, key='interpolate_key_frames-vid2vid')
            st.session_state["interpolate_x_frames"] = st.number_input('Number Frames to Interpolate', value=st.session_state['defaults'].vid2vid.interpolate_x_frames,step=1, key='interpolate_x_framesvid2vid')
            st.session_state["resume_from_timestring"] = st.checkbox('Resume From Timestring', value=False, key='resume_from_timestring-vid2vid')
            st.session_state["resume_timestring"] = st.text_input("Resume Timestring:", value=st.session_state['defaults'].vid2vid.resume_timestring, help="Some Video Path", key='resume_timestring-vid2vid')


        with st.expander("Single Frame"):
            st.session_state["ddim_eta"] = st.number_input('DDIM ETA', value=st.session_state['defaults'].vid2vid.ddim_eta,step=1e-1,format="%.1f", key='ddim_eta-vid2vid')
            st.session_state["save_samples"] = st.checkbox('Save Samples', value=True, key='save_samples-vid2vid')
            st.session_state["save_settings"] = st.checkbox('Save Settings', value=False, key='save_settings-vid2vid')#For now
            st.session_state["display_samples"] = st.checkbox('Display Samples', value=True, key='display_samples-vid2vid')
            st.session_state["filename_format"] = st.selectbox(
                'Filename Format',
                ("{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"),
                key='filename_format-vid2vid')
            st.session_state["seed_behavior"] = st.selectbox(
                'Seed Behavior',
                ("iter","fixed","random"),
                key='seed_behavior-vid2vid')
            st.session_state["make_grid"] = st.checkbox('Make Grid', value=False, key='make_grid-vid2vid')
            st.session_state["grid_rows"] = st.number_input('Hight', value=st.session_state['defaults'].vid2vid.grid_rows,step=1, key='grid_rows-vid2vid')
            st.session_state["use_init"] = st.checkbox('Use Init', value=False, key='use_init-vid2vid')
            st.session_state["strength"] = st.number_input('Strength', value=st.session_state['defaults'].vid2vid.strength,step=1e-1,format="%.1f", key='strength-vid2vid')
            st.session_state["strength_0_no_init"] = st.checkbox('Strength 0', value=True, help="Set the strength to 0 automatically when no init image is used", key='strength_0_no_init-vid2vid')
            st.session_state["init_image"] = st.text_input("Init Image:", value=st.session_state['defaults'].vid2vid.init_image, help="The image to be used as init", key='init_image-vid2vid')
            st.session_state["use_mask"] = st.checkbox('Use Mask', value=False, key='use_mask-vid2vid')
            st.session_state["use_alpha_as_mask"] = st.checkbox('Use Alpha as Mask', value=False, key='use_alpha_as_mask-vid2vid')
            st.session_state["mask_file"] = st.text_input("Init Image:", value=st.session_state['defaults'].vid2vid.mask_file, help="The Mask to be used", key='mask_file-vid2vid')
            st.session_state["invert_mask"] = st.checkbox('Invert Mask', value=False, key='invert_mask-vid2vid')
            st.session_state["mask_brightness_adjust"] = st.number_input('Brightness Adjust', value=st.session_state['defaults'].vid2vid.mask_brightness_adjust,step=1e-1,format="%.1f", help="Adjust the brightness of the mask", key='mask_brightness_adjust-vid2vid')
            st.session_state["mask_contrast_adjust"] = st.number_input('Contrast Adjust', value=st.session_state['defaults'].vid2vid.mask_contrast_adjust,step=1e-1,format="%.1f", help="Adjust the contrast of the mask", key='mask_contrast_adjust-vid2vid')

    if videopath:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        st.session_state["video_init_path"] = (file_path)

    if generate_button:
        def_runner.run_batch()







