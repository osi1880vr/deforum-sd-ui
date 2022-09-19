# base webui import and utils.
import streamlit as st
from tools.sd_utils import *

# streamlit imports


#other imports

# Temp imports


# end of imports
#---------------------------------------------------------------------------------------------------------------

import os
from PIL import Image

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

class PluginInfo():
    plugname = "video_gallery"
    description = "Video Library"
    isTab = True
    displayPriority = 8

def getLatestGeneratedVideosFromPath():
    #get the latest images from the generated images folder
    #get the path to the generated images folder
    #generatedImagesPath = os.path.join(os.getcwd(), st.session_state['defaults'].general.sd_concepts_library_folder)
    #test path till we have defaults
    if st.session_state['defaults'].general.default_path_mode == "subfolders":
        generatedVideosPath = st.session_state['defaults'].general.outdir
    else:
        generatedVideosPath = f'{st.session_state["defaults"].general.outdir}/_mp4s'
    #get all the files from the folders and subfolders
    files = []
    ext = ('mp4')
    #get the latest 10 images from the output folder without walking the subfolders
    for r, d, f in os.walk(generatedVideosPath):

        for file in f:
            if file.endswith(ext):
                files.append(os.path.join(r, file))
    #sort the files by date
    files.sort(reverse=True, key=os.path.getmtime)
    latest = files
    latest.reverse()

    # reverse the list so the latest images are first and truncate to
    # a reasonable number of images, 10 pages worth
    #return [Image.open(f) for f in latest]
    return [f for f in latest]

def layoutFunc():
    #st.markdown(f"<h1 style='text-align: center; color: white;'>Navigate 300+ Textual-Inversion community trained concepts</h1>", unsafe_allow_html=True)

    latestVideos = getLatestGeneratedVideosFromPath()
    st.session_state['latestVideos'] = latestVideos

    #with history_tab:
    ##---------------------------------------------------------
    ## image slideshow test
    ## Number of entries per screen
    #slideshow_N = 9
    #slideshow_page_number = 0
    #slideshow_last_page = len(latestImages) // slideshow_N

    ## Add a next button and a previous button

    #slideshow_prev, slideshow_image_col , slideshow_next = st.columns([1, 10, 1])

    #with slideshow_image_col:
    #slideshow_image = st.empty()

    #slideshow_image.image(st.session_state['latestImages'][0])

    #current_image = 0

    #if slideshow_next.button("Next", key=1):
    ##print (current_image+1)
    #current_image = current_image+1
    #slideshow_image.image(st.session_state['latestImages'][current_image+1])
    #if slideshow_prev.button("Previous", key=0):
    ##print ([current_image-1])
    #current_image = current_image-1
    #slideshow_image.image(st.session_state['latestImages'][current_image - 1])


    #---------------------------------------------------------

    # image gallery
    # Number of entries per screen
    gallery_N = 9
    if not "vgalleryPage" in st.session_state:
        st.session_state["vgalleryPage"] = 0
    gallery_last_page = len(latestVideos) // gallery_N

    # Add a next button and a previous button

    gallery_prev, gallery_refresh, gallery_pagination , gallery_next = st.columns([2, 2, 8, 1])

    # the pagination doesnt work for now so its better to disable the buttons.

    if gallery_refresh.button("Refresh", key=5):
        st.session_state["vgalleryPage"] = 0

    if gallery_next.button("Next", key=7):

        if st.session_state["vgalleryPage"] + 1 > gallery_last_page:
            st.session_state["vgalleryPage"] = 0
        else:
            st.session_state["vgalleryPage"] += 1

    if gallery_prev.button("Previous", key=6):

        if st.session_state["vgalleryPage"] - 1 < 0:
            st.session_state["vgalleryPage"] = gallery_last_page
        else:
            st.session_state["vgalleryPage"] -= 1

    #print(st.session_state["galleryPage"])
    # Get start and end indices of the next page of the dataframe
    gallery_start_idx = st.session_state["vgalleryPage"] * gallery_N
    gallery_end_idx = (1 + st.session_state["vgalleryPage"]) * gallery_N

    #---------------------------------------------------------

    placeholder = st.empty()

    #populate the 3 images per column
    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1_cont = st.container()
        col2_cont = st.container()
        col3_cont = st.container()

        #print (len(st.session_state['latestImages']))
        videos = list(reversed(st.session_state['latestVideos']))[gallery_start_idx:(gallery_start_idx+gallery_N)]

        with col1_cont:
            with col1:
                [st.video(videos[index]) for index in [0, 3, 6] if index < len(videos)]
        with col2_cont:
            with col2:
                [st.video(videos[index]) for index in [1, 4, 7] if index < len(videos)]
        with col3_cont:
            with col3:
                [st.video(videos[index]) for index in [2, 5, 8] if index < len(videos)]
