# base webui import and utils.
from webui_streamlit import st
import streamlit.components.v1 as components
#from ui.sd_utils import *
from barfi import st_barfi, barfi_schemas, Block
from scripts.tools.blocks import *
from scripts.tools.img2var import *

import PIL
import sys

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
from threading import current_thread

from scripts.tools.deforum_runner import runner
def_runner = runner()

import os
from PIL import Image
import pandas as pd

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

#prompt_parts = pd.Series([a for a in 5])

class PluginInfo():
    plugname = "noodle"
    description = "aiNodes"
    isTab = True
    displayPriority = 1

def getDrawerImagesFromPath(drawer):
    if st.session_state['defaults'].general.default_path_mode == "subfolders":
        generatedImagesPath = os.path.join(st.session_state['defaults'].general.outdir, "_node_drawers", drawer)
    else:
        generatedImagesPath = f'{st.session_state["defaults"].general.outdir}/_node_drawers/{drawer}'
    os.makedirs(generatedImagesPath, exist_ok=True)
    # get all the files from the folders and subfolders
    files = []
    ext = ('jpeg', 'jpg', "png")
    # get the latest 10 images from the output folder without walking the subfolders
    for r, d, f in os.walk(generatedImagesPath):
        for file in f:
            if file.endswith(ext):
                files.append(os.path.join(r, file))
    # sort the files by date
    files.sort(reverse=True, key=os.path.getmtime)
    latest = files
    latest.reverse()

    return latest #[Image.open(f) for f in latest[:100]]

def saveDrawerImagesToPath(drawer):
    if st.session_state['defaults'].general.default_path_mode == "subfolders":
        path = os.path.join(st.session_state['defaults'].general.outdir, "_node_drawers", drawer)
    else:
        path = f'{st.session_state["defaults"].general.outdir}/_node_drawers/{drawer}'
    os.makedirs(path, exist_ok=True)
    drawer_idx = 0
    for image in st.session_state['currentImages']:
        filename = f'{drawer}_{drawer_idx:05}.png'
        fpath = os.path.join(path, filename)
        image.save(fpath)
        drawer_idx += 1
def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


helpText = """Images fed into a Preview Node"""
helpText2 = """will exist in memory, and can"""
helpText3 = """be accessed with the getListItem tool."""
def layoutFunc():

    st.session_state["with_nodes"] = True
    if 'v' not in st.session_state:
        c = 7
        st.session_state['v'] = 3

    #with st.form("Nodes"):

    #outputimgs = variations(["/content/deforum-sd-ui-colab/output/samples/00001.png"], outdir='output', var_samples=4, var_plms="k_lms", v_cfg_scale=7.5, v_steps=5, v_W=512, v_H=512, v_ddim_eta=0, v_GFPGAN=False, v_bg_upsampling=False, v_upscale=1)


    drawers = st.empty()
    with drawers.container():
        st.session_state["node_info"] = st.empty()
        st.session_state["node_progress"] = st.empty()

        #    refresh_btn = col1.form_submit_button("Run node sequence")
        with st.expander("drawers"):
            dcol1, dcol2 = st.columns([3,2], gap="medium")
            with dcol1:
                compute_engine = st.checkbox('Activate barfi compute engine', value=False)
                btn = st.button('clear cache')
                d_name = st.text_input("Drawer Name", value="testdrawer")
                s_btn = st.button('save drawer')
                l_btn = st.button('load drawer')
            with dcol2:
                if st.session_state['defaults'].general.default_path_mode == "subfolders":
                    drawerPath = os.joinpath(st.session_state['defaults'].general.outdir, "_node_drawers")
                else:
                    drawerPath = f'{st.session_state["defaults"].general.outdir}/_node_drawers/'

                drawerlist = listdirs(drawerPath)
                load_schema = st.selectbox("Select node graph", barfi_schemas())
                load_drawer = st.selectbox("Select Drawer", drawerlist)

            if btn:
                st.session_state['currentImages'] = []

            if s_btn:
                saveDrawerImagesToPath(d_name)
            if l_btn:
                images = getDrawerImagesFromPath(load_drawer)
                st.session_state["currentImages"] = []
                for i in images:
                    img = Image.open(i)
                    st.session_state["currentImages"].append(img)
    col1, col2, col3= st.columns([6,1,1], gap="small")


    with col1:

        if compute_engine:
            barfi_result = st_barfi(base_blocks=default_blocks_category,
                                    compute_engine=compute_engine,
                                    load_schema=load_schema)


    with col2:
        output = st.empty()

        placeholder = st.empty()

        #populate the 3 images per column

        #print(type(st.session_state["node_preview_img_object"]))
        with placeholder.container():

            col_cont1 = st.container()


            #print (len(st.session_state['latestImages']))
            #if 'currentImages' in st.session_state:


            #    with col_cont:
            #st.session_state["node_preview_image"] = st.empty()
            #if "node_preview_img_object" in st.session_state:
            #    st.session_state["node_preview_image"] = st.image(st.session_state["node_preview_img_object"])

            #[st.image(images[index]) for index in [0, 1, 2, 3, 4, 5] if index < len(images)]
            #        [st.image(image) for image in images]
            if 'currentImages' in st.session_state:
                images = list(st.session_state['currentImages'])
                start = 0
                if (len(images) % 2) == 0:
                    half = len(images)/2
                else:
                    half = (len(images) + 1.5) / 2
                half = int(half)
                for i in range(half):
                    with col_cont1:
                        st.write(f"Image Index: [ **{start}** ] size:{images[start].size} mode:{images[start].mode}")
                        st.image(images[start])
                        start = start + 2

    with col3:
        output = st.empty()

        placeholder = st.empty()

        #populate the 3 images per column

        #print(type(st.session_state["node_preview_img_object"]))
        with placeholder.container():

            col_cont2 = st.container()


            #print (len(st.session_state['latestImages']))
            #if 'currentImages' in st.session_state:


            #    with col_cont:
            #st.session_state["node_preview_image"] = st.empty()
            #if "node_preview_img_object" in st.session_state:
            #    st.session_state["node_preview_image"] = st.image(st.session_state["node_preview_img_object"])

            #[st.image(images[index]) for index in [0, 1, 2, 3, 4, 5] if index < len(images)]
            #        [st.image(image) for image in images]
            if 'currentImages' in st.session_state:
                images = list(st.session_state['currentImages'])
                start = 1
                if (len(images) % 2) == 0:
                    half = len(images)/2
                else:
                    half = (len(images) + 0.5) / 2
                half = int(half)
                for i in range(half):
                    with col_cont2:
                        st.write(f"Image Index: [ **{start}** ] size:{images[start].size} mode:{images[start].mode}")
                        st.image(images[start])
                        start = start + 2

def createHTMLGallery():
    list1 = []
    for i in range(1):
        list1.append(st.text_input(f'test{i}', key='my unique keystring', placeholder="alwaysFindMe"))
    return list1