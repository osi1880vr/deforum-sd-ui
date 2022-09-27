# base webui import and utils.
from webui_streamlit import st
import streamlit.components.v1 as components
#from ui.sd_utils import *
from barfi import st_barfi, barfi_schemas, Block
from scripts.tools.blocks import *
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




helpText = """Images fed into a Preview Node"""
helpText2 = """will exist in memory, and can"""
helpText3 = """be accessed with the getListItem tool."""
def layoutFunc():

    st.session_state["with_nodes"] = True
    if 'v' not in st.session_state:
        c = 7
        st.session_state['v'] = 3

    #with st.form("Nodes"):
    topc1, topc2 = st.columns([1,3], gap="large")
    with topc1:
        load_schema = st.selectbox("Select node graph", barfi_schemas())

    with topc2:
        compute_engine = st.checkbox('Activate barfi compute engine', value=False)
        st.session_state["node_info"] = st.empty()
        st.session_state["node_progress"] = st.empty()

    col1, col2, col3 = st.columns([6,1,1], gap="small")
    #    refresh_btn = col1.form_submit_button("Run node sequence")

    with col1:

        if compute_engine:
            barfi_result = st_barfi(base_blocks=default_blocks_category,
                                    compute_engine=compute_engine,
                                    load_schema=load_schema)



            #[st.write(index) for index in indexList]

        #print(barfi_result['Feed-1']['block'].get_interface(name='Output 1'))
        #st.write(barfi_result['Integer-1']['block'].get_interface(name='Output 1'))
        #st.write(barfi_result['Label Encoder-1']['block'].get_interface(name='Labeled Data'))
        #st.write(barfi_result)
        #print(type(barfi_result))
        #print(barfi_result.keys())
        #print(barfi_result.values())
        #for a in barfi_result.keys():
        #  print(a)
        #for a in barfi_result.values():
        #    #print(a.keys())
        #    print(a["interfaces"].keys())
        #    print(a["interfaces"].values())

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