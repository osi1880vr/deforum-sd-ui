# base webui import and utils.
import streamlit as st
import streamlit.components.v1 as components
#from ui.sd_utils import *
from barfi import st_barfi, barfi_schemas, Block
from scripts.tools.blocks import *


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
    description = "Noodle"
    isTab = True
    displayPriority = 5


with open('./scripts/tools/processChart/html/flowchart.html', 'r', encoding="utf-8") as f:
    components.html3 = f.read()

def layoutFunc():
    #with st.form("Nodes"):
    col1, col2 = st.columns([3,1], gap="medium")
    #    refresh_btn = col1.form_submit_button("Run node sequence")

    st.session_state["node_preview_image"] = st.empty
    with col1:

        load_schema = st.selectbox('Select a saved schema:', barfi_schemas())

        compute_engine = st.checkbox('Activate barfi compute engine', value=True)
        if compute_engine:
            barfi_result = st_barfi(base_blocks=default_blocks_category,
                                    compute_engine=compute_engine, load_schema=load_schema)
    with col2:
        placeholder = st.empty()

        #populate the 3 images per column
        with placeholder.container():
            col_cont = st.container()


            #print (len(st.session_state['latestImages']))
            images = list(reversed(st.session_state['currentImages']))

            with col_cont:

                [st.image(images[index]) for index in [0, 1, 2, 3, 4, 5] if index < len(images)]



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



def createHTMLGallery():
    list1 = []
    for i in range(1):
        list1.append(st.text_input(f'test{i}', key='my unique keystring', placeholder="alwaysFindMe"))
    return list1