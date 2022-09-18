# base webui import and utils.
from webui_streamlit import st
import streamlit.components.v1 as components
from ui.sd_utils import *

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

class plugin_info():
    plugname = "noodletab"
    description = "Noodle-Lab"
    isTab = True
    displayPriority = 0

with open('./scripts/ui/processChart/html/flowchart.html', 'r', encoding="utf-8") as f:
    components.html3 = f.read()


def layout():

    #list2 = createHTMLGallery()
    st.session_state["test"] = st.text_input("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.")
    st.session_state["findme"] = st.text_input("Input Text","", placeholder="alwaysFindMe", key='alwaysFindMe')
    #st.markdown(list2)
    st.components.v1.html(components.html3, width=1600, height=1280, scrolling=False)

def createHTMLGallery():
    list1 = []
    for i in range(1):
        list1.append(st.text_input(f'test{i}', key='my unique keystring', placeholder="alwaysFindMe"))
    return list1

