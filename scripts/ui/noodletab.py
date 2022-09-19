# base webui import and utils.
import streamlit as st
import streamlit.components.v1 as components
#from ui.sd_utils import *
from barfi import st_barfi, barfi_schemas, Block

from barfi import Block
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

feed = Block(name='Feed')
feed.add_output()
def feed_func(self):
    self.set_interface(name='Output 1', value=4)
feed.add_compute(feed_func)

splitter = Block(name='Splitter')
splitter.add_input()
splitter.add_output()
splitter.add_output()
def splitter_func(self):
    in_1 = self.get_interface(name='Input 1')
    value = (in_1/2)
    self.set_interface(name='Output 1', value=value)
    self.set_interface(name='Output 2', value=value)
splitter.add_compute(splitter_func)

mixer = Block(name='Mixer')
mixer.add_input()
mixer.add_input()
mixer.add_output()
def mixer_func(self):
    in_1 = self.get_interface(name='Input 1')
    in_2 = self.get_interface(name='Input 2')
    value = (in_1 + in_2)
    self.set_interface(name='Output 1', value=value)
mixer.add_compute(mixer_func)

result = Block(name='Result')
result.add_input()
def result_func(self):
    in_1 = self.get_interface(name='Input 1')
result.add_compute(result_func)

textblock = Block(name='Text')
textblock.add_output()
def tx_func(self):
    self.set_interface(name='Output 1', value="This should appear")
textblock.add_compute(tx_func)

file_block = Block(name='File Selection')
file_block.add_option(name='display-option', type='display', value='Enter the path of the file to open.')
file_block.add_option(name='file-path-input', type='input')
file_block.add_output(name='File Path')
def file_block_func(self):
    file_path = self.get_option(name='file-path-input')
    self.set_interface(name='File Path', value=file_path)
file_block.add_compute(file_block_func)

import csv

select_file_block = Block(name='Select File')
select_file_block.add_option(name='display-option', type='display', value='Select the file to load data.')
select_file_block.add_option(name='select-file', type='select', items=['file_1.csv', 'file_2.csv'], value='file_1')
select_file_block.add_output(name='File Data')
def select_file_block_func(self):
    file_path = self.get_option(name='select-file')
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    self.set_interface(name='File Data', value=data[0])
select_file_block.add_compute(select_file_block_func)

load_file_block = Block(name='Load File')
load_file_block.add_option(name='display-option', type='display', value='Enter the name of the file to load its data.')
load_file_block.add_option(name='file-path-input', type='input')
load_file_block.add_output(name='File Data')
def load_file_block_func(self):
    file_path = self.get_option(name='file-path-input')
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    self.set_interface(name='File Data', value=data[0])
load_file_block.add_compute(load_file_block_func)


slider_block = Block(name='Slider')

# Add the input and output interfaces
slider_block.add_input()
slider_block.add_output()

# Add an optional display text to the block
slider_block.add_option(name='display-option', type='display', value='This is a Block with Slider option.')

# Add the interface options to the Block
slider_block.add_option(name='slider-option-1', type='slider', min=0, max=10, value=2.5)

def slider_block_func(self):
    # Implement your computation function here
    # Use the values from the input and input-options (checbox, slider, input-text..) with the
    # get_interface() and get_option() method

    # Get the value of the input interface
    input_1_value = self.get_interface(name='Input 1')

    # Get the value of the option
    slider_1_value = self.get_option(name='slider-option-1')

    # Implement your logic using the values
    # Here
    # And obtain the value to set to the output interface
    # output_1_value = ...

    # Set the value of the output interface
    output_1_value = 0
    self.set_interface(name='Output 1', value=output_1_value)

# Add the compute function to the block
slider_block.add_compute(slider_block_func)


select_block = Block(name='Select')

# Add the input and output interfaces
select_block.add_input()
select_block.add_output()

# Add an optional display text to the block
select_block.add_option(name='display-option', type='display', value='This is a Block with Select option.')

# Add the interface options to the Block
select_block.add_option(name='select-option', type='select', items=['Select A', 'Select B', 'Select C'], value='Select A')

def select_block_func(self):
    # Implement your computation function here
    # Use the values from the input and input-options (checbox, slider, input-text..) with the
    # get_interface() and get_option() method

    # Get the value of the input interface
    input_1_value = self.get_interface(name='Input 1')

    # Get the value of the option
    select_1_value = self.get_option(name='select-option-1')

    # Implement your logic using the values
    # Here
    # And obtain the value to set to the output interface
    # output_1_value = ...

    # Set the value of the output interface
    output_1_value = 0
    self.set_interface(name='Output 1', value=output_1_value)

# Add the compute function to the block
select_block.add_compute(select_block_func)



number_block = Block(name='Number')

# Add the input and output interfaces
number_block.add_input()
number_block.add_output()

# Add an optional display text to the block
number_block.add_option(name='display-option', type='display', value='This is a Block with Number option.')

# Add the interface options to the Bloc
number_block.add_option(name='number-option-1', type='number')

def number_block_func(self):
    # Implement your computation function here
    # Use the values from the input and input-options (checbox, slider, input-text..) with the
    # get_interface() and get_option() method

    # Get the value of the input interface
    input_1_value = self.get_interface(name='Input 1')

    # Get the value of the option
    number_1_value = self.get_option(name='number-option-1')

    # Implement your logic using the values
    # Here
    # And obtain the value to set to the output interface
    # output_1_value = ...

    # Set the value of the output interface
    output_1_value = 0
    self.set_interface(name='Output 1', value=output_1_value)

# Add the compute function to the block
number_block.add_compute(number_block_func)

integer_block = Block(name='Integer')

# Add the input and output interfaces
integer_block.add_input()
integer_block.add_output()

# Add an optional display text to the block
integer_block.add_option(name='display-option', type='display', value='This is a Block with Integer option.')

# Add the interface options to the Block
integer_block.add_option(name='integer-option-1', type='integer')

def integer_block_func(self):
    # Implement your computation function here
    # Use the values from the input and input-options (checbox, slider, input-text..) with the
    # get_interface() and get_option() method

    # Get the value of the input interface
    input_1_value = self.get_interface(name='Input 1')

    # Get the value of the option
    integer_1_value = self.get_option(name='integer-option-1')

    # Implement your logic using the values
    # Here
    # And obtain the value to set to the output interface
    # output_1_value = ...

    # Set the value of the output interface
    output_1_value = 0
    self.set_interface(name='Output 1', value=output_1_value)

# Add the compute function to the block
integer_block.add_compute(integer_block_func)

from sklearn import preprocessing

label_encoder_block = Block(name='Label Encoder')
label_encoder_block.add_option(name='display-option', type='display', value='Label Encode of the input data.')
label_encoder_block.add_input(name='Data')
label_encoder_block.add_output(name='Labels')
label_encoder_block.add_output(name='Labeled Data')
def label_encoder_block_func(self):
    data = self.get_interface(name='Data')
    le = preprocessing.LabelEncoder()
    le.fit(data)
    self.set_interface(name='Labels', value=le.classes_)
    self.set_interface(name='Labeled Data', value=le.transform(data))
label_encoder_block.add_compute(label_encoder_block_func)



#Upscaler Block - test
upscale_block = Block(name='Upscale')

upscale_block.add_option(name='Upscale Strength', type='slider', min=1, max=8, value=1)
upscale_block.add_option(name='Input Image', type='input')

upscale_block.add_output(name='Path')
upscale_block.add_output(name='Function')

def upscale_func(self):
    data = 'doUpscale'
    self.set_interface(name='Function', value=data)
    data = self.get_option(name='Input Image')
    self.set_interface(name='Path', value=data)
upscale_block.add_compute(upscale_func)

#Dream Block - test
dream_block = Block(name='Dream')
dream_block.add_option(name='Prompt', type='input')

dream_block.add_output(name='Prompt')
dream_block.add_output(name='Image')

def dream_func(self):
    prompt = self.get_option(name='Prompt')
    st.session_state["prompt"] = prompt
    def_runner.run_txt2img()

dream_block.add_compute(dream_func)






def layoutFunc():
    col1, col2 = st.columns([3,1], gap="medium")

    st.session_state["node_preview_image"] = st.empty
    with col1:
        load_schema = st.selectbox('Select a saved schema:', barfi_schemas())

        compute_engine = st.checkbox('Activate barfi compute engine', value=False)

        barfi_result = st_barfi(base_blocks=[dream_block, upscale_block, label_encoder_block, slider_block, number_block, integer_block, select_block, select_file_block, file_block, textblock, feed, result, mixer, splitter],
                                compute_engine=compute_engine, load_schema=load_schema)

        if barfi_result:
            print(barfi_result)
            for a in barfi_result.keys():
                if "Feed" in a:
                    print (barfi_result[a]['block'].get_interface(name='Output 1'))
                elif 'Upscale' in a:
                    print (barfi_result[a]['block'].get_interface(name='Function'))
                    print (barfi_result[a]['block'].get_interface(name='Path'))
    with col2:
        if "node_image" in st.session_state:
            st.session_state["node_preview_image"] = st.image(st.session_state["node_image"])

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




