from barfi import Block
from scripts.tools.deforum_runner import runner
import streamlit as st
def_runner = runner()

#SD Custom Blocks:
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
#If an input is not connected, its value is none.
dream_block = Block(name='Dream')

dream_block.add_input(name='Prompt')
dream_block.add_input(name='Seed')
dream_block.add_input(name='CFG Scale')

dream_block.add_option(name='seedInfo', type='display', value='SEED:')
if dream_block.get_interface(name='Seed') != None:
    dream_block.add_option(name='Seed', type='input', value=dream_block.get_interface(name='Seed'))
else:
    dream_block.add_option(name='Seed', type='input')

dream_block.add_option(name='promptInfo', type='display', value='PROMPT:')
dream_block.add_option(name='Prompt', type='input')
dream_block.add_option(name='Sampler', type='select', items=["ddim", "plms", "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"], value='klms')

dream_block.add_output(name='PromptOut')
dream_block.add_output(name='ImageOut')
dream_block.add_output(name='SeedOut')

def dream_func(self):
    if self.get_interface(name='Prompt') != None:
        prompt = self.get_interface(name='Prompt')
    else:
        prompt = self.get_option(name='Prompt')
    if self.get_interface(name='Seed') != None:
        seed = self.get_interface(name='Seed')
        self.set_option(name='Seed', value='Works')
    else:
        seed = self.get_option(name='Seed')

    st.session_state["seed"] = seed
    st.session_state["prompt"] = prompt
    st.session_state['sampler'] = self.get_option(name='Sampler')
    def_runner.run_txt2img()
    self.set_interface(name='ImageOut', value=st.session_state["node_pipe"])
    self.set_interface(name='PromptOut', value=prompt)
    self.set_interface(name='SeedOut', value=seed)

dream_block.add_compute(dream_func)

#Debug Block
debug_block = Block(name='Debug')
debug_block.add_input(name='Input')
debug_block.add_output(name='Output')

def debug_func(self):
    data = self.get_interface(name='Input')
    print(f'Input Type: {type(data)}')
    print(f'Input Content: {data}')
debug_block.add_compute(debug_func)


#Duplicator_Block
dup_block = Block(name='Duplicate')

dup_block.add_input(name='Input')
dup_block.add_output(name='Output-1')
dup_block.add_output(name='Output-2')
def dup_func(self):
    data = self.get_interface(name='Input')
    self.set_interface(name='Output-1', value=data)
    self.set_interface(name='Output-2', value=data)
dup_block.add_compute(dup_func)

#Original Blocks as demo below:
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

#from sklearn import preprocessing

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










