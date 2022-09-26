# base webui import and utils.
import streamlit as st
# from ui.sd_utils import *

# streamlit imports


# other imports

# Temp imports 


# end of imports
# ---------------------------------------------------------------------------------------------------------------

import os
from PIL import Image

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass


class PluginInfo():
	plugname = "home"
	description = "Home"
	isTab = False
	displayPriority = 0


def getLatestGeneratedImagesFromPath():
	# get the latest images from the generated images folder
	# get the path to the generated images folder
	generatedImagesPath = os.path.join(os.getcwd(), 'outputs')
	# get all the files from the folders and subfolders
	files = []
	# get the laest 10 images from the output folder without walking the subfolders
	for r, d, f in os.walk(generatedImagesPath):
		for file in f:
			if '.png' in file:
				files.append(os.path.join(r, file))
	# sort the files by date
	files.sort(key=os.path.getmtime)

	# reverse the list so the latest images are first
	# we only want 10 images so we only open 10
	n = 0
	for f in files:
		img = Image.open(f)
		files[files.index(f)] = img
		n += 1
		if n >= 9:
			break

	# get the latest 10 files
	# get all the files with the .png or .jpg extension
	# sort files by date
	# get the latest 10 files
	latestFiles = files
	# reverse the list
	latestFiles.reverse()
	return latestFiles

def layoutFunc():
	# streamlit home page layout
	# center the title
	st.markdown("<h1 style='text-align: center; color: white;'>Welcome to AI Pixel Dreamer</h1>", unsafe_allow_html=True)
