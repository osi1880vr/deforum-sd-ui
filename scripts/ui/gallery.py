# base webui import and utils.
import streamlit as st
from tools.sd_utils import *

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
	plugname = "gallery"
	description = "Image Gallery"
	isTab = True
	displayPriority = 4


def getLatestGeneratedImagesFromPath():
	# get the latest images from the generated images folder
	# get the path to the generated images folder
	# generatedImagesPath = os.path.join(os.getcwd(), st.session_state['defaults'].general.sd_concepts_library_folder)
	# test path till we have defaults
	if st.session_state['defaults'].general.default_path_mode == "subfolders":
		generatedImagesPath = st.session_state['defaults'].general.outdir
	else:
		generatedImagesPath = f'{st.session_state["defaults"].general.outdir}/_batch_images'
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

	# reverse the list so the latest images are first and truncate to
	# a reasonable number of images, 10 pages worth
	return latest #[Image.open(f) for f in latest[:100]]


def layoutFunc():

	latestImages = getLatestGeneratedImagesFromPath()
	st.session_state['latestImages'] = latestImages

	# image gallery
	# Number of entries per screen
	gallery_N = 9
	if not "galleryPage" in st.session_state:
		st.session_state["galleryPage"] = 0
	gallery_last_page = len(latestImages) // gallery_N

	# Add a next button and a previous button

	placeholder2 = st.empty()

	with placeholder2.container():
		gallery_prev, gallery_minus_10, gallery_refresh, gallery_next, gallery_plus_10, empty_space = st.columns(6)
		bcol1_cont = st.container()
		bcol2_cont = st.container()
		bcol3_cont = st.container()
		bcol4_cont = st.container()
		bcol5_cont = st.container()
		# the pagination work for now so its better to enable the buttons.
		with gallery_minus_10:
			with bcol1_cont:
				if gallery_prev.button("-90", key="-90"):
					if st.session_state["galleryPage"] - 10 < 0:
						st.session_state["galleryPage"] = gallery_last_page
					else:
						st.session_state["galleryPage"] -= 10

		with gallery_prev:
			with bcol2_cont:
				if gallery_prev.button("Previous", key="Previous"):

					if st.session_state["galleryPage"] - 1 < 0:
						st.session_state["galleryPage"] = gallery_last_page
					else:
						st.session_state["galleryPage"] -= 1

		with gallery_refresh:
			with bcol3_cont:
				if gallery_refresh.button("Refresh", key="Refresh"):
					st.session_state["galleryPage"] = 0

		with gallery_next:
			with bcol4_cont:
				if gallery_next.button("Next", key="Next"):

					if st.session_state["galleryPage"] + 1 > gallery_last_page:
						st.session_state["galleryPage"] = 0
					else:
						st.session_state["galleryPage"] += 1

		with gallery_plus_10:
			with bcol5_cont:
				if gallery_next.button("+90", key="+90"):

					if st.session_state["galleryPage"] + 10 > gallery_last_page:
						if st.session_state["galleryPage"] ==  gallery_last_page:
							st.session_state["galleryPage"] = 0
						else:
							st.session_state["galleryPage"] = gallery_last_page
					else:
						st.session_state["galleryPage"] += 10






		# print(st.session_state["galleryPage"])
	# Get start and end indices of the next page of the dataframe
	gallery_start_idx = st.session_state["galleryPage"] * gallery_N
	gallery_end_idx = (1 + st.session_state["galleryPage"]) * gallery_N

	# ---------------------------------------------------------


	images = list(reversed(st.session_state['latestImages']))[gallery_start_idx:(gallery_start_idx + gallery_N)]

	placeholder =  st.empty()
	placeholder1 = st.empty()


	with placeholder.container():
		lcol1, lcol2, lcol3, lcol4, lcol5, lcol6, lcol7,lcol8,lcol9 = st.columns(9)
		lcol1_cont = st.container()
		lcol2_cont = st.container()
		lcol3_cont = st.container()
		lcol4_cont = st.container()
		lcol5_cont = st.container()
		lcol6_cont = st.container()
		lcol7_cont = st.container()
		lcol8_cont = st.container()
		lcol9_cont = st.container()

		with lcol1_cont:
			with lcol1:
				[st.image(images[index], width=90) for index in [0] if index < len(images)]
		with lcol2_cont:
			with lcol2:
				[st.image(images[index], width=90) for index in [1] if index < len(images)]
		with lcol3_cont:
			with lcol3:
				[st.image(images[index], width=90) for index in [2] if index < len(images)]
		with lcol4_cont:
			with lcol4:
				[st.image(images[index], width=90) for index in [3] if index < len(images)]
		with lcol5_cont:
			with lcol5:
				[st.image(images[index], width=90) for index in [4] if index < len(images)]
		with lcol6_cont:
			with lcol6:
				[st.image(images[index], width=90) for index in [5] if index < len(images)]
		with lcol7_cont:
			with lcol7:
				[st.image(images[index], width=90) for index in [6] if index < len(images)]
		with lcol8_cont:
			with lcol8:
				[st.image(images[index], width=90) for index in [7] if index < len(images)]
		with lcol9_cont:
			with lcol9:
				[st.image(images[index], width=90) for index in [8] if index < len(images)]




	# populate the 3 images per column
	with placeholder1.container():
		col1, col2, col3 = st.columns(3)
		col1_cont = st.container()
		col2_cont = st.container()
		col3_cont = st.container()

		with col1_cont:
			with col1:
				[st.image(images[index]) for index in [0, 3, 6] if index < len(images)]
		with col2_cont:
			with col2:
				[st.image(images[index]) for index in [1, 4, 7] if index < len(images)]
		with col3_cont:
			with col3:
				[st.image(images[index]) for index in [2, 5, 8] if index < len(images)]