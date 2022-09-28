# base webui import and utils.

import streamlit as st

import importlib
# streamlit imports
# import streamlit_nested_layout

# streamlit components section
from st_on_hover_tabs import on_hover_tabs
from streamlit_server_state import server_state, server_state_lock
import toml


# other imports
import os


# import k_diffusion as K
from time import sleep

from omegaconf import OmegaConf
import warnings

from tools.singleton import Singleton

# end of imports
# ---------------------------------------------------------------------------------------------------------------

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass



my_singleton = Singleton()


def set_page_title(title):
	"""
    Simple function to allows us to change the title dynamically.
    Normally you can use `st.set_page_config` to change the title but it can only be used once per app.
    """

	st.sidebar.markdown(unsafe_allow_html=True, body=f"""
                            <iframe height=0 srcdoc="<script>
                            const title = window.parent.document.querySelector('title') \

                            const oldObserver = window.parent.titleObserver
                            if (oldObserver) {{
                            oldObserver.disconnect()
                            }} \

                            const newObserver = new MutationObserver(function(mutations) {{
                            const target = mutations[0].target
                            if (target.text !== '{title}') {{
                            target.text = '{title}'
                            }}
                            }}) \

                            newObserver.observe(title, {{ childList: true }})
                            window.parent.titleObserver = newObserver \

                            title.text = '{title}'
                            </script>" />
                            """)

# remove some annoying deprecation warnings that show every now and then.
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.session_state["defaults"] = OmegaConf.load("scripts/tools/config/webui_streamlit.yaml")

if (os.path.exists("scripts/tools/config/userconfig_streamlit.yaml")):
	user_defaults = OmegaConf.load("scripts/tools/config/userconfig_streamlit.yaml")
	st.session_state["defaults"] = OmegaConf.merge(st.session_state["defaults"], user_defaults)

if (os.path.exists(".streamlit/config.toml")):
	st.session_state["streamlit_config"] = toml.load(".streamlit/config.toml")
defaults = st.session_state["defaults"]

# We import sd_utils after we have our defaults loaded
from tools.sd_utils import *

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(st.session_state["defaults"].general.gpu)


# functions to load css locally OR remotely starts here. Options exist for future flexibility. Called as st.markdown with unsafe_allow_html as css injection
# TODO, maybe look into async loading the file especially for remote fetching
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
	st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def load_css(isLocal, nameOrURL):
	if (isLocal):
		local_css(nameOrURL)
	else:
		remote_css(nameOrURL)


def layout():
	st.set_page_config(page_title="Ai Pixel Dreamer", layout="wide", initial_sidebar_state="collapsed")

	with st.empty():
		# load css as an external file, function has an option to local or remote url. Potential use when running from cloud infra that might not have access to local path.
		load_css(True, 'scripts/tools/css/streamlit.main.css')
	# check if the models exist on their respective folders
	if os.path.exists(os.path.join(defaults.general.GFPGAN_dir, "GFPGANv1.3.pth")):
		st.session_state["GFPGAN_available"] = True
	else:
		st.session_state["GFPGAN_available"] = False

	if os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments", "pretrained_models",
								   f"{defaults.general.RealESRGAN_model}.pth")):
		st.session_state["RealESRGAN_available"] = True
	else:
		st.session_state["RealESRGAN_available"] = False

	#with st.sidebar:
	# we should use an expander and group things together when more options are added so the sidebar is not too messy.
	# with st.expander("Global Settings:"):
	#	st.write("Global Settings:")
	#	defaults.general.update_preview = st.checkbox("Update Image Preview", value=defaults.general.update_preview,
	#												  help="If enabled the image preview will be updated during the generation instead of at the end. You can use the Update Preview \
	#						      Frequency option bellow to customize how frequent it's updated. By default this is enabled and the frequency is set to 1 step.")
	#	defaults.general.update_preview_frequency = st.text_input("Update Image Preview Frequency",
	#															  value=defaults.general.update_preview_frequency,
	#															  help="Frequency in steps at which the the preview image is updated. By default the frequency is set to 1 step.")

	# txt2img_tab, img2img_tab, txt2video, postprocessing_tab = st.tabs(["Text-to-Image Unified", "Image-to-Image Unified", "Text-to-Video","Post-Processing"])
	# scan plugins folder for plugins and add them to the st.tabs
	plugins = {}
	for plugin in os.listdir("scripts/ui"):
		if plugin.endswith(".py"):
			# return the description of the plugin
			pluginModule = importlib.import_module(f"scripts.ui.{plugin[:-3]}")
			importlib.reload(pluginModule)
			pluginDescription = pluginModule.PluginInfo.description
			pluginPriority = pluginModule.PluginInfo.displayPriority
			pluginIsTab = pluginModule.PluginInfo.isTab
			# if the plugin is a tab, add it to the tabs
			if pluginIsTab:
				plugins[pluginDescription] = [pluginModule, pluginPriority]

	plugins = {k: v for k, v in sorted(plugins.items(), key=lambda x: x[1][1])}
	pluginTabs = st.tabs(plugins)
	#print(pluginTabs)

	increment = 0
	for k in plugins.keys():
		with pluginTabs[increment]:
			plugins[k][0].layoutFunc()
			increment += 1

	# print(plugins)
	# print(pluginTabs)
	# print(plugins)
	# sort the plugins by priority



# print(plugin)
# print(plugin.description)
# plugin.layout
	with st.sidebar:
		tabs = on_hover_tabs(tabName=['Stable Diffusion', "Textual Inversion","Model Manager","Settings"],
							 iconName=['dashboard','model_training' ,'cloud_download', 'settings'], default_choice=0)

	if tabs =='AI Pixel Dreamer':
		# set the page url and title
		st.experimental_set_query_params(page='stable-diffusion')
		set_page_title("AI Pixel Dreamer")




		#txt2img_tab, img2img_tab, txt2vid_tab, concept_library_tab = st.tabs(["Text-to-Image", "Image-to-Image",
		#																	  "Text-to-Video","Concept Library"])
		#with home_tab:
		#from home import layout
		#layout()



	#
	elif tabs == 'Model Manager':
		# set the page url and title
		st.experimental_set_query_params(page='model-manager')
		set_page_title("Model Manager - AI Pixel Dreamer")

		from ui.model_manager import layout
		layout()

	elif tabs == 'Textual Inversion':
		# set the page url and title
		st.experimental_set_query_params(page='textual-inversion')

		from ui.textual_inversion import layout
		layout()

	elif tabs == 'Settings':
		# set the page url and title
		st.experimental_set_query_params(page='settings')
		set_page_title("Settings - AI Pixel Dreamer")

		from ui.settings import layout
		layout()
if __name__ == '__main__':
	layout()
