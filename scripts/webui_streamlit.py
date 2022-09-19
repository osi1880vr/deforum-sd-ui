# base webui import and utils.
import streamlit as st
import importlib
# streamlit imports
#import streamlit_nested_layout

#streamlit components section
from st_on_hover_tabs import on_hover_tabs

#other imports
import os

#import k_diffusion as K
from omegaconf import OmegaConf
import warnings


# end of imports
#---------------------------------------------------------------------------------------------------------------

try:
	# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
	from transformers import logging

	logging.set_verbosity_error()
except:
	pass



# remove some annoying deprecation warnings that show every now and then.
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.session_state["defaults"] = OmegaConf.load("scripts/tools/config/webui_streamlit.yaml")
if (os.path.exists("scripts/tools/config/userconfig_streamlit.yaml")):
	user_defaults = OmegaConf.load("scripts/tools/config/userconfig_streamlit.yaml")
	st.session_state["defaults"] = OmegaConf.merge(st.session_state["defaults"], user_defaults)

defaults = st.session_state["defaults"]

#We import sd_utils after we have our defaults loaded
from tools.sd_utils import *

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(st.session_state["defaults"].general.gpu)

# functions to load css locally OR remotely starts here. Options exist for future flexibility. Called as st.markdown with unsafe_allow_html as css injection
# TODO, maybe look into async loading the file especially for remote fetching 
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
	st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def load_css(isLocal, nameOrURL):
	if(isLocal):
		local_css(nameOrURL)
	else:
		remote_css(nameOrURL)

def layout():
	
	st.set_page_config(page_title="Stable Diffusion Playground", layout="wide", initial_sidebar_state="collapsed")

	with st.empty():
		# load css as an external file, function has an option to local or remote url. Potential use when running from cloud infra that might not have access to local path.
		load_css(True, 'scripts/tools/css/streamlit.main.css')
	# check if the models exist on their respective folders
	if os.path.exists(os.path.join(defaults.general.GFPGAN_dir, "experiments", "pretrained_models", "GFPGANv1.3.pth")):
		GFPGAN_available = True
	else:
		GFPGAN_available = False

	if os.path.exists(os.path.join(defaults.general.RealESRGAN_dir, "experiments","pretrained_models", f"{defaults.general.RealESRGAN_model}.pth")):
		RealESRGAN_available = True
	else:
		RealESRGAN_available = False	

	with st.sidebar:
		# we should use an expander and group things together when more options are added so the sidebar is not too messy.
		#with st.expander("Global Settings:"):
		st.write("Global Settings:")
		defaults.general.update_preview = st.checkbox("Update Image Preview", value=defaults.general.update_preview,
                                                              help="If enabled the image preview will be updated during the generation instead of at the end. You can use the Update Preview \
							      Frequency option bellow to customize how frequent it's updated. By default this is enabled and the frequency is set to 1 step.")
		defaults.general.update_preview_frequency = st.text_input("Update Image Preview Frequency", value=defaults.general.update_preview_frequency,
                                                                          help="Frequency in steps at which the the preview image is updated. By default the frequency is set to 1 step.")


	#txt2img_tab, img2img_tab, txt2video, postprocessing_tab = st.tabs(["Text-to-Image Unified", "Image-to-Image Unified", "Text-to-Video","Post-Processing"])
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
	
	#print(plugins)
	#print(pluginTabs)
	#print(plugins)
	# sort the plugins by priority
	plugins = {k: v for k, v in sorted(plugins.items(), key=lambda x: x[1][1])}
	pluginTabs = st.tabs(plugins)
	increment = 0
	for k in plugins.keys():
		with pluginTabs[increment]:
				plugins[k][0].layoutFunc()
				increment += 1

			#print(plugin)
			# print(plugin.description)
			#plugin.layout
	
if __name__ == '__main__':
	layout()