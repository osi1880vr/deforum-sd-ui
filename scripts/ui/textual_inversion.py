
from webui_streamlit import st


# streamlit imports
from streamlit import StopException
from streamlit_tensorboard import st_tensorboard
from streamlit_server_state import server_state, server_state_lock

from scripts.tools.ui_uitils import *
from scripts.tools.ti import *

# end of imports
#---------------------------------------------------------------------------------------------------------------

class PluginInfo():
	plugname = "textual_inversion"
	description = "Textual Inversion"
	isTab = False
	displayPriority = 0
if 'ti' not in st.session_state:
	st.session_state['ti'] = {}
if 'textual_inversion' not in st.session_state:
	st.session_state['textual_inversion'] = {}

def layout():

	with st.form("ti_form"):

		set_page_title("Textual Inversion - AI Pixel Dreamer")

		config_tab, output_tab = st.tabs(["Textual Inversion Setup", "Ouput"])

		with config_tab:
			col1, col2, col3, col4, col5 = st.columns(5, gap='large')


			if 'ti' not in server_state:
				server_state['ti'] = {}

			if "attr" not in st.session_state['ti']:
				st.session_state['ti']["attr"] = {}


			with col1:
				st.session_state['ti']["attr"]["pretrained_model_name_or_path"] = st.text_input("Pretrained Model Path",
																								value=st.session_state["defaults"].textual_inversion.pretrained_model_name_or_path,
																								help="Path to pretrained model or model identifier from huggingface.co/models.")

				st.session_state['ti']["attr"]["tokenizer_name"] = st.text_input("Tokenizer Name",
																				 value=st.session_state["defaults"].textual_inversion.tokenizer_name,
																				 help="Pretrained tokenizer name or path if not the same as model_name")

				st.session_state['ti']["attr"]["train_data_dir"] = st.text_input("train_data_dir", value="", help="A folder containing the training data.")

				st.session_state['ti']["attr"]["placeholder_token"] = st.text_input("Placeholder Token", value="", help="A token to use as a placeholder for the concept.")

				st.session_state['ti']["attr"]["initializer_token"] = st.text_input("Initializer Token", value="", help="A token to use as initializer word.")

				st.session_state['ti']["attr"]["learnable_property"] = st.selectbox("Learnable Property", ["object", "style"], index=0, help="Choose between 'object' and 'style'")

				st.session_state['ti']["attr"]["repeats"] = int(st.text_input("Number of times to Repeat", value=100, help="How many times to repeat the training data."))

				with col2:
					st.session_state['ti']["attr"]["output_dir"] = st.text_input("Output Directory",
																				 value=str(os.path.join("outputs", "textual_inversion")),
																				 help="The output directory where the model predictions and checkpoints will be written.")

					st.session_state['ti']["attr"]["seed"] = seed_to_int(st.text_input("Seed", help="A seed for reproducible training."))

					st.session_state['ti']["attr"]["resolution"] = int(st.text_input("Resolution",  value=512,
																					 help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"))

					st.session_state['ti']["attr"]["center_crop"] = st.checkbox("Center Image", value=True, help="Whether to center crop images before resizing to resolution")

					st.session_state['ti']["attr"]["train_batch_size"] = int(st.text_input("Train Batch Size",  value=1, help="Batch size (per device) for the training dataloader."))

					st.session_state['ti']["attr"]["num_train_epochs"] = int(st.text_input("Number of Steps to Train",  value=100, help="Number of steps to train."))

					st.session_state['ti']["attr"]["max_train_steps"] = int(st.text_input("Max Number of Steps to Train", value=5000,
																						  help="Total number of training steps to perform.  If provided, overrides 'Number of Steps to Train'."))

					with col3:
						st.session_state['ti']["attr"]["gradient_accumulation_steps"] = int(st.text_input("Gradient Accumulation Steps",  value=1,
																										  help="Number of updates steps to accumulate before performing a backward/update pass."))

						st.session_state['ti']["attr"]["learning_rate"] = float(st.text_input("Learning Rate", value=5.0e-04,
																							  help="Initial learning rate (after the potential warmup period) to use."))

						st.session_state['ti']["attr"]["scale_lr"] = st.checkbox("Scale Learning Rate", value=True,
																				 help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")

						st.session_state['ti']["attr"]["lr_scheduler"] = st.text_input("Learning Rate Scheduler",  value="constant",
																					   help=("The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
																							 " 'constant', 'constant_with_warmup']" ))

						st.session_state['ti']["attr"]["lr_warmup_steps"] = int(st.text_input("Learning Rate Warmup Steps", value=500, help="Number of steps for the warmup in the lr scheduler."))

						st.session_state['ti']["attr"]["adam_beta1"] = float(st.text_input("Adam Beta 1",  value=0.9, help="The beta1 parameter for the Adam optimizer."))

						st.session_state['ti']["attr"]["adam_beta2"] = float(st.text_input("Adam Beta 2", value=0.999, help="The beta2 parameter for the Adam optimizer."))

						st.session_state['ti']["attr"]["adam_weight_decay"] = float(st.text_input("Adam Weight Decay",  value=1e-2, help="Weight decay to use."))

						st.session_state['ti']["attr"]["adam_epsilon"] = float(st.text_input("Adam Epsilon",  value=1e-08, help="Epsilon value for the Adam optimizer"))

						with col4:
							st.session_state['ti']["attr"]["mixed_precision"] = st.selectbox("Mixed Precision", ["no", "fp16", "bf16"], index=1,
																							 help="Whether to use mixed precision. Choose" "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
																								  "and an Nvidia Ampere GPU.")

							st.session_state['ti']["attr"]["local_rank"] = int(st.text_input("Local Rank",  value=1, help="For distributed training: local_rank"))

							st.session_state['ti']["attr"]["checkpoint_frequency"] = int(st.text_input("Checkpoint Frequency",  value=500, help="How often to save a checkpoint and sample image"))

							# stable_sample_batches is crashing when saving the samples so for now I will disable it util its fixed.
							#st.session_state['ti']["attr"]["stable_sample_batches"] = int(st.text_input("Stable Sample Batches",  value=0,
							#help="Number of fixed seed sample batches to generate per checkpoint"))

							st.session_state['ti']["attr"]["stable_sample_batches"] = 0

							st.session_state['ti']["attr"]["random_sample_batches"] = int(st.text_input("Random Sample Batches",  value=2,
																										help="Number of random seed sample batches to generate per checkpoint"))

							st.session_state['ti']["attr"]["sample_batch_size"] = int(st.text_input("Sample Batch Size",  value=1, help="Number of samples to generate per batch"))

							st.session_state['ti']["attr"]["sample_steps"] = int(st.text_input("Sample Steps",  value=100,
																							   help="Number of steps for sample generation. Higher values will result in more detailed samples, but longer runtimes."))

							st.session_state['ti']["attr"]["custom_templates"] = st.text_input("Custom Templates",  value="",
																							   help="A semicolon-delimited list of custom template to use for samples, using {} as a placeholder for the concept.")
							with col5:
								st.session_state['ti']["attr"]["resume_from"] = st.text_input(label="Resume From", help="Path to a directory to resume training from (ie, logs/token_name)")

							#st.session_state['ti']["attr"]["resume_checkpoint"] = st.file_uploader("Resume Checkpoint", type=["bin"],
							#help="Path to a specific checkpoint to resume training from (ie, logs/token_name/checkpoints/something.bin).")

							#st.session_state['ti']["attr"]["config"] = st.file_uploader("Config File",  type=["json"],
							#help="Path to a JSON configuration file containing arguments for invoking this script."
							#"If resume_from is given, its resume.json takes priority over this.")
			#				
			if "resume_from" in st.session_state['ti']["attr"]:
				if st.session_state['ti']["attr"]["resume_from"]:
					if os.path.exists(os.path.join(st.session_state['ti']['args']['resume_from'], "resume.json")):
						with open(os.path.join(st.session_state['ti']['args']['resume_from'], "resume.json"), 'rt') as f:
							st.session_state['ti']["attr"] = json.load(f)["attr"]
						#print(st.session_state['ti']["attr"])

			#elif st.session_state['ti']["attr"]["config"] is not None:
			#with open(st.session_state['ti']["attr"]["config"], 'rt') as f:
			#args = parser.parse_args(namespace=argparse.Namespace(**json.load(f)["attr"]))

			env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
			if env_local_rank != -1 and env_local_rank != st.session_state['ti']["attr"]["local_rank"]:
				st.session_state['ti']["attr"]["local_rank"] = env_local_rank

			if st.session_state['ti']["attr"]["train_data_dir"] is None:
				st.error("You must specify --train_data_dir")

			if st.session_state['ti']["attr"]["pretrained_model_name_or_path"] is None:
				st.error("You must specify --pretrained_model_name_or_path")

			if st.session_state['ti']["attr"]["placeholder_token"] is None:
				st.error("You must specify --placeholder_token")

			if st.session_state['ti']["attr"]["initializer_token"] is None:
				st.error("You must specify --initializer_token")

			if st.session_state['ti']["attr"]["output_dir"] is None:
				st.error("You must specify --output_dir")

			# add a spacer and the submit button for the form.

			st.session_state["textual_inversion"]["message"] = st.empty()
			st.session_state["textual_inversion"]["progress_bar"] = st.empty()

			st.write("---")

			submit = st.form_submit_button(help="")
			if submit:
				if "pipe" in st.session_state:
					del st.session_state["pipe"]
				if "model" in st.session_state:
					del st.session_state["model"]

				st.session_state["textual_inversion"]["message"].info("Textual Inversion Running. For more info check the progress on your console or the Ouput Tab.")

				try:
					try:
						textual_inversion()
					except RuntimeError:
						if "pipeline" in server_state["textual_inversion"]:
							del server_state['ti']["checker"]
							del server_state['ti']["unwrapped"]
							del server_state['ti']["pipeline"]

						textual_inversion()

				except StopException:
					print(f"Received Streamlit StopException")

				st.session_state["textual_inversion"]["message"].empty()

		with output_tab:
			#st.info("Under Construction. :construction_worker:")

			# Start TensorBoard
			st_tensorboard(logdir=os.path.join("outputs", "textual_inversion"), port=8888)
			
		