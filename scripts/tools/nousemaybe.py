@retry(tries=5)
def load_models(continue_prev_run = False, use_GFPGAN=False, use_RealESRGAN=False, RealESRGAN_model="RealESRGAN_x4plus",
                CustomModel_available=False, custom_model="Stable Diffusion v1.4"):
    """Load the different models. We also reuse the models that are already in memory to speed things up instead of loading them again. """

    print ("Loading models.")
    torch_gc()
    st.session_state["progress_bar_text"].text("Loading models...")

    # Generate random run ID
    # Used to link runs linked w/ continue_prev_run which is not yet implemented
    # Use URL and filesystem safe version just in case.
    st.session_state["run_id"] = base64.urlsafe_b64encode(
        os.urandom(6)
    ).decode("ascii")

    # check what models we want to use and if the they are already loaded.

    if use_GFPGAN:
        if "GFPGAN" in st.session_state:
            print("GFPGAN already loaded")
        else:
            # Load GFPGAN
            if os.path.exists(st.session_state["defaults"].general.GFPGAN_dir):
                try:
                    st.session_state["GFPGAN"] = load_GFPGAN()
                    print("Loaded GFPGAN")
                except Exception:
                    import traceback
                    print("Error loading GFPGAN:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
    else:
        if "GFPGAN" in st.session_state:
            del st.session_state["GFPGAN"]

    if use_RealESRGAN:
        if "RealESRGAN" in st.session_state and st.session_state["RealESRGAN"].model.name == RealESRGAN_model:
            print("RealESRGAN already loaded")
        else:
            #Load RealESRGAN
            try:
                # We first remove the variable in case it has something there,
                # some errors can load the model incorrectly and leave things in memory.
                del st.session_state["RealESRGAN"]
            except KeyError:
                pass

            if os.path.exists(st.session_state["defaults"].general.RealESRGAN_dir):
                # st.session_state is used for keeping the models in memory across multiple pages or runs.
                st.session_state["RealESRGAN"] = load_RealESRGAN(RealESRGAN_model)
                print("Loaded RealESRGAN with model "+ st.session_state["RealESRGAN"].model.name)

    else:
        if "RealESRGAN" in st.session_state:
            del st.session_state["RealESRGAN"]

    if "model" in st.session_state:
        if "model" in st.session_state: #and st.session_state["loaded_model"] == custom_model:
            # TODO: check if the optimized mode was changed?
            print("Model already loaded")

            return
        else:
            try:
                del st.session_state.model
                del st.session_state.modelCS
                del st.session_state.modelFS
                del st.session_state.loaded_model
            except KeyError:
                pass

    # At this point the model is either
    # is not loaded yet or have been evicted:
    # load new model into memory
    st.session_state.custom_model = custom_model

    config, device, model, modelCS, modelFS = load_sd_model(custom_model)

    st.session_state.device = device
    st.session_state.model = model
    st.session_state.modelCS = modelCS
    st.session_state.modelFS = modelFS
    st.session_state.loaded_model = custom_model

    print("Model loaded.")