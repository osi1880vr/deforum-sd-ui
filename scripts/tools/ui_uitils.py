import streamlit as st
from streamlit_server_state import server_state, server_state_lock

import os


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


def human_readable_size(size, decimal_places=3):
    """Return a human readable size from bytes."""
    for unit in ['B','KB','MB','GB','TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"



def custom_models_available():
    #
    # Allow for custom models to be used instead of the default one,
    # an example would be Waifu-Diffusion or any other fine tune of stable diffusion
    server_state["custom_models"]:sorted = []

    for root, dirs, files in os.walk(os.path.join("models", "custom")):
        for file in files:
            if os.path.splitext(file)[1] == '.ckpt':
                server_state["custom_models"].append(os.path.splitext(file)[0])


    if len(server_state["custom_models"]) > 0:
        st.session_state["CustomModel_available"] = True
        server_state["custom_models"].append("Stable Diffusion v1.4")
    else:
        st.session_state["CustomModel_available"] = False