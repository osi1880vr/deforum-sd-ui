from webui_streamlit import st
from scripts.tools.blocks import *

class PluginInfo():
    plugname = "LayoutTest"
    description = "Test"
    isTab = False
    displayPriority = 10


def layoutFunc():

    st.write("this should be one row, and im trying my best to fill this space with characters")
    st.write("")
    st.write("")
    st.write("")

    col1, col2, col3 = st.cols()

    with col1:
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
    with col2:
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
    with col3:
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
        st.write("Column1")
