import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

def signaturefunk():
    canvas_result = st_canvas(
        fill_color="#eee",
        stroke_width=5,
        stroke_color="black",
        background_color="white",
        update_streamlit=False,
        height=200,
        width=700,
        drawing_mode="freedraw",
    )

