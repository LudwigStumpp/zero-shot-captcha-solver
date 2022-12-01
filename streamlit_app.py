import glob
import os

import streamlit as st
from streamlit_image_select import image_select

from zero_shot_captcha_solver import load_clip, solve_captcha_from_path


@st.cache(allow_output_mutation=True)
def load_model():
    return load_clip()


title = "Zero Shot Captcha Solver"

st.set_page_config(
    page_title=title,
    page_icon="🖼️",
)
st.title(title)

st.write("### Select an image")

images = glob.glob("examples/*.jpg")
img = image_select(
    label=None,
    use_container_width=False,
    images=images,
    captions=[images.split(os.sep)[-1].split(".")[0] for images in images],
)

st.write("Selected image:")
st.image(img)

st.write("### Guess")
if img is not None and st.button("Solve"):
    # show loading indicator
    with st.spinner("Solving captcha..."):
        model = load_model()
        st.write(solve_captcha_from_path(img, model=model))
