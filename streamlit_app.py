import glob
import os

import streamlit as st
from streamlit_image_select import image_select

from zero_shot_captcha_solver import load_clip, solve_captcha_from_path


@st.cache(allow_output_mutation=True)
def load_model():
    return load_clip()


def setup_basic():
    title = "Zero Shot Captcha Solver"

    st.set_page_config(
        page_title=title,
        page_icon="🖼️",
    )
    st.title(title)

    st.markdown(
        """
        This is a demo of a zero shot captcha solver using OpenAI's [CLIP]("https://arxiv.org/abs/2103.00020") model.
        Please refer to the project's [GitHub repository](https://github.com/LudwigStumpp/zero-shot-captcha-solver)
        for more information on how this works.
        """
    )


def setup_image_select() -> str:
    st.header("1. Select a Captcha Image")

    images = glob.glob("examples/*.jpg")
    img_path = image_select(
        label=None,
        use_container_width=False,
        images=images,
        captions=[images.split(os.sep)[-1].split(".")[0] for images in images],
    )

    if img_path is not None:
        st.write("Selected image:")
        st.write(img_path.split(os.sep)[-1].split(".")[0])
        st.image(img_path)

    return img_path


def setup_solver(img_path: str):
    st.header("2. Solve Captcha")

    if img_path is not None:
        st.markdown(
            "This will try to mark the images that contain the object"
            + f"'{img_path.split(os.sep)[-1].split('.')[0]}'."
        )
        if st.button("Solve"):
            with st.spinner("Solving captcha..."):
                model = load_model()
                st.write(solve_captcha_from_path(img_path, model=model))


def setup_footer():
    st.markdown(
        """
        ---
        Made with ❤️ in Munich by [Ludwig Stumpp](https://ludwigstumpp.com).
        """
    )


def main():
    setup_basic()
    img = setup_image_select()
    setup_solver(img)
    setup_footer()


if __name__ == "__main__":
    main()
