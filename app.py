
import io
import time

import streamlit as st
import pandas as pd
from PIL import Image

from detector import load_model, detect_people


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Campus AI",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 750;
        color: #172b4d;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #63738a;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e3e9f2;
    }

    div.stButton > button[kind="primary"] {
        background-color: #3267e9;
        border-radius: 8px;
        font-weight: 600;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load model once
# --------------------------------------------------
@st.cache_resource
def get_model():
    return load_model()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("🏫 Smart Campus")
    st.caption("AI People Counting System")

    st.divider()

    st.subheader("Navigation")
    st.markdown("🏠 **People Counting**")
    st.markdown("🖼️ Image Analysis")
    st.markdown("📊 Detection Results")

    st.divider()

    st.success("AI detection system")

    st.caption("Powered by YOLO and OpenCV")


# --------------------------------------------------
# Main header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">People Counting Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image to detect and count people using AI.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Image upload and settings
# --------------------------------------------------
left, right = st.columns([2, 1])

with left:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Choose a JPG or PNG image containing people."
    )

with right:
    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Higher values require more confident detections."
    )


# --------------------------------------------------
# Process uploaded image
# --------------------------------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.divider()

        original_col, result_col = st.columns(2)

        with original_col:
            st.subheader("Original Image")
            st.image(
                image,
                use_container_width=True
            )

        with result_col:
            st.subheader("AI Detection Output")

            if st.button(
                "🔍 Count People",
                type="primary",
                use_container_width=True
            ):

                try:
                    with st.spinner(
                        "AI is detecting people in your image..."
                    ):
                        model = get_model()

                        start_time = time.perf_counter()

                        count, annotated_image, detections = (
                            detect_people(
                                model,
                                image,
                                confidence
                            )
                        )

                        elapsed = (
                            time.perf_counter() - start_time
                        )

                    # Store results so they remain visible after reruns.
                    st.session_state["count"] = count
                    st.session_state["annotated_image"] = (
                        annotated_image
                    )
                    st.session_state["detections"] = detections
                    st.session_state["elapsed"] = elapsed
                    st.session_state["source_name"] = (
                        uploaded_file.name
                    )
                    st.session_state["source_image"] = image.copy()

                except Exception as error:
                    st.error(
                        "Detection failed. Please check the "
                        "terminal for details."
                    )
                    st.exception(error)

            # Display results only for the currently uploaded image.
            current_name = uploaded_file.name
            previous_name = st.session_state.get("source_name")

            if (
                previous_name == current_name
                and "annotated_image" in st.session_state
            ):
                st.image(
                    st.session_state["annotated_image"],
                    use_container_width=True
                )
            else:
                st.info(
                    "Click Count People to display the AI result."
                )


    except Exception as error:
        st.error(
            "Unable to read this image. Please upload a valid "
            "JPG or PNG file."
        )
        st.exception(error)


# --------------------------------------------------
# Results section
# --------------------------------------------------
if (
    uploaded_file is not None
    and st.session_state.get("source_name") == uploaded_file.name
    and "count" in st.session_state
):

    st.divider()
    st.subheader("Detection Summary")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "People Detected",
            st.session_state["count"]
        )

    with metric2:
        st.metric(
            "Inference Time",
            f'{st.session_state["elapsed"]:.2f} sec'
        )

    with metric3:
        st.metric(
            "Image Width",
            f'{st.session_state["source_image"].width} px'
        )

    st.subheader("Detection Details")

    detections = st.session_state["detections"]

    if detections:
        df = pd.DataFrame(detections)

        df["confidence"] = df["confidence"].map(
            lambda value: f"{value:.1%}"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No people were detected in this image.")

    # Download annotated image
    buffer = io.BytesIO()

    Image.fromarray(
        st.session_state["annotated_image"]
    ).save(buffer, format="PNG")

    st.download_button(
        label="⬇️ Download Annotated Image",
        data=buffer.getvalue(),
        file_name="smart_campus_detection.png",
        mime="image/png"
    )

    st.caption(
        "Note: detected people are model predictions. "
        "Occlusion, lighting, image quality, and overlapping "
        "people can affect counting accuracy."
    )

else:
    if uploaded_file is None:
        st.info(
            "Upload a JPG or PNG image above to get started."
        )
