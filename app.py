import streamlit as st
import numpy as np
import mediapipe as mp
from pathlib import Path
from PIL import Image, ImageDraw

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Human Pose Estimation",
    page_icon="🧍",
    layout="centered"
)

st.title("🧍 Human Pose Estimation")
st.write(
    "Detect human body keypoints using a pre-trained MediaPipe model."
)


# ==================================================
# MODEL LOCATION
# ==================================================

MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_full.task"


# ==================================================
# LOAD PRE-TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "pose_landmarker_full.task was not found."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    return vision.PoseLandmarker.create_from_options(options)


# ==================================================
# SKELETON CONNECTIONS
# ==================================================

CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),
    (23, 24),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),

    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32)
]


# ==================================================
# HUMAN POSE DETECTION
# ==================================================

def detect_pose(image):

    # Convert image to RGB
    image_rgb = np.array(
        image.convert("RGB")
    )

    height, width = image_rgb.shape[:2]

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    # Load model
    landmarker = load_model()

    # Detect pose
    result = landmarker.detect(mp_image)

    # Convert image to PIL
    output = Image.fromarray(image_rgb)

    # Drawing object
    draw = ImageDraw.Draw(output)

    # Check if pose was detected
    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # ------------------------------------------
        # Draw body keypoints
        # ------------------------------------------

        for point in landmarks:

            x = int(point.x * width)
            y = int(point.y * height)

            draw.ellipse(
                (
                    x - 5,
                    y - 5,
                    x + 5,
                    y + 5
                ),
                fill="blue"
            )

        # ------------------------------------------
        # Draw skeleton
        # ------------------------------------------

        for start, end in CONNECTIONS:

            p1 = landmarks[start]
            p2 = landmarks[end]

            x1 = int(p1.x * width)
            y1 = int(p1.y * height)

            x2 = int(p2.x * width)
            y2 = int(p2.y * height)

            draw.line(
                (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                fill="green",
                width=3
            )

        return np.array(output), len(landmarks)

    return np.array(output), 0


# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# PROCESS IMAGE
# ==================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )

    if st.button("Detect Human Pose"):

        try:

            with st.spinner(
                "Detecting human pose..."
            ):

                output_image, keypoints = detect_pose(
                    image
                )

            st.subheader(
                "Pose Estimation Result"
            )

            st.image(
                output_image,
                use_container_width=True
            )

            if keypoints > 0:

                st.success(
                    "Human pose detected successfully!"
                )

                st.write(
                    "Number of keypoints detected:",
                    keypoints
                )

            else:

                st.warning(
                    "No human pose detected in the image."
                )

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )
