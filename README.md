# Human Pose Estimation Using a Pre-Trained Deep Learning Model

This mini-project uses Google's MediaPipe Pose Landmarker as a pre-trained model for human pose estimation.

## Included
- `Human_Pose_Estimation.ipynb` — complete Jupyter notebook
- `requirements.txt` — required Python packages
- `images/` — put input images here
- `videos/` — put an input video here
- `models/` — the notebook downloads the pre-trained `.task` model here
- `outputs/` — processed images and videos are saved here

## How to run
1. Extract the ZIP.
2. Open Jupyter Notebook/JupyterLab in the extracted folder.
3. Open `Human_Pose_Estimation.ipynb`.
4. Run cells from top to bottom.
5. Put `person1.jpg` and optionally `person2.jpg` in `images/`.
6. Put `pose_video.mp4` in `videos/` for the video section.
7. The first run downloads the pre-trained model from Google.
8. Capture screenshots of the demonstrated outputs for submission.

## Model
MediaPipe Pose Landmarker, Pose Landmarker Full model bundle.

Official documentation:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

Official repository:
https://github.com/google-ai-edge/mediapipe

License:
Apache License 2.0 for the MediaPipe repository. Check the specific model bundle terms before redistribution.

## Important
This project performs inference only. It does not train a deep-learning model from scratch.
