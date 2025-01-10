#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile

def _display_detected_frames(conf, model, st_frame, image):


    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True
                   )

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_container_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_container_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


# def infer_uploaded_video(conf, model):
#     """
#     Execute inference for uploaded video
#     :param conf: Confidence of YOLOv8 model
#     :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#     :return: None
#     """
#     source_video = st.sidebar.file_uploader(
#         label="Choose a video..."
#     )

#     if source_video:
#         st.video(source_video)

#     if source_video:
#         if st.button("Execution"):
#             with st.spinner("Running..."):
#                 try:
#                     tfile = tempfile.NamedTemporaryFile()
#                     tfile.write(source_video.read())
#                     vid_cap = cv2.VideoCapture(
#                         tfile.name)
#                     st_frame = st.empty()
#                     while (vid_cap.isOpened()):
#                         success, image = vid_cap.read()
#                         if success:
#                             _display_detected_frames(conf,
#                                                      model,
#                                                      st_frame,
#                                                      image
#                                                      )
#                         else:
#                             vid_cap.release()
#                             break
#                 except Exception as e:
#                     st.error(f"Error loading video: {e}")

def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video and save detected frames with timestamps.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(label="Choose a video...")

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    # Temporary file for the video
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)

                    # Get total duration of the video
                    frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    total_duration = frame_count / fps  # Total duration in seconds
                    st.write(f"Total video duration: {total_duration:.2f} seconds")

                    # Create output directory
                    output_dir = "detected_objects"
                    os.makedirs(output_dir, exist_ok=True)

                    st_frame = st.empty()
                    frame_timestamps = []  # To store timestamps of detected frames

                    while vid_cap.isOpened():
                        success, frame = vid_cap.read()
                        if not success:
                            st.write("Video processing completed.")
                            break

                        # Get the timestamp of the current frame in seconds
                        frame_timestamp = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds

                        # Perform object detection
                        results = model.predict(frame, conf=conf)
                        detections = results[0].boxes.data.cpu().numpy()  # Bounding box tensors

                        # If objects detected, save frame with timestamp
                        if len(detections) > 0:
                            frame_timestamps.append(frame_timestamp)
                            res_plotted = results[0].plot()
                            save_path = os.path.join(output_dir, f"frame_{frame_timestamp:.2f}.png")
                            cv2.imwrite(save_path, res_plotted, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                        # Display frame with detections
                        res_plotted = results[0].plot()
                        st_frame.image(res_plotted, channels="BGR", use_container_width=True)

                    vid_cap.release()

                    # # Save timestamps to a text file
                    # with open(os.path.join(output_dir, "timestamps.txt"), "w") as f:
                    #     for timestamp in frame_timestamps:
                    #         f.write(f"{timestamp:.2f}\n")

                    # st.write(f"Detection timestamps saved to '{output_dir}/timestamps.txt'.")

                except Exception as e:
                    st.error(f"Error processing video: {e}")

def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(1)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
