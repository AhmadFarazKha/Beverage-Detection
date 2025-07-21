import streamlit as st
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
import os

st.title('Beverage Detection System')

load_dotenv()
# Use trained model if available, else fall back to pre-trained
model_path = 'runs/detect/train/weights/best.pt' if os.path.exists('runs/detect/train/weights/best.pt') else 'yolov8n.pt'
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if 'running' not in st.session_state:
    st.session_state.running = False

if st.button('Start Detection') and not st.session_state.running:
    if not cap.isOpened():
        st.error('Error: Could not access webcam. Try a different index.')
    else:
        st.session_state.running = True
        stframe = st.image([])
        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error('Error: Could not read webcam frame.')
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels='BGR')
            if st.button('Stop Detection'):
                st.session_state.running = False
        cap.release()