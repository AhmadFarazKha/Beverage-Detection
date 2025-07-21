import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# Load environment variables (optional for cloud APIs)
load_dotenv()

# Initialize YOLOv8 model (use trained model if available, else pre-trained)
model_path = 'runs/detect/train/weights/best.pt' if os.path.exists('runs/detect/train/weights/best.pt') else 'yolov8n.pt'
model = YOLO(model_path)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Error: Could not access webcam. Try a different index (e.g., 1).')
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Filter and display detected classes
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            print(f'Detected: {class_name}')

    # Render results on frame
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow('Beverage Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()