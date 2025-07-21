from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO
import os
from dotenv import load_dotenv

app = Flask(__name__)

# HTML template for the root page
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Beverage Detection</title>
</head>
<body>
    <h1>Beverage Detection System</h1>
    <p>Click <a href="/video_feed">here</a> to start the video feed.</p>
    <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Beverage Detection Feed">
</body>
</html>
"""

# Root route to display a simple page
@app.route('/')
def index():
    return render_template_string(html_template)

# Video feed route
def gen_frames():
    load_dotenv()
    model_path = 'runs/detect/train/weights/best.pt' if os.path.exists('runs/detect/train/weights/best.pt') else 'yolov8n.pt'
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'Error: Could not access webcam.\r\n')
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)