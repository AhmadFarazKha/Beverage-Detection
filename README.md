# Beverage-Detection

A Python-based computer vision system to detect and classify beverages in real-time using YOLOv8 and OpenCV.


# Beverage Detection System

A Python-based computer vision application to detect and classify beverages (e.g., water bottles, soda cans, juice cartons) in real-time using YOLOv8 and OpenCV.

## Setup

1. Clone the repository using GitHub Desktop or:

   ```bash

   git clone https://github.com/your-username/Beverage-Detection.git

   ```
2. Create a virtual environment:

   ```powershell

   python -m venv venv

   .\venv\Scripts\Activate.ps1

   ```
3. Install dependencies:

   ```powershell

   pip install -r requirements.txt

   ```
4. Download a beverage dataset from Roboflow in YOLOv8 format and extract to `dataset/`.
5. Train the model:

   ```powershell

   yolo train model=yolov8n.pt data=beverage_dataset.yaml epochs=50 imgsz=640

   ```
6. Run the detection script:

   ```powershell

   python main.py

   ```

   Or run the Streamlit app:

   ```powershell

   streamlit run streamlit_app.py

   ```

## Requirements

- Python 3.8+
- YOLOv8 (ultralytics)
- OpenCV
- PyTorch
- python-dotenv
- roboflow
- streamlit

## Usage

- Run `main.py` for webcam-based detection or `streamlit_app.py` for a web interface.
- Press 'q' (main.py) or 'Stop Detection' (Streamlit) to quit.

## Dataset

- Download a beverage dataset from Roboflow (e.g., 'Beverage Detection').
- Extract to `dataset/` with `train`, `valid`, and `test` folders.
- Update `beverage_dataset.yaml` with correct class names from your dataset.

## Notes

- Ensure your webcam is connected (default device index: 0).
- For GPU support, install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`.
- If webcam fails, try `cv2.VideoCapture(1)` in `main.py` or `streamlit_app.py`.

## Troubleshooting

- If `ValueError: signal only works in main thread` occurs, ensure `streamlit_app.py` is run with `streamlit run` and avoid Flask `app.run()`.
- Check `dataset/` folder for images and labels.
- Verify `beverage_dataset.yaml` paths match your dataset structure.

## Future Enhancements

- Deploy to AWS or Flask for a web-based interface.
- Add more beverage classes to the dataset.
