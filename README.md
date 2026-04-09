# AI Worker Monitoring & Posture Detection System

This project is an **Artificial Intelligence (Computer Vision)** based worker monitoring system designed for workplace safety and productivity analysis. Using the Ultralytics YOLOv8 Pose Estimation model, it analyzes workers' posture data and detects states such as "Working", "Slumping (Warning)", or "Sleeping" in real-time.


## Key Features

* **Perspective-Aware Tracking:** Accurately detects leaning or slumping from any camera angle using dynamic shoulder-width normalization.
* **Anti-Jitter Filtering:** Uses a temporal memory buffer to ignore split-second movements (like sneezing) and prevent false alarms.
* **Occlusion Resistance:** Continues to track posture reliably via shoulder coordinates, even when the worker's face is hidden.
* **Smart Noise Elimination:** Leverages high-confidence pose estimation to strictly filter out inanimate objects like chairs or coats.

## Technologies Used

* **Python 3.x**
* **Ultralytics YOLOv8** (Model: `yolov8x-pose.pt` - Extra Large model preferred for highest accuracy)
* **OpenCV (`cv2`)** (For video processing and visualization)
* **NumPy** (For vector mathematics and distance calculations)

## 📦 Installation

Follow these steps to run the project on your local machine:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/harunzsln/WorkerDetectionProject.git](https://github.com/harunzsln/WorkerDetectionProject.git)
    cd worker-monitoring-system
    ```

2.  Install the required dependencies:
    ```bash
    pip install ultralytics opencv-python numpy
    ```

## 🚀 Usage

1.  Place the video you want to analyze in the project root directory and rename it to `worker_test_video.mp4` (or update the `cv2.VideoCapture("...")` path in the code).
2.  Run the main Python script:
    ```bash
    python main.py
    ```
3.  Once the process is complete, the analysis results will be saved in the same directory as `analysed_video.mp4`.

## ⚙️ Configuration Settings

You can optimize the variables at the beginning of the code according to your environment:

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `MODEL_NAME` | The YOLO pose model to use. `yolov8n-pose.pt` can be used for faster performance. | `yolov8x-pose.pt` |
| `SLEEP_LIMIT` | Minimum time (in seconds) that must pass before triggering the "SLEEPING" alert. | `4.0` |
| `MIN_CONFIDENCE` | Minimum confidence score required for the model to accept an object as human (kept high to filter out chairs). | `0.55` |
| `maxlen=25` | Memory buffer size for temporal filtering. Corresponds to approximately 1 second of video feed. | `25` |

## 🧠 Algorithm Logic (How It Works)

The system utilizes core keypoints obtained from the YOLOv8 Pose model: Nose (`0`), Left Shoulder (`5`), Right Shoulder (`6`).

1.  The distance between the two shoulder points (`shoulder_width`) is calculated to establish the person's "size" on screen as a reference (Scale Invariance).
2.  The vertical distance from the nose point to the midpoint of the shoulders (`avg_shoulder_y`) is calculated.
3.  If the nose approaches the shoulder line closer than 15% of the shoulder width (`head_dist_ratio < 0.15`), it is interpreted as the person leaning forward/backward or collapsing onto the desk.
4.  This state is recorded in a buffer, and if it exceeds a specified duration (`SLEEP_LIMIT`), it is displayed on the interface as a red alert.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.