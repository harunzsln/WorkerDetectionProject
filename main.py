import cv2
import time
import numpy as np
from collections import deque
from ultralytics import YOLO

# 1. SETUP
# We use Pose model because it's better at ignoring non-human objects (like chairs)
model = YOLO("yolov8x-pose.pt")

# 2. CONFIGURATION
SLEEP_LIMIT = 4.0
MIN_CONFIDENCE = 0.55  # Increased to prevent chairs being detected as persons
# Track historical head positions for Z-axis movements
worker_history = {}  # {id: {"timer": float, "buffer": deque, "last_box": list}}

cap = cv2.VideoCapture("worker_test_videos.mp4")
w, h, fps = (
    int(cap.get(i))
    for i in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS]
)
out = cv2.VideoWriter(
    "analysed_video2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # STEP 1: TRACKING WITH HIGH CONFIDENCE
    results = model.track(frame, persist=True, verbose=False, conf=MIN_CONFIDENCE)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        keypoints_list = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().tolist()

        for box, track_id, kps, conf in zip(boxes, ids, keypoints_list, confs):
            # STEP 2: REJECT POOR DETECTIONS (Chairs/Noise)
            if conf < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = box

            # Initialization for new IDs
            if track_id not in worker_history:
                worker_history[track_id] = {"timer": None, "buffer": deque(maxlen=25)}

            is_sleeping_pose = False

            # STEP 3: OCCLUSION-RESISTANT LOGIC
            # If head (Nose:0) is hidden, we use Shoulders (5, 6) or Ears (3, 4)
            nose = kps[0]
            l_shoulder = kps[5]
            r_shoulder = kps[6]

            # Check if critical points are detected (not [0,0])
            if nose[1] > 0 and l_shoulder[1] > 0:
                avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)

                # A: Head Down (Vertical)
                head_down = nose[1] > (avg_shoulder_y - (shoulder_width * 0.1))

                # B: Lean Forward/Backward (Depth Analysis)
                # If person leans forward/backward, the apparent distance between
                # nose and shoulder line decreases significantly in perspective.
                head_dist_ratio = abs(nose[1] - avg_shoulder_y) / (
                    shoulder_width + 1e-6
                )
                leaning_extreme = (
                    head_dist_ratio < 0.15
                )  # Close to shoulder level in perspective

                is_sleeping_pose = head_down or leaning_extreme

            # STEP 4: TEMPORAL FILTERING (Prevents flickering status)
            worker_history[track_id]["buffer"].append(is_sleeping_pose)
            reliable_status = sum(worker_history[track_id]["buffer"]) > (
                len(worker_history[track_id]["buffer"]) * 0.7
            )

            if reliable_status:
                if worker_history[track_id]["timer"] is None:
                    worker_history[track_id]["timer"] = current_time

                duration = current_time - worker_history[track_id]["timer"]
                status_text = (
                    f"SLEEPING: {int(duration)}s"
                    if duration > SLEEP_LIMIT
                    else "STATUS: WARNING"
                )
                color = (0, 0, 255) if duration > SLEEP_LIMIT else (0, 165, 255)
            else:
                worker_history[track_id]["timer"] = None
                status_text = "STATUS: WORKING"
                color = (0, 255, 0)

            # DRAWING
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID:{track_id} {status_text}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    out.write(frame)

cap.release()
out.release()
