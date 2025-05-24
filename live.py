import cv2
import time
import queue
import threading
from ultralytics import YOLO
import streamlink
from save_csv import save_vehicle_counts

# Constants and config
YOUTUBE_URL = "https://youtu.be/6dp-bvQ7RWo"  # Shinjuku live stream URL
FRAME_SKIP = 2  # Process every 2nd frame
FRAME_QUEUE_SIZE = 10
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
IMG_SIZE = 1280

# Vehicle class IDs mapping for YOLO
VEHICLE_CLASS_IDS = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Global flags and variables
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
stop_capture = False
total_vehicle_counts = {cls_id: 0 for cls_id in VEHICLE_CLASS_IDS}

def get_stream_url(youtube_url):
    streams = streamlink.streams(youtube_url)
    if "best" not in streams:
        raise RuntimeError("No valid stream found.")
    return streams["best"].url

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    return cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

def capture_frames(cap, queue_obj):
    global stop_capture
    while not stop_capture:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot read frame.")
            break
        try:
            queue_obj.put_nowait(frame)
        except queue.Full:
            # Skip frames if queue full to avoid lag
            pass
        time.sleep(0.02)  # ~50 FPS capture rate

def draw_label(frame, label, pos, font_scale=1, thickness=2):
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x, y - h - 5), (x + w + 5, y + 2), (0, 255, 0), -1)
    cv2.putText(frame, label, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def main():
    global stop_capture

    try:
        stream_url = get_stream_url(YOUTUBE_URL)
        print(f"Streaming from: {stream_url}")
    except Exception as e:
        print(f"Error getting stream URL: {e}")
        return

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Could not open video stream.")
        return

    # Setup capture resolution and properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Load YOLO model (change to yolov10n.pt if you want the latest)
    model = YOLO("yolov8s.pt")

    # Start capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue), daemon=True)
    capture_thread.start()

    frame_counter = 0

    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        # Preprocess frame (optional)
        processed_frame = preprocess_frame(frame)

        # YOLOv8 track inference
        results = model.track(
            processed_frame,
            persist=False,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE
        )[0]

        frame_vehicle_counts = {cls_id: 0 for cls_id in VEHICLE_CLASS_IDS}
        vehicle_indexes = {cls_id: 1 for cls_id in VEHICLE_CLASS_IDS}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            class_name = VEHICLE_CLASS_IDS[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            frame_vehicle_counts[cls_id] += 1
            total_vehicle_counts[cls_id] += 1

            index = vehicle_indexes[cls_id]
            vehicle_indexes[cls_id] += 1

            label = f"{class_name} {index}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            draw_label(frame, label, (x1, y1))

        # Draw summary info
        y_offset = 40
        total_vehicles = sum(frame_vehicle_counts.values())
        draw_label(frame, f"Current Vehicles: {total_vehicles}", (20, y_offset), font_scale=1.5, thickness=3)
        y_offset += 40

        for cls_id, count in frame_vehicle_counts.items():
            label = f"{VEHICLE_CLASS_IDS[cls_id]}s: {count}"
            draw_label(frame, label, (20, y_offset), font_scale=1, thickness=2)
            y_offset += 30

        cv2.imshow("Vehicle Counter (YouTube Live)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
            print("Exit requested by user.")
            break

    # Cleanup
    stop_capture = True
    capture_thread.join()
    cap.release()
    cv2.destroyAllWindows()

    # Save vehicle counts to CSV
    save_vehicle_counts(total_vehicle_counts, VEHICLE_CLASS_IDS)
    print("Vehicle counts saved to CSV.")

if __name__ == "__main__":
    main()
