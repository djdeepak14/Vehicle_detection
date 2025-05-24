from save_csv import save_vehicle_counts
import cv2
from ultralytics import YOLO
import threading
import queue
import time
from playsound import playsound

# Load YOLO model
model = YOLO("yolov8s.pt")

# Sound file path
def play_ping():
    try:
        playsound('/Users/deepakkhanal/traffic/ping.mp3', block=False)
    except Exception as e:
        print("Sound Error:", e)

# Use DroidCam or fallback to local video
url = "http://192.168.99.172:4747/video"
vedio = "/Users/deepakkhanal/traffic/vedio.mp4"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print(f"Error: clear not open DroidCam at {url}.")
    print("Ensure DroidCam is running, both devices are on the same Wi-Fi, and permissions are granted.")
    exit()
else:
    print(f"DroidCam found at {url}")
# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Processing resolution: {frame_width}x{frame_height}")

VEHICLE_CLASS_IDS = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
frame_queue = queue.Queue(maxsize=10)
stop_thread = False

# Persistent total vehicle counter
total_vehicle_counts = {cls_id: 0 for cls_id in VEHICLE_CLASS_IDS}

def capture_frames():
    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        time.sleep(0.005)

capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

frame_count = 0
skip_frames = 2

try:
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        print(f"Processing frame {frame_count}")
        frame_vehicle_counts = {cls_id: 0 for cls_id in VEHICLE_CLASS_IDS}
        vehicle_indexes = {cls_id: 1 for cls_id in VEHICLE_CLASS_IDS}
        vehicle_detected = False

        results = model.track(frame, persist=False, tracker="bytetrack.yaml", conf=0.25, iou=0.5, imgsz=640)[0]
        print(f"Number of detections: {len(results.boxes)}")

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            if not vehicle_detected:
                play_ping()
                vehicle_detected = True

            class_name = VEHICLE_CLASS_IDS[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            frame_vehicle_counts[cls_id] += 1
            total_vehicle_counts[cls_id] += 1

            index = vehicle_indexes[cls_id]
            vehicle_indexes[cls_id] += 1

            label = f"{class_name} {index}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            font_scale = 3
            thickness = 6
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1 - 5), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Draw current vehicle summary
        y_offset = 40
        total_vehicles = sum(frame_vehicle_counts.values())
        cv2.putText(frame, f"Current Vehicles: {total_vehicles}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        cv2.putText(frame, f"Current Vehicles: {total_vehicles}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        y_offset += 30

        for cls_id, count in frame_vehicle_counts.items():
            label = f"{VEHICLE_CLASS_IDS[cls_id]}s: {count}"
            cv2.putText(frame, label, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, label, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 30

        cv2.imshow("Vehicle Counter", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    stop_thread = True
    capture_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    save_vehicle_counts(total_vehicle_counts, VEHICLE_CLASS_IDS)