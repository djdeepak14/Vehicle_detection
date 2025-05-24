import cv2
from ultralytics import YOLO
from collections import defaultdict

# loading the model of the YOLO (you look once)
model = YOLO('yolo11n.pt')

# we need different classes from the already trained version dataset which is also called COCO
class_list = model.names
print(class_list)

# capture the video using OpenCV which helps video capturing frame by frame
cap = cv2.VideoCapture('/Users/deepakkhanal/traffic/vedio.mp4')  

line_Y_red = 1400

# dictionary to count the vehicles by the class
class_count = defaultdict(int)

# to keep the track of the vehicles IDS that have crossed the line
crossed_ids = set()

# for the detection of the vehicles in the video by frame
while cap.isOpened():  # 
    ret, frame = cap.read()
    if not ret:
        break

    # we are going to run the yolo tracking on the frame
    results = model.track(frame, persist=True, classes = [1,2,3,5,6,7])



    # ensure the results are not empty
    if results[0].boxes.data is not None:
        # get the detected boxes and their class indices and track ids
        boxes = results[0].boxes.xyxy.cpu()
        tracks_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()
       

        cv2.line(frame, (3700, line_Y_red), (690, line_Y_red), (0, 0, 255), 3)
        cv2.putText(frame, "Vehicle cross ", (690, line_Y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,255 ), 3)
        # loop through the detected boxes
        for box, track_id, class_idx, confidence in zip(boxes, tracks_ids, class_indices, confidences):
            # get the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box)
            ## making the red dot in the bounding box
            cx = (x1 + x2) // 2 # center x coordinate
            cy = (y1 + y2) // 2 # center y coordinate
            class_name = class_list[class_idx]  # get the class name from the index
            confidence = f"{confidence * 100:.2f}%" # format confidence to 2 decimal places
            # put the track id on the frame
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {class_name} {confidence}", (x1, y1 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255)  
, 3)

            # draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            
            # check if the vehicle has crossed the line
            if cy > line_Y_red and track_id not in crossed_ids:
                #mark the vehicle as crossed
                crossed_ids.add(track_id)
                class_count[class_name] += 1
            
            
        # displays the counts of the vehicles on the frame
        y_offset = 50
        for class_name, count in class_count.items():
            cv2.putText(frame, f"{class_name}: {count}", (30, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            y_offset += 30
    # showing the frame
    cv2.imshow("yolo tracking and counting the vehicles", frame)

    # interrupting the process by using the key 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

# release the resources
cap.release()
cv2.destroyAllWindows()
