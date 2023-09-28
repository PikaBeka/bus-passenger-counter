from ultralytics import YOLO
from norfair import Tracker, Detection
from utils.draw_utils import LineZone, Point
from collections import defaultdict
from utils.stream_udp import Stream
from utils.cls_git import Yolo_detections, Norfair_Detections
import cv2
import numpy as np

DEBUG = 1

if DEBUG:
    cap = cv2.VideoCapture('./input/shake.mp4')
    # cap = cv2.VideoCapture('./input/abzal_bus.mp4')
else:
    STREAM_PORT = 5255
    stream = Stream(port=STREAM_PORT)
    stream.connect()

print("Connected")

def draw_track(frame, rectPoint1, rectPoint2, points, track_id):
    cv2.putText(frame, str(track_id), rectPoint1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame, rectPoint1, rectPoint2, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

model = YOLO('yolov8s.pt')

# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))

# LINE_START = Point(1245, 0)
# LINE_END = Point(803, 1052)

LINE_START = Point(620, 675)
LINE_END = Point(1007, 612)

line_counter = LineZone(start=LINE_START, end=LINE_END)
track_history = defaultdict(lambda: [])
det_cls = Yolo_detections()
norfair_det = Norfair_Detections()

shake_threshold = 15

# Initialize variables
prev_gray = None
prev_pts = None
def check_shake(frame):
    global prev_gray, prev_pts
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If this is the first frame, initialize variables
    if prev_gray is None:
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Calculate optical flow (i.e., the movement of points between frames)
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    # Filter out points that are not moving
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    # Calculate the mean distance between the points in the two frames
    mean_distance = cv2.norm(good_new - good_old) / len(good_new)

    # If the mean distance is above a certain threshold, the camera is shaking
    if mean_distance > shake_threshold:
        return True

    # Update variables for next iteration
    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)
    return False

def check_crossed(frame):
    yolo_det = det_cls.detect(frame)

    norf_res = norfair_det.update(yolo_det)

    norfair_det.draw_bboxes(res=norf_res, frame=frame)

    line_counter.trigger(detections=norf_res)

    cv2.line(
        frame,
        (line_counter.vector.start.x, line_counter.vector.start.y),
        (line_counter.vector.end.x, line_counter.vector.end.y),
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
        shift=0,
    )
    
    cv2.putText(
        frame,
        f"Out count:{line_counter.out_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"In count:{line_counter.in_count}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )
    print(f"In {line_counter.in_count}, out {line_counter.out_count}")

    return frame


while True:
    if DEBUG:
        ret, frame = cap.read()
        if ret == False:
            print("Error on read")
            break
    else:
        frame = stream.get_frames()
    frame = cv2.resize(frame, (1920, 1080))
    status = check_shake(frame)
    if status == False:
        print("No shaking")
        frame = check_crossed(frame)
    else:
        print("shaking")
    
    if DEBUG:
        cv2.imshow("Bus", frame)
        # out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if DEBUG:
    print('Finished, releasing cap')
    cap.release()
    # out.release()
    cv2.destroyAllWindows()