from ultralytics import YOLO
from utils.draw_utils import LineZone, TrackableObject
from utils.cls_git import (
    Yolo_detections,
    Norfair_Detections,
    video_writer_advanced
)
import cv2
import config
from utils.bus_utils import (
    check_shake,
    check_door,
    update_status,
    draw_line,
    put_in_out_text,
    filter_boxes_near_door,
    downscale_boxes,
)
import time

DEBUG = 0

if DEBUG:
    cap = cv2.VideoCapture('./input/test/test_8.avi')
else:
    # URI = 0
    URI = f"rtsp://{config.NAME1}:{config.PSWD1}@{config.IP1}"
    WIDTH = 960
    HEIGHT = 540
    DISCONNECT_TIMEOUT = 60

    pipeline = f"gst-launch-1.0 rtspsrc location={URI} latency=0 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"

    # pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera not connected")
    exit()
else:
    print("Connected camera 1")

line_counter = LineZone(start=config.LINE_START1, end=config.LINE_END1)
det_cls = Yolo_detections(model='best.pt', task='predict')
norfair_det = Norfair_Detections()
video_writer = video_writer_advanced()
trackableObjects = {}


def check_crossed(frame):
    yolo_det = det_cls.detect(frame, conf=0.35, iou=0.72)

    yolo_det = filter_boxes_near_door(yolo_det, 190, 780)

    yolo_det_scaled = downscale_boxes(yolo_det)

    norf_res = norfair_det.update(yolo_det_scaled)

    norfair_det.draw_bboxes(res=norf_res, frame=frame)

    for detection in norf_res:
        if detection.id in trackableObjects:
            trackableObjects[detection.id].update(detection)
        else:
            trackableObjects[detection.id] = TrackableObject(detection)
    line_counter.trigger(tracks=trackableObjects, mode='camera1')

    start = (line_counter.vector.start.x, line_counter.vector.start.y)
    end = (line_counter.vector.end.x, line_counter.vector.end.y)
    draw_line(frame, start, end)

    out_count = line_counter.out_count
    in_count = line_counter.in_count

    put_in_out_text(frame, in_count, out_count)

    if DEBUG:
        print(f"In {line_counter.in_count}, out {line_counter.out_count}")

    video_writer.update(frame)

    return frame


door_det_cls = Yolo_detections(model='door1.pt')

global_door_status = 'close'
prev_status = 'close'
start = time.time()

while True:
    ret, frame = cap.read()
    if ret == False:
        print("Error on read")
        break
    frame = cv2.resize(frame, (960, 540))
    door_status = check_door(door_det_cls, frame)
    if door_status == 'open':
        global_door_status = 'open'
        start = time.time()
    elif door_status == 'close' and time.time() - start > 4:
        global_door_status = 'close'
        line_counter.clear()
    if global_door_status == 'open':
        video_writer.start_recording()
        frame = check_crossed(frame)
    else:
        video_writer.release()
    if DEBUG:
        cv2.imshow("Bus", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

if DEBUG:
    print('Finished, releasing cap')
    cap.release()
    cv2.destroyAllWindows()
