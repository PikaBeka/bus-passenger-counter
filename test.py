from ultralytics import YOLO
from utils.draw_utils import LineZone
from utils.cls_git import Yolo_detections, Norfair_Detections, video_writer_advanced
import cv2
import config
import datetime
import time

URI = f"rtsp://{config.NAME1}:{config.PSWD1}@{config.IP1}"

cap = cv2.VideoCapture(URI)
if not cap.isOpened():
    print("Camera not connected")
    exit()
else:
    print("Connected")

start_time = time.time()

while True:
    # if DEBUG:
    current_time = datetime.datetime.now()
    elapsed_time = time.time() - start_time
    ret, frame = cap.read()
    if ret == False:
        print("Error on read")
        break
    if int(elapsed_time)>=4:

        cv2.imwrite("frames/%s.jpg"%current_time, frame)
        start_time = time.time()

print('Finished, releasing cap')
cap.release()
# cv2.destroyAllWindows()