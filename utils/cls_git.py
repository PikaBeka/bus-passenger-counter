import cv2
import numpy as np
from ultralytics import YOLO
import socket
from norfair import Detection, Tracker
from datetime import datetime
import uuid
import time
import os


class Settings:
    """The class with global variables used throughout the code"""

    last_time_notif = None
    entered_time = None
    exit_time = None
    first_time_notif = None
    UDP_PORT_NO_NOTIFICATIONS = 4242
    UDP_IP_ADDRESS = "192.168.0.134"
    Sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    past_y_dict = {}
    file_for_video = "videos/"
    file_for_detections = "videos/"

    if not os.path.exists(file_for_video):
        os.makedirs(file_for_video)

    if not os.path.exists(file_for_detections):
        os.makedirs(file_for_detections)

setting = Settings()

class Yolo_detections:
    """Class with results of detecting people with YOLOv8 model"""

    def __init__(self, task = 'predict', model='yolov8s.pt'):
        # we use tensorrt speeded-up weights
        if task == 'predict':
            self.model = YOLO(model)
        elif task == 'pose':
            self.model = YOLO("services/models/yolov8s-pose.pt")

    def detect(self, frame, classes=[0], centers=False, conf = 0.4):
        """Detecting people"""
        yolo_detections = self.model.predict(
            frame, classes=classes, conf=conf, verbose=False
        )
        res = yolo_detections[0].boxes.cpu().numpy()
        if centers:
            boxes = res.xywh.astype(np.uint32)
        else:
            boxes = res.xyxy.astype(np.uint32)
        cls = res.cls.astype(np.uint8)
        conf = res.conf
        return boxes, cls, conf
    
    def track(self, frame):
        results = self.model.track(frame, classes=[0], persist=True, verbose=False, tracker='botsort.yaml')[
            0].boxes.cpu().numpy()

        boxes = results.xyxy.astype(np.uint32)
        if results.id is None:
            track_ids = []
        else:
            track_ids = results.id.astype(np.uint8)

        clss = results.cls.astype(np.uint8)
        conf = results.conf

        return boxes, clss, track_ids, conf
    
    def pose(self, frame):
        results = self.model.track(frame, verbose=False, conf=0.4, tracker='botsort.yaml')

        return results

class Norfair_Detections:
    """Norfair is used as a tracker standard in our company"""

    def __init__(self):
        self.tracker = Tracker(distance_function="euclidean", distance_threshold=300)

    def transform_yolo2norfair(self, yolo):
        """Pass the result of yolo detections for Norfair Tracker"""
        self.boxes, self.cls, self.conf = yolo
        detections = []

        for i, box in enumerate(self.boxes):
            detections.append(
                [box[0], box[1], box[2], box[3], self.conf[i], self.cls[i]]
            )
        detections = np.asarray(detections)
        norfair_detections = [Detection(points) for points in detections]

        return norfair_detections

    def update(self, yolo_det):
        """The function that updates tracking results in the main loop"""

        norfair_detections = self.transform_yolo2norfair(yolo_det)
        tracked_objects = self.tracker.update(detections=norfair_detections)
        return tracked_objects

    def draw_bboxes(self, frame, res):
        """The function that draws bounding boxes on people"""

        for box in res:
            track_id = box.id
            x1, y1 = int(box.estimate[0, 0]), int(box.estimate[0, 1])
            x2, y2 = int(box.estimate[0, 2]), int(box.estimate[0, 3])
            cv2.putText(frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
class video_writer_advanced:
    """The more advanced class for video recording"""

    def __init__(self):
        self.fcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        self.fps = 20
        self.path = None
        self.key = None
        self.record = False
        self.video_writer = None
        self.record_start_time = None
        self.id = None
        self.preview_path = None
        self.preview_key = None

    def start_recording(self, height=1080, width=1920):
        """The function that creates the id for video and initializes the videowriter"""
        if not self.record:
            self.id = str(uuid.uuid4())
            self.record_start_time = time.time()
            current_datetime = datetime.now()
            formatted_date = current_datetime.strftime("%Y-%m-%d")
            formatted_time = current_datetime.strftime("%H:%M:%S")
            self.save_file = setting.file_for_detections + formatted_date + " " + formatted_time + " " + self.id + ".avi"
            self.path = self.save_file
            self.key = self.id + ".avi"
            self.video_writer = cv2.VideoWriter(
                self.save_file, self.fcc, self.fps, (width, height)
            )
            self.record = True

    def update(self, frame):
        """The function that adds frames to the video. The process stops once the video duration reaches 3 sec"""
        if self.record:
            # print("writing a frame")
            self.video_writer.write(frame)
            # print(os.listdir(setting.file_for_detections))

            if time.time() - self.record_start_time > 20:
                self.record = False
                self.video_writer.release()
    
    def release(self):
        if self.record:
            self.record = False
            self.video_writer.release()