import cv2
import numpy as np
from ultralytics import YOLO
import norfair
import sys
import socket
import json
from norfair import Detection, Paths, Tracker, Video
from datetime import datetime
import uuid
from .cloud import post_video, post_preview
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
    # file_for_detections = "/media/ml/HardDisk/detections/"

    if not os.path.exists(file_for_video):
        os.makedirs(file_for_video)

    # if not os.path.exists(file_for_detections):
    #     os.makedirs(file_for_detections)


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


def check_last_notif_time():
    """The fucntion to check how much time passed since last notification,
    returns the time difference between now and that time"""
    now = datetime.now()

    if setting.last_time_notif != None:
        diff = now - setting.last_time_notif
        diff_sec = diff.total_seconds()
    else:
        # diff == -1 only if this is the first time notification is sent
        diff_sec = -1
    return diff_sec


def send_data(data, port):
    """The function that converself.preview_keyts a dictionary to a json and sends it using UDP protocol"""

    msg = json.dumps(data).encode("utf-8")
    try:
        setting.Sock.connect((setting.UDP_IP_ADDRESS, port))
        setting.Sock.send(msg)

    except socket.gaierror:
        print("There an error resolving the host")


def send_dict_notification_after_recording(action, vid_id):
    """The function that sends notification. It requires video id (to the contrary of the function send_dict_notification)"""

    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    time = now.strftime("%H:%M:%S")
    # setting.id_for_video = vid_id

    dct = {
        "date": date,
        "time": time,
        "object": action,
        "camera": "kamera 2",
        "video_id": vid_id,
    }
    print("it worked")
    send_data(dct, setting.UDP_PORT_NO_NOTIFICATIONS)


def update_last_time_notif():
    """The function that updates the time when the last notification was sent to now"""
    setting.last_time_notif = datetime.now()


def calculate_std_dev(frame):
    """Calculate the standard deviation in the image, used for detecting the blocking of a camera"""

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray_frame)

    return std_dev


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
            self.save_file = setting.file_for_detections + self.id + ".avi"
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

            if time.time() - self.record_start_time > 3:
                self.record = False
                self.video_writer.release()

                self.preview_path = setting.file_for_detections + self.id + ".png"
                cv2.imwrite(self.preview_path, frame)
                self.preview_key = self.id + ".png"
                post_video(self.path, self.key)
                # post_preview(self.preview_path, self.preview_key )


def get_entered_time(tracked_objects, frame, video):
    """The fucntion to get the time the person entered the place"""

    (h_l, w_l) = frame.shape[:2]
    # this is the y value crosssing which is tracked in the code
    center_y_frame = int(4 * h_l / 5)

    if tracked_objects:
        for object in tracked_objects:
            y1 = int(object.estimate[0, 1])
            y2 = int(object.estimate[0, 3])
            # calculate the y coordinate of the center of bounding box
            center_y = (y2 - y1) / 2 + y1

            # if the person crossed the line (entered) - we record this time point
            if center_y > center_y_frame:
                now = datetime.now()
                setting.entered_time = now
                setting.exit_time = None

            # if this person appeared in previous consecutive frames (once the person dissapears, it gets another id!)
            if object.id in setting.past_y_dict:
                # if the person is moving towards the entrance and has not crossed the line
                # if (setting.past_y_dict[object.id] < center_y) & (center_y < center_y_frame) & (setting.exit_time == None):
                if (setting.past_y_dict[object.id] < center_y) & (
                    center_y < center_y_frame
                ):
                    # start recording
                    video.start_recording(height=h_l, width=w_l)
                # if the person exited (moved away from entrance), entered time == None
                if (setting.past_y_dict[object.id] > center_y) & (
                    center_y < center_y_frame
                ):
                    setting.entered_time = None
                    setting.exit_time = time.time()

            # the code to handle the length of the dictionary past_y_dict
            if len(setting.past_y_dict) == 0:
                setting.first_time_notif = time.time()

            setting.past_y_dict[object.id] = center_y
            # if more than 15 min since the first addition of key-value pair is passed, clear the dictionary
        if time.time() - setting.first_time_notif > 900:
            setting.past_y_dict.clear()
            setting.first_time_notif = None

    return setting.entered_time


def calculate_absense(entered_time):
    """The function that calculates how much time the person was absent"""

    if entered_time != None:
        now = datetime.now()
        absence_time = now - entered_time
        absence_time = absence_time.total_seconds()
    else:
        absence_time = -1
    return absence_time


def draw_line(frame):
    """The function that draws a line (for better entrance visualization) - used for toilet code only"""
    (h_l, w_l) = frame.shape[:2]
    center_y_frame = int(4 * h_l / 5)
    start_point = (0, center_y_frame - 150)
    end_point = (w_l, center_y_frame - 150)
    green = (200, 112, 4)
    cv2.line(frame, start_point, end_point, green, 9)


class detect_lying_running:
    """The class to detect whether the person is running or lying"""

    def __init__(self):
        self.running_id = 0
        self.lying_id = 0
        self.past_detected_objects = {}
        self.first_time = None

    def check_lying(self, tracked_objects, video):
        """The function that examines the proportions of bboxes to understand whether the person is lying or not"""
        for box in tracked_objects:
            x1, y1 = int(box.estimate[0, 0]), int(box.estimate[0, 1])
            x2, y2 = int(box.estimate[0, 2]), int(box.estimate[0, 3])

            if box.id > self.lying_id and x2 - x1 > 3 * (y2 - y1):
                time_diff = check_last_notif_time()
                # send notifications once in 10 sec
                if (time_diff > 10) or (time_diff == -1):
                    self.lying_id = box.id
                    print("Adam zhatyr")
                    video.start_recording()
                    id = video.id
                    send_dict_notification_after_recording("Adam zhatyr", id)
                    update_last_time_notif()

    def check_running(self, tracked_objects, video):
        """The function to check whether the person is running by tracking his/her speed in horizontal direction"""
        for box in tracked_objects:
            x1, y1 = int(box.estimate[0, 0]), int(box.estimate[0, 1])
            x2, y2 = int(box.estimate[0, 2]), int(box.estimate[0, 3])
            center_x = (x2 - x1) / 2 + x1

            if box.id > self.running_id and box.id in self.past_detected_objects:
                self.running_id = box.id
                change = abs(self.past_detected_objects[box.id] - center_x)
                # 0.085 - can be adjusted
                if change > 0.085 * (x2 - x1):
                    time_diff = check_last_notif_time()
                    if (time_diff > 10) or (time_diff == -1):
                        print("Adam zhugurude")
                        video.start_recording()
                        id = video.id

                        send_dict_notification_after_recording("Adam zhugurude", id)
                        update_last_time_notif()
            # handling the overfilling of the dictionary
            if len(self.past_detected_objects) == 0:
                self.first_time = time.time()

            self.past_detected_objects[box.id] = center_x
        # clear dictionary every 15 minutes
        if self.first_time != None:
            if time.time() - self.first_time > 900:
                self.past_detected_objects.clear()
                self.first_time = None


def check_long_absence(absence_time, video_id):
    """Send the notification if the person is absent more than 5 seconds"""
    if absence_time > 30:
        time_diff = check_last_notif_time()
        # send notification once in 5 sec
        if (time_diff > 5) or (time_diff == -1):
            if video_id != None:
                send_dict_notification_after_recording(
                    "Adamnyn ornynda bolmauy 5 s asyp ketti", vid_id=video_id
                )
                update_last_time_notif()


def get_last_pixel_value(image):
    """The fucntion that checks whether the image is corrupted or not"""
    height, width, _ = image.shape

    last_pixel = image[height - 1, width - 1]
    blue = last_pixel[0]
    green = last_pixel[1]
    red = last_pixel[2]
    if (
        (blue in range(123, 130))
        and (green in range(123, 130))
        and (red in range(123, 130))
    ):
        return True
    else:
        return False


def notify_if_blocked(video):
    time_diff = check_last_notif_time()
    if (time_diff > 10) or (time_diff == -1):
        video.start_recording()
        id = video.id
        send_dict_notification_after_recording("Kamera zhabylgan", id)
        update_last_time_notif()


def notify_if_crowd(video):
    time_diff = check_last_notif_time()
    if (time_diff > 10) or (time_diff == -1):
        video.start_recording()
        id = video.id
        send_dict_notification_after_recording("Top anyqtaldy", id)
        update_last_time_notif()


def notify_action(video, action):
    time_diff = check_last_notif_time()
    if (time_diff > 10) or (time_diff == -1):
        video.start_recording()
        id = video.id
        send_dict_notification_after_recording(action, id)
        update_last_time_notif()
