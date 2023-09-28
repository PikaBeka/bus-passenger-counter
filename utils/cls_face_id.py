import face_recognition
import cv2
import numpy as np
import math
from datetime import datetime
import uuid
import json
import socket
from .cloud import post_video, post_preview
import time
import os


class Settings:
    '''The class with global variables used throughout the code'''
    dict_for_face_id = {}
    last_time_notif = None

    UDP_PORT_NO_NOTIFICATIONS = 4242
    UDP_IP_ADDRESS = "192.168.0.119"
    DOOR_UDP_PORT = 4545
    Sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    id_for_video = None
    file_for_video = "videos/"
    file_for_detections = "/media/ml/HardDisk/detections/"


    if not os.path.exists(file_for_video):
        os.makedirs(file_for_video)
    
    if not os.path.exists(file_for_detections):
        os.makedirs(file_for_detections)

setting = Settings()


def name_list(path):
    filenames = os.listdir(path)
    filenames_without_extension = [os.path.splitext(
        filename)[0] for filename in filenames]
    return filenames_without_extension


def send_data(data):
    '''The function that converts a dictionary to a json and sends it using UDP protocol'''
    msg = json.dumps(data).encode('utf-8')

    try:
        setting.Sock.connect(
            (setting.UDP_IP_ADDRESS, setting.UDP_PORT_NO_NOTIFICATIONS))
        setting.Sock.send(msg)
    except socket.gaierror:
        print('There an error resolving the host')


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def check_last_notif_time():
    ''' The fucntion to check how much time passed since last notification, 
    returns the time difference between now and that time'''
    now = datetime.now()

    if setting.last_time_notif != None:
        diff = now - setting.last_time_notif
        diff_sec = diff.total_seconds()
    else:
        # diff == 0 only if this is the first time notification is sent
        diff_sec = -1
    return diff_sec


def update_last_time_notif():

    setting.last_time_notif = datetime.now()


def send_dict_notification_after_recording(action, vid_id):
    '''The function that sends notification. It requires video id  '''
    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    time = now.strftime("%H:%M:%S")
    # setting.id_for_video = vid_id

    dct = {
        'date': date,
        'time': time,
        'object': action,
        'camera': 'kamera 2',
        'video_id': vid_id
    }

    send_data(dct)


class video_writer_advanced():
    '''The more advanced class for video recording'''

    def __init__(self):
        self.fcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
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
        '''The function that creates the id for video and initializes the videowriter'''
        if not self.record:
            self.id = str(uuid.uuid4())
            self.record_start_time = time.time()
            self.save_file = setting.file_for_detections + self.id + '.avi'
            self.path = self.save_file
            self.key = self.id + '.avi'
            self.video_writer = cv2.VideoWriter(
                self.save_file, self.fcc, self.fps, (width, height))
            self.record = True

    def update(self, frame):
        '''The function that adds frames to the video. The process stops once the video duration reaches 3 sec'''
        if self.record:
            # print("writing a frame")
            self.video_writer.write(frame)

            if time.time() - self.record_start_time > 3:
                self.record = False
                self.video_writer.release()

                self.preview_path = setting.file_for_detections + self.id + '.png'
                cv2.imwrite(self.preview_path, frame)
                self.preview_key = self.id + '.png'
                post_video(self.path, self.key)
                # post_preview(self.preview_path, self.preview_key )


class Face_detector:
    '''Initialize face detector+recognizer'''

    def __init__(self):
        self.known_face_encodings = []
        self.face_locations = None
        self.face_landmarks = None
        self.face_dict = {}
        self.last_recognized_face_encoding = None
        self.last_recognized_face = None

    def create_database(self, known_face_names):
        '''The function that creates the database from pictures with known names'''
        for name in known_face_names:
            image = face_recognition.load_image_file(
                "/etc/services/static/" + str(name) + ".png")
            print(name)
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)
        return self.known_face_encodings

    def create_encoding(self, name):
        image = face_recognition.load_image_file(
           "/etc/services/static/" + str(name) + ".png")

        encoding = face_recognition.face_encodings(image)[0]
        return encoding

    def is_same(self, rgb_small_frame):
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        if face_encodings:
            face_encoding = face_encodings[0]
            if self.last_recognized_face_encoding is not None:
                results = face_recognition.compare_faces(
                    [self.last_recognized_face_encoding], face_encoding)
                if results[0]:
                    return True
                else:
                    return False

    def detect_and_recognise(self, rgb_small_frame, known_face_encodings, known_face_names):
        '''The function that finds the locations of faces and then recognizes them
        It returns the name'''
        name = "Unknown"
        confidence = '???'
        face_locations = face_recognition.face_locations(
            rgb_small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:

                confidence = face_confidence(face_distances[best_match_index])
                if int(confidence[:2]) < 95:
                    name = 'Unknown'
                    confidence = "???"
                else:
                    name = known_face_names[best_match_index]
        self.last_recognized_face = name
        if self.last_recognized_face != 'Unknown':
            self.last_recognized_face_encoding = self.create_encoding(name)

        return name

    def notify(self, name, video):
        '''The function that send the notification and starts the recording of the video
        If name is unknown, nothing is displayed or sent'''
        if name != "Unknown":
            if name not in self.face_dict:
                video.start_recording()
                id = video.id
                self.face_dict[name] = time.time()
                send_dict_notification_after_recording(
                    str(name)+" anyqtaldy", id)
                update_last_time_notif()

                # msg = "open".encode('utf-8')
                # setting.Sock.sendto(
                #     msg, (setting.UDP_IP_ADDRESS, setting.DOOR_UDP_PORT))
            else:
                if time.time() - self.face_dict[name] > 5:
                    self.face_dict.clear()


def get_last_pixel_value(image):
    '''The fucntion that checks whether the image is corrupted or not '''
    height, width, _ = image.shape

    last_pixel = image[height - 1, width - 1]
    blue = last_pixel[0]
    green = last_pixel[1]
    red = last_pixel[2]
    if (blue in range(123, 130)) and (green in range(123, 130)) and (red in range(123, 130)):
        return True
    else:

        return False


def average_point(points):
    return int(sum(point[0] for point in points) / len(points)), \
        int(sum(point[1] for point in points) / len(points))


def check_look_forward(frame):

    rgb_frame = frame[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for face_landmarks in face_landmarks_list:
        # Calculate average point for each ear (taking points from the outline of the face)
        left_ear_point = average_point(face_landmarks['chin'][:8])
        right_ear_point = average_point(face_landmarks['chin'][8:])
        print('ears')
        # Calculate average point for each eye
        left_eye_point = average_point(face_landmarks['left_eye'])
        right_eye_point = average_point(face_landmarks['right_eye'])
        print('eyes')
        # If the eyes are between the ears, the person is looking forward
        if left_ear_point[0] < left_eye_point[0] < right_eye_point[0] < right_ear_point[0]:
            print("Looking forward")
        else:
            print("Looking sideways")

    # def notify(self,name, video):
    #     if name != "Unknown":
    #         time_diff = check_last_notif_time()
    #         if (time_diff > 10) or (time_diff == -1):
    #             video.start_recording()
    #             id = video.id
    #             setting.dict_for_face_id[name] = [time.time()]
    #             send_dict_notification_toilet(str(name)+" anyqtaldy", id)
    #             update_last_time_notif()
