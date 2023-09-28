import cv2
import socket
import numpy as np
import time


class Stream:
    def __init__(self, port, server_ip="192.168.3.10", sevrer_port=5252, reconnect_timeout=50, width=1920, height=1080):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = (server_ip, sevrer_port)
        self.frame = None
        self.port = port
        self.reconnect_timeout = reconnect_timeout
        self.width = width
        self.height = height

    def send_port_to_server(self):
        message = str(self.port)
        self.sock.sendto(message.encode(), self.server_address)
        data, address = self.sock.recvfrom(2048)
        print(
            f"Connecting to server: {address[0]}:{address[1]},  {data.decode()}")

        self.last_connect = time.time()

    def connect(self):
        self.send_port_to_server()

        self.cap = cv2.VideoCapture(
            f"udpsrc port={self.port} ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,width={self.width},height={self.height},format=BGR ! appsink", 
            cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("Error: Could not open GStreamer pipeline")
            exit(1)

    def get_frames(self):

        if time.time() - self.last_connect > self.reconnect_timeout:
            self.send_port_to_server()

        ret, self.frame = self.cap.read()

        # if self.frame is not None and type(self.frame) == np.ndarray:
        #     self.frame = cv2.resize(
        #         self.frame, (self.frame.shape[1]//2, self.frame.shape[0]//2))

        return self.frame