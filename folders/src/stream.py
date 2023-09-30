import cv2
import socket
from threading import Thread
import time
import config

URI = f"rtsp://{config.NAME1}:{config.PSWD1}@{config.IP1}"
WIDTH = 1920
HEIGHT = 1080
DISCONNECT_TIMEOUT = 60

pipeline = f"gst-launch-1.0 rtspsrc location={URI} latency=0 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# Check if the pipeline is created successfully
if not cap.isOpened():
    print("The first camera is not available")
    URI = f"rtsp://{config.NAME1}:{config.PSWD1}@{config.IP1}"
    pipeline = f"gst-launch-1.0 rtspsrc location={URI} latency=0 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Cameras are not available")

FPS = cap.get(cv2.CAP_PROP_FPS)
print(FPS)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('0.0.0.0', 5252)
sock.bind(server_address)


class ConnectClients(Thread):
    def __init__(self, sock):
        super(ConnectClients, self).__init__()

        self.sock = sock
        self.clients = {}

    def run(self):
        global FPS, WIDTH, HEIGHT
        while True:
            # Receive the UDP packet
            data, address = sock.recvfrom(1024)
            port = int(data.decode())

            # Print the received data
            # print(f"Received: {data.decode()} from {address[0]}:{address[1]}")

            # Send a response (optional)
            response = "Port recieved!"
            sock.sendto(response.encode(), address)

            if self.clients.get(port) is None:

                gst_out = f"appsrc ! videoconvert ! x264enc noise-reduction=1000 speed-preset=superfast tune=zerolatency key-int-max=30 ! rtph264pay config-interval=1 pt=96 ! udpsink host={address[0]} port={port}"
                fourcc_fmt = cv2.VideoWriter_fourcc(*'X264')
                out = cv2.VideoWriter(gst_out,
                                      fourcc=fourcc_fmt,
                                      apiPreference=cv2.CAP_GSTREAMER,
                                      fps=FPS,
                                      frameSize=(WIDTH, HEIGHT))

                self.clients[port] = {"cap": out, "last_connect": time.time()}

                print(f"client connected: {port}")

            else:
                self.clients[port]["last_connect"] = time.time()
                print(f"client refreshed: {port}")


clients_server = ConnectClients(sock)
clients_server.deamon = True
clients_server.start()

ret, frame = cap.read()

while ret:
    delete_ports = []
    for port, client in clients_server.clients.items():

        if time.time() - client["last_connect"] > DISCONNECT_TIMEOUT:
            print(f"client disconnected: {port}")
            client["cap"].release()
            delete_ports.append(port)
        else:
            client["cap"].write(frame)

    for port in delete_ports:
        del clients_server.clients[port]

    ret, frame = cap.read()

print("done")
