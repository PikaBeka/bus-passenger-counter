# from database import DB

# dct = {
#     'date': "date",
#     'time': "time",
#     'object': "action",
#     'camera': 'kamera 2',
#     'video_id': 'vid_id'
# }

# DB.insert(DB.detections, dct)
import json
import socket


class Settings:
    '''The class with global variables used throughout the code'''
    last_time_notif = None
    entered_time = None
    exit_time = None
    first_time_notif = None
    UDP_PORT_NO_NOTIFICATIONS = 4242
    UDP_IP_ADDRESS = "192.168.0.21"
    Sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    past_y_dict = {}
    file_for_video = "videos/"
    file_for_detections = "/media/ml/HardDisk/detections/"


setting = Settings()


def send_data(data, port):
    '''The function that converself.preview_keyts a dictionary to a json and sends it using UDP protocol'''

    msg = json.dumps(data).encode('utf-8')
    try:
        setting.Sock.connect((setting.UDP_IP_ADDRESS, port))
        setting.Sock.send(msg)

    except socket.gaierror:
        print('There an error resolving the host')


dct = {
    'sentence': 'che tam'
}
send_data(dct, 4242)
