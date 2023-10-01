import cv2
import config
import datetime
import time

URI = f"rtsp://{config.NAME1}:{config.PSWD1}@{config.IP1}"
# URI = "input/busfinal.mp4"

cap = cv2.VideoCapture(URI)
if not cap.isOpened():
    print("Camera not connected")
    exit()
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Connected")
    print("FPS is ", fps)

start_time = time.time()

while cap.isOpened():
    # if DEBUG:
    current_time = datetime.datetime.now()
    elapsed_time = time.time() - start_time
    ret, frame = cap.read()
    if ret == False:
        print("Error on read")
        break
    else:
        print("Time:", elapsed_time)
    if elapsed_time>=4:
        print("Creating a frame")
        status = cv2.imwrite(f"frames/{str(current_time)}.jpg", frame)
        if status == False:
            print("Shit")
        start_time = time.time()

print('Finished, releasing cap')
cap.release()
# cv2.destroyAllWindows()