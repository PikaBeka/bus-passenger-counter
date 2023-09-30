from ultralytics import YOLO
from utils.draw_utils import LineZone
from utils.cls_git import Yolo_detections, Norfair_Detections, video_writer_advanced
import cv2
import config

DEBUG = 0

if DEBUG:
    cap = cv2.VideoCapture('./input/busfinal.mp4')
    # cap = cv2.VideoCapture('./input/shake.mp4')
    # cap = cv2.VideoCapture('./input/abzal_bus.mp4')
else:
    # URI = 0
    URI = f"rtsp://{config.NAME1}:{config.PSWD1}@{config.IP1}"
    WIDTH = 1920
    HEIGHT = 1080
    DISCONNECT_TIMEOUT = 60

    pipeline = f"gst-launch-1.0 rtspsrc location={URI} latency=0 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera not connected")
    exit()
else:
    print("Connected")


# LINE_START = Point(620, 675)
# LINE_END = Point(1007, 612)

line_counter = LineZone(start=config.LINE_START1, end=config.LINE_END1)
det_cls = Yolo_detections()
norfair_det = Norfair_Detections()
video_writer = video_writer_advanced()

shake_threshold = config.SHAKE

prev_gray = None
prev_pts = None
def check_shake(frame):
    global prev_gray, prev_pts

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    mean_distance = cv2.norm(good_new - good_old) / len(good_new)

    if mean_distance > shake_threshold:
        return True

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

    if DEBUG:
        print(f"In {line_counter.in_count}, out {line_counter.out_count}")

    video_writer.update(frame)

    return frame


while True:
    # if DEBUG:
    ret, frame = cap.read()
    if ret == False:
        print("Error on read")
        break
    frame = cv2.resize(frame, (1920, 1080))
    status = check_shake(frame)
    if status == False:
        print("No shaking")
        video_writer.start_recording()
        frame = check_crossed(frame)
    else:
        # video_writer.release()
        print("shaking")
    
    # if DEBUG:
    #     cv2.imshow("Bus", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

if DEBUG:
    print('Finished, releasing cap')
    cap.release()
    cv2.destroyAllWindows()