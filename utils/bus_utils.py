import cv2
import time
import numpy as np

shake_threshold = 25

prev_gray = None
prev_pts = None


def check_shake(frame):
    global prev_gray, prev_pts
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None)

    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    mean_distance = cv2.norm(good_new - good_old) / len(good_new)

    if mean_distance > shake_threshold:
        return True

    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    return False


def check_door(det_cls, frame):
    yolo_det = det_cls.detect(frame, classes=[0, 1], conf=0.5)
    _, door_class, _ = yolo_det
    if np.any(door_class) == 0:
        return "close"
    else:
        return "open"


def update_status(global_status, prev_status, new_status, startTime):
    if new_status != prev_status:  # Check if status has changed
        prev_status = new_status
        startTime = time.time()  # Reset the duration
    else:  # Status remains the same
        if time.time() - startTime >= 2:  # Check if the duration has reached 3 seconds
            global_status = new_status
    return global_status, prev_status


def draw_line(frame, start, end):
    cv2.line(
        frame,
        start,
        end,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
        shift=0,
    )


def put_in_out_text(frame, in_count, out_count):
    cv2.putText(
        frame,
        f"Out count:{out_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"In count:{in_count}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )


def filter_boxes_near_door(yolo_det, x_left, x_right):
    filtered_box = []
    filtered_clss = []
    filtered_conf = []

    boxes, clsss, confs = yolo_det

    if len(boxes) == 0:
        return filtered_box, filtered_clss, filtered_conf

    for i, box in enumerate(boxes):
        if box[0] < x_left or box[2] > x_right:
            continue
        filtered_box.append(box)
        filtered_clss.append(clsss[i])
        filtered_conf.append(confs[i])
    return filtered_box, filtered_clss, filtered_conf


def downscale_boxes(detections, x_scale=0.4, y_scale=0.4):
    boxes, clsss, confs = detections
    scaled_boxes = []
    for i, box in enumerate(boxes):
        x_center = (box[0]+box[2]) // 2
        y_center = (box[1]+box[3]) // 2
        width = box[2] - box[0]
        height = box[3] - box[1]

        width_scaled = width * x_scale
        height_scaled = height * y_scale

        x1 = x_center - width_scaled // 2
        y1 = y_center - height_scaled // 2

        x2 = x_center + width_scaled // 2
        y2 = y_center + height_scaled // 2

        scaled_boxes.append([x1, y1, x2, y2])

    return scaled_boxes, clsss, confs
