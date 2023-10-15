from typing import Dict, Tuple, List
import numpy as np
import cv2


class Point:
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def toTuple(self):
        return (self.x, self.y)


class Vector:
    start: Point
    end: Point

    def __init__(self, start: Point, end: Point):
        self.start: Point = start
        self.end: Point = end

    def is_in(self, point: Point) -> bool:
        v1 = Vector(self.start, self.end)
        v2 = Vector(self.start, point)
        cross_product = (v1.end.x - v1.start.x) * (v2.end.y - v2.start.y) - (
            v1.end.y - v1.start.y
        ) * (v2.end.x - v2.start.x)
        return cross_product < 0


class TrackableObject():
    def __init__(self, detection):
        self.id = detection.id
        x1, y1 = int(detection.estimate[0, 0]), int(detection.estimate[0, 1])
        x2, y2 = int(detection.estimate[0, 2]), int(detection.estimate[0, 3])
        self.xyxy = [x1, y1, x2, y2]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        self.centroids = [[x_center, y_center]]

    def update(self, detection):
        x1, y1 = int(detection.estimate[0, 0]), int(detection.estimate[0, 1])
        x2, y2 = int(detection.estimate[0, 2]), int(detection.estimate[0, 3])
        self.xyxy = [x1, y1, x2, y2]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        self.centroids.append([x_center, y_center])


class LineZone:
    """
    Count the number of objects that cross a line.
    """

    def __init__(self, start: Point, end: Point):
        """
        Initialize a LineCounter object.

        Attributes:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.

        """
        self.vector = Vector(start=start, end=end)
        self.tracker_states: Dict[str, bool] = {}
        self.counted: Dict[int, bool] = {}
        self.centroids: Dict[int, List] = {}
        self.trigger_points: Dict[int, str] = {}
        self.in_count: int = 0
        self.out_count: int = 0

    def trigger(self, tracks, trigger_point='top_left', mode='normal'):
        """
        Update the in_count and out_count for the detections that cross the line.

        Attributes:
            detections (Detections): The detections for which to update the counts.

        """

        for trackableObject in tracks.values():
            # print(trackableObject)
            # print(type(trackableObject))
            tracker_id = trackableObject.id

            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            if tracker_id in self.counted:
                continue

            # we check if all four anchors of bbox are on the same side of vector
            x1, y1, x2, y2 = trackableObject.xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            y = [c[1] for c in trackableObject.centroids]
            direction = trackableObject.centroids[0][1] - np.mean(y)

            if tracker_id not in self.trigger_points:
                if mode == 'camera1':
                    if direction == 0:
                        continue
                    if direction < 0:
                        self.trigger_points[tracker_id] = 'top left'
                    else:
                        self.trigger_points[tracker_id] = 'bottom right'
                else:
                    if y1 <= self.getY():
                        self.trigger_points[tracker_id] = 'top left'
                    else:
                        self.trigger_points[tracker_id] = 'bottom right'

            # print(f"{tracker_id} directions is {direction}")
            # print(f"Trigger point: {self.trigger_points[tracker_id]}")

            if self.trigger_points[tracker_id] == 'top left':
                tracker_state = triggers[0]
            else:
                tracker_state = triggers[3]

            if tracker_id not in self.tracker_states:
                self.tracker_states[tracker_id] = tracker_state
                # print("Added tracker_id")
                continue

            # handle detection on the same side of the line
            if self.tracker_states.get(tracker_id) == tracker_state:
                # print("On the same line")
                continue

            self.tracker_states[tracker_id] = tracker_state
            self.counted[tracker_id] = True
            if tracker_state:
                self.out_count += 1
            else:
                self.in_count += 1
        # print()

    def getY(self):
        return (self.vector.start.y + self.vector.end.y) / 2

    def clear(self):
        self.tracker_states: Dict[str, bool] = {}
        self.counted: Dict[int, bool] = {}
        self.centroids: Dict[int, List] = {}


def polygon_to_mask(polygon: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    width, height = resolution_wh
    mask = np.zeros((height, width))

    cv2.fillPoly(mask, [polygon], color=1)
    return mask


class PolygonZone:
    """
    A class for defining a polygon-shaped zone within a frame for detecting objects.

    Attributes:
        polygon (np.ndarray): A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        frame_resolution_wh (Tuple[int, int]): The frame resolution (width, height)
        triggering_position (Position): The position within the bounding
            box that triggers the zone (default: Position.BOTTOM_CENTER)
        current_count (int): The current count of detected objects within the zone
        mask (np.ndarray): The 2D bool mask for the polygon zone
    """

    def __init__(
        self,
        polygon: np.ndarray,
        frame_resolution_wh: Tuple[int, int],
    ):
        self.polygon = polygon.astype(int)
        self.frame_resolution_wh = frame_resolution_wh
        self.current_count = 0

        width, height = frame_resolution_wh
        self.mask = polygon_to_mask(
            polygon=polygon, resolution_wh=(width + 1, height + 1)
        )

    def trigger(self, point) -> np.ndarray:
        """
        Determines if the detections are within the polygon zone.

        Parameters:
            detections (Detections): The detections
                to be checked against the polygon zone

        Returns:
            np.ndarray: A boolean numpy array indicating
                if each detection is within the polygon zone
        """
        x, y = point
        x = min(x, 1920)
        y = min(y, 1080)
        is_in_zone = self.mask[y, x]
        self.current_count = np.sum(is_in_zone)
        return is_in_zone.astype(bool)

    def getHeight(self):
        average_top = (self.polygon[0] + self.polygon[1]) / 2
        average_bottom = (self.polygon[2] + self.polygon[3]) / 2
        return average_bottom[1] - average_top[1]


def draw_lines_in_pose(frame, pointA, pointB):
    x1, y1 = pointA
    x2, y2 = pointB

    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)


def draw_pose(frame, keypoints):
    for keypoint in keypoints:
        if len(keypoint) == 0:
            break
        for point in keypoint:
            x, y = point
            cv2.circle(frame, (x, y), radius=3,
                       color=(0, 0, 255), thickness=-1)

        # head
        draw_lines_in_pose(frame, keypoint[4], keypoint[2])
        draw_lines_in_pose(frame, keypoint[2], keypoint[0])
        draw_lines_in_pose(frame, keypoint[0], keypoint[1])
        draw_lines_in_pose(frame, keypoint[1], keypoint[3])

        # left arm
        draw_lines_in_pose(frame, keypoint[10], keypoint[8])
        draw_lines_in_pose(frame, keypoint[8], keypoint[6])

        # right arm
        draw_lines_in_pose(frame, keypoint[9], keypoint[7])
        draw_lines_in_pose(frame, keypoint[7], keypoint[5])

        # body
        draw_lines_in_pose(frame, keypoint[6], keypoint[5])
        draw_lines_in_pose(frame, keypoint[5], keypoint[11])
        draw_lines_in_pose(frame, keypoint[11], keypoint[12])
        draw_lines_in_pose(frame, keypoint[12], keypoint[6])

        # left leg
        draw_lines_in_pose(frame, keypoint[12], keypoint[14])
        draw_lines_in_pose(frame, keypoint[14], keypoint[16])

        # right leg
        draw_lines_in_pose(frame, keypoint[11], keypoint[13])
        draw_lines_in_pose(frame, keypoint[13], keypoint[15])
