import cv2
import numpy as np


def getPredict(model, frame, classes):
    """
    Function that returns predictions from the model using the frame
    Input:
    * model - YOLOv8 model that predicts
    * frame [image] - what to predict
    * classes [List] - filter by classes
    Returns:
    * bounding boxes [List]
    * detected classes [List]
    """

    results = model.predict(frame, classes=classes,
                            conf=0.3, verbose=False)

    res = results[0].boxes.cpu().numpy()
    boxes = res.xyxy.astype(np.int32)
    clss = results[0].boxes.cls.cpu().numpy()

    return boxes, clss


def rescale_bboxes(bboxes, scale=1.3):
    """
    Function that rescales the bounding boxes of the predictions
    Input:
    * bboxes [List] - list of bounding boxes to rescale
    * scale = 1.01 [Integer] - scale factor
    Output:
    * rescaled_bboxes [List] - list of rescaled bounding boxes
    """
    rescaled_bboxes = []
    for bbox in bboxes:
        x_top, y_top, x_bottom, y_bottom = bbox
        new_width = int((x_bottom-x_top) * scale)
        new_height = int((y_bottom-y_top) * scale)
        delta_width = (new_width - (x_bottom-x_top)) // 2
        delta_height = (new_height - (y_bottom-y_top)) // 2

        new_x_tl = x_top - delta_width
        new_y_tl = y_top - delta_height
        new_x_br = x_bottom + delta_width
        new_y_br = y_bottom + delta_height
        rescaled_bboxes.append((new_x_tl, new_y_tl, new_x_br, new_y_br))

    return rescaled_bboxes


def filter_normalize(image, scale=2):
    """
    This function, filter_normalize, takes an input image and performs a series of image processing operations to filter and normalize it.
    Input:
    * image - Image to normalize and resize
    * scale - scale factor of resize
    Output:
    * resized_image - resized and normalized image
    """
    # Step 1: Denoise the colored image
    img = cv2.fastNlMeansDenoisingColored(image, None, 5, 10, 7, 15)

    # Step 2: Convert the denoised image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Create an empty matrix for normalized image
    norm_img = np.zeros((image.shape[0], image.shape[1]))

    # Step 4: Normalize the grayscale image
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

    # Step 5: Resize the image based on the specified scale
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image


def filter_bboxes(bboxes_list, min_ratio=1):
    """
    This function, filter_bboxes, is designed to filter a list of bounding boxes based on their aspect ratios.
    Input:
    * bboxes_list [List] - A list of bounding boxes
    * min-ratio - threshold to filter bounding boxes
    Output:
    * filtered_bboxes - bounding boxes that satisfy the filter requirement
    """
    # Initialize an empty list to store the filtered bounding boxes
    filtered_bboxes = []

    # Iterate through each bounding box in the input list
    for bbox in bboxes_list:
        # Unpack the bounding box coordinates
        x_topleft, y_topleft, x_bottomright, y_bottomright = bbox

        # Calculate the width and height of the bounding box
        width = x_bottomright - x_topleft
        height = y_bottomright - y_topleft

        # Calculate the aspect ratio of the bounding box
        ratio = width / height

        # Check if the aspect ratio is greater than or equal to the specified minimum ratio
        if ratio >= min_ratio:
            # If the condition is met, add the bounding box to the filtered list
            filtered_bboxes.append(bbox)

    # Return the list of filtered bounding boxes
    return filtered_bboxes


def is_integer(s):
    """
    This function checks whether a given string can be converted to an integer.
    Input:
    * s: a string
    Output:
    * boolean: whether a string can be converted to an integer
    """
    try:
        # Attempt to convert the input string to an integer
        int(s)
        # If successful, return True, indicating that it's an integer
        return True
    except ValueError:
        # If a ValueError is raised during the conversion, return False, indicating it's not an integer
        return False


def valid_plate(plate):
    """
    This function checks if a given license plate follows a specific format.
    Input:
    * plate [String]: a License plate
    Output:
    * Boolean: whether a plate follows a format
    """
    # Calculate the length of the input license plate
    length = len(plate)

    # Extract the first three characters and last two characters of the license plate
    nums, last = plate[:3], plate[-2:]

    # Initialize a variable for the middle part of the license plate
    mid = ""

    # Determine the middle part of the license plate based on its length
    if length == 7:
        mid = plate[-4:-2]

    elif length == 8:
        mid = plate[-5:-2]
    else:
        # If the length is not 7 or 8, return False as it doesn't match the expected format
        return False

    # Check conditions for a valid license plate
    cond1 = is_integer(nums) and mid.isalpha() and is_integer(last)

    # Initialize and check a condition for the last two characters
    cond2 = False
    if is_integer(last):
        cond2 = int(last) < 16 and int(last) > 0
        cond2 = True

    cond3 = plate[0].isalpha() and is_integer(plate[1:])

    # Return True if both conditions are met, indicating a valid license plate, otherwise return False
    if (cond1 and cond2) or (cond2 and cond3):
        return True
    else:
        return False
