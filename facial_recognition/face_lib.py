import os
from typing import List, Sequence

import dlib
import numpy as np
from skimage import io
from skimage import transform
from skimage import color
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Union


def shape2points(
    shape,
    dtype: str = "int",
    point_num: int = 68
) -> np.ndarray:
    """
    Transforms a dlib shape object to a numpy array.
    
    Arguments
    shape     : A dlib shape object from dlib predictor
    Optional
    dtype     : A string that defines the data type of the returned
                array. Defaults to int.
    point_num : The number of points in the shape object. Defaults to
                the 68 landmarks in face.
    Return
    np.ndarray containing the points in a (point_num, 2) shape
    
    Example usage
    shape = predictor(image, detections) # dlib predictor
    # Transform the dlib shape object to numpy array landmarks
    landmarks = shape2points(shape)
    """
    coords = np.zeros((point_num, 2), dtype=dtype)
    for i in range(point_num):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def crop_face(
    raw_image: np.ndarray,
    landmarks: np.ndarray,
    left_eye_inds: List[int] = [36, 37, 38, 39, 40, 41],
    right_eye_inds: List[int] = [42, 43, 44, 45, 46, 47],
    resize: Sequence[int] | None = (128, 128),
    #resize: Union[Sequence[int], None] = (128, 128),
) -> np.ndarray:
    """
    Crops a face from a given image.
    
    Arguments
    raw_image      : A numpy array image with the face in it
    landmarks      : A numpy array with the landmarks that correspond to the face
    Optional
    left_eye_inds  : A list of the landmark indices of the left eye. Defaults to
                     the indices from the 68 landmark model.
    right_eye_inds : A list of the landmark indices of the right eye. Defaults to
                     the indices from the 68 landmark model.
    resize         : A tuple of the width and height that the image will be resized
                     to. Defaults to (128, 128).
    Return
    Cropped image as a numpy array with size resize unless resize set to None.
    
    Example usage
    cropped_image = crop_face(raw_image, landmarks, resize=None)
    
    """

    # If not gray image, transform it
    if len(raw_image.shape) == 3 and raw_image.shape[2] == 3:
        raw_image = color.rgb2gray(raw_image)

    # Extract the left eye coordinate
    left_eye = landmarks[left_eye_inds, :].sum(0).astype("float") / len(left_eye_inds)
    # Extract the right eye coordinate
    right_eye = landmarks[right_eye_inds, :].sum(0).astype("float") / len(right_eye_inds)

    # Distance between two eyes
    dist_between_eyes = np.sqrt(sum((left_eye - right_eye) ** 2))

    x1 = left_eye[0]
    y1 = left_eye[1]
    x2 = right_eye[0]
    y2 = right_eye[1]

    sina = (y1 - y2) / dist_between_eyes
    cosa = (x2 - x1) / dist_between_eyes

    lefttopy = y1 + dist_between_eyes * 0.4 * sina - dist_between_eyes * 0.6 * cosa
    lefttopx = x1 - dist_between_eyes * 0.4 * cosa - dist_between_eyes * 0.6 * sina

    face_height = int(round(dist_between_eyes * 2.2))
    face_width = int(round(dist_between_eyes * 1.8))

    norm_face = np.zeros((face_height, face_width))

    [wi, hi] = raw_image.shape

    for h in range(0, face_height):
        starty = lefttopy + h * cosa
        startx = lefttopx + h * sina

        for w in range(0, face_width):
            if np.uint16(starty - w * sina) > wi:
                norm_face[h, w] = raw_image[np.uint16(wi), np.uint16(startx + w * cosa)]

            elif np.uint16(startx + w * cosa) > hi:
                norm_face[h, w] = raw_image[np.uint16(starty - w * sina), np.uint16(hi)]

            else:
                norm_face[h, w] = raw_image[
                    np.uint16(starty - w * sina), np.uint16(startx + w * cosa)
                ]

    if resize:
        norm_face = transform.resize(norm_face, resize)

    return norm_face