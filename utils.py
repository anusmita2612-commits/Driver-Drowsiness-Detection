import math
import numpy as np

# Eye indices chosen from MediaPipe Face Mesh to form the six points used by the EAR formula.
# Left eye indices (approx): [33, 160, 158, 133, 153, 144]
# Right eye indices (approx): [362, 385, 387, 263, 373, 380]
# These indices map to the typical positions p1..p6 in standard EAR formula.

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def landmark_to_point(landmark, image_width, image_height):
    """Convert a mediapipe landmark to (x, y) pixel coordinates."""
    return np.array([int(landmark.x * image_width), int(landmark.y * image_height)], dtype=np.float32)


def eye_points_from_landmarks(landmarks, eye_idx, image_width, image_height):
    """Return an array of 6 (x, y) points for the given eye indices.

    landmarks: list of mediapipe landmarks for one face
    eye_idx: list of 6 integer indices
    """
    pts = [landmark_to_point(landmarks[i], image_width, image_height) for i in eye_idx]
    return np.array(pts, dtype=np.float32)


def euclidean(a, b):
    return np.linalg.norm(a - b)


def compute_ear(eye_points):
    """Compute the Eye Aspect Ratio (EAR) for one eye.

    eye_points: np.array with shape (6, 2) ordered as [p1, p2, p3, p4, p5, p6]
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    # Ensure correct shape
    if eye_points.shape != (6, 2):
        raise ValueError("eye_points must be shape (6, 2)")

    p1, p2, p3, p4, p5, p6 = eye_points
    vert1 = euclidean(p2, p6)
    vert2 = euclidean(p3, p5)
    horiz = euclidean(p1, p4)

    if horiz == 0:
        return 0.0

    ear = (vert1 + vert2) / (2.0 * horiz)
    return float(ear)


def average_ear(left_eye_pts, right_eye_pts):
    """Compute average EAR across both eyes."""
    return (compute_ear(left_eye_pts) + compute_ear(right_eye_pts)) / 2.0
