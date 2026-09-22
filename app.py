import cv2
import mediapipe as mp
import math
import time
import winsound
import threading
from config import EAR_THRESHOLD, DROWSY_TIME
# -----------------------------
# MediaPipe Face Landmarker
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


# -----------------------------
# Eye Landmark Points
# -----------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# -----------------------------
# Settings
# -----------------------------

# -----------------------------
# Variables
# -----------------------------

eyes_closed_start = None
drowsiness_count = 0
was_drowsy = False

alarm_running = False


# -----------------------------
# Distance Function
# -----------------------------

def distance(point1, point2):

    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


# -----------------------------
# EAR Calculation
# -----------------------------

def calculate_ear(eye):

    vertical_1 = distance(
        eye[1],
        eye[5]
    )

    vertical_2 = distance(
        eye[2],
        eye[4]
    )

    horizontal = distance(
        eye[0],
        eye[3]
    )

    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )

    return ear


# -----------------------------
# Continuous Alarm
# -----------------------------

def alarm():

    global alarm_running

    while alarm_running:

        winsound.Beep(
            1000,
            500
        )

        time.sleep(0.1)


# -----------------------------
# Start Webcam
# -----------------------------

cap = cv2.VideoCapture(0)


# -----------------------------
# Start Face Landmarker
# -----------------------------

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Could not access camera"
            )

            break


        # -----------------------------
        # Convert BGR → RGB
        # -----------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # -----------------------------
        # MediaPipe Image
        # -----------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # -----------------------------
        # Detect Face
        # -----------------------------

        result = landmarker.detect(
            mp_image
        )


        # -----------------------------
        # Face Found
        # -----------------------------

        if result.face_landmarks:

            face = result.face_landmarks[0]


            left_points = []
            right_points = []


            # -----------------------------
            # Left Eye
            # -----------------------------

            for index in LEFT_EYE:

                landmark = face[index]

                x = int(
                    landmark.x *
                    frame.shape[1]
                )

                y = int(
                    landmark.y *
                    frame.shape[0]
                )

                left_points.append(
                    (x, y)
                )

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )


            # -----------------------------
            # Right Eye
            # -----------------------------

            for index in RIGHT_EYE:

                landmark = face[index]

                x = int(
                    landmark.x *
                    frame.shape[1]
                )

                y = int(
                    landmark.y *
                    frame.shape[0]
                )

                right_points.append(
                    (x, y)
                )

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )


            # -----------------------------
            # Calculate EAR
            # -----------------------------

            left_ear = calculate_ear(
                left_points
            )

            right_ear = calculate_ear(
                right_points
            )

            ear = (
                left_ear +
                right_ear
            ) / 2


            # -----------------------------
            # Display EAR
            # -----------------------------

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            # -----------------------------
            # Eyes Closed
            # -----------------------------

            if ear <= EAR_THRESHOLD:

                if eyes_closed_start is None:

                    eyes_closed_start = time.time()


                closed_duration = (
                    time.time()
                    - eyes_closed_start
                )


                cv2.putText(
                    frame,
                    f"Eyes Closed: "
                    f"{closed_duration:.1f}s",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


                # -----------------------------
                # Drowsiness
                # -----------------------------

                if closed_duration >= DROWSY_TIME:

                    if not was_drowsy:

                        drowsiness_count += 1

                        was_drowsy = True


                    cv2.putText(
                        frame,
                        "DROWSY!",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )


                    # -----------------------------
                    # Start Alarm
                    # -----------------------------

                    if not alarm_running:

                        alarm_running = True

                        alarm_thread = threading.Thread(
                            target=alarm,
                            daemon=True
                        )

                        alarm_thread.start()


                else:

                    cv2.putText(
                        frame,
                        "EYES CLOSED",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )


            # -----------------------------
            # Eyes Open
            # -----------------------------

            else:

                eyes_closed_start = None
                was_drowsy = False


                # Stop alarm

                alarm_running = False


                cv2.putText(
                    frame,
                    "NORMAL",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )


        # -----------------------------
        # No Face
        # -----------------------------

        else:

            eyes_closed_start = None
            was_drowsy = False

            alarm_running = False

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        # -----------------------------
        # Drowsiness Counter
        # -----------------------------

        cv2.putText(
            frame,
            f"Drowsiness Events: "
            f"{drowsiness_count}",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # -----------------------------
        # Show Window
        # -----------------------------

        cv2.imshow(
            "Driver Drowsiness Detection",
            frame
        )


        # -----------------------------
        # Press Q to Exit
        # -----------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            alarm_running = False

            break


# -----------------------------
# Release Resources
# -----------------------------

cap.release()

cv2.destroyAllWindows()