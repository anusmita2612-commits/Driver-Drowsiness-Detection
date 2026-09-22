import streamlit as st
import av
import cv2
import mediapipe as mp
import math
import time
import threading

# Windows alarm is available only on Windows
try:
    import winsound
except ImportError:
    winsound = None

from collections import deque
from streamlit_webrtc import webrtc_streamer
from config import EAR_THRESHOLD, DROWSY_TIME


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Driver Drowsiness AI",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🚗 Driver Drowsiness AI")

st.write(
    "Real-time computer vision system for driver alertness monitoring"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    st.write(
        "EAR Threshold:",
        EAR_THRESHOLD
    )

    st.write(
        "Drowsiness Time:",
        f"{DROWSY_TIME} seconds"
    )

    st.divider()

    st.header("🧠 AI Model")

    st.write(
        "MediaPipe Face Landmarker"
    )

    st.write(
        "Detects facial landmarks and tracks eye movement."
    )

    st.divider()

    st.header("🎨 Status Legend")

    st.write("🟢 NORMAL")
    st.write("🔴 EYES CLOSED")
    st.write("🔴 DROWSY")
    st.write("🟡 NO FACE")


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/face_landmarker.task"


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


landmarker = FaceLandmarker.create_from_options(
    options
)


# =========================================================
# EYE LANDMARKS
# =========================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# =========================================================
# SHARED VARIABLES
# =========================================================

state_lock = threading.Lock()

latest_ear = 0.0

latest_status = "NO FACE"

latest_events = 0

ear_history = deque(
    maxlen=100
)

event_history = []

eyes_closed_start = None

was_drowsy = False

alarm_running = False

alarm_thread = None


# =========================================================
# DISTANCE
# =========================================================

def distance(point1, point2):

    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


# =========================================================
# EAR CALCULATION
# =========================================================

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

    if horizontal == 0:

        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )


# =========================================================
# ALARM
# =========================================================

def alarm():

    global alarm_running

    while alarm_running:

        # Windows computer
        if winsound is not None:

            try:

                winsound.Beep(
                    1000,
                    500
                )

            except Exception:

                pass

        time.sleep(0.1)


def start_alarm():

    global alarm_running
    global alarm_thread

    if not alarm_running:

        alarm_running = True

        alarm_thread = threading.Thread(
            target=alarm,
            daemon=True
        )

        alarm_thread.start()


def stop_alarm():

    global alarm_running

    alarm_running = False


# =========================================================
# VIDEO CALLBACK
# =========================================================

def video_frame_callback(frame):

    global latest_ear
    global latest_status
    global latest_events

    global eyes_closed_start
    global was_drowsy

    # -----------------------------------------------------
    # GET CAMERA FRAME
    # -----------------------------------------------------

    img = frame.to_ndarray(
        format="bgr24"
    )

    height, width = img.shape[:2]


    # =====================================================
    # CREATE SMALLER FRAME FOR AI PROCESSING
    # =====================================================

    process_width = 640

    if width > process_width:

        scale = process_width / width

        process_height = int(
            height * scale
        )

        process_img = cv2.resize(
            img,
            (
                process_width,
                process_height
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        process_img = img


    # =====================================================
    # RGB CONVERSION
    # =====================================================

    rgb = cv2.cvtColor(
        process_img,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # =====================================================
    # MEDIAPIPE DETECTION
    # =====================================================

    result = landmarker.detect(
        mp_image
    )


    status = "NO FACE"

    ear_value = 0.0


    # =====================================================
    # FACE DETECTED
    # =====================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        process_height, process_width_actual = process_img.shape[:2]

        left_eye = []

        right_eye = []


        # -------------------------------------------------
        # LEFT EYE
        # -------------------------------------------------

        for index in LEFT_EYE:

            landmark = face_landmarks[index]

            x = int(
                landmark.x * process_width_actual
            )

            y = int(
                landmark.y * process_height
            )

            left_eye.append(
                (x, y)
            )


        # -------------------------------------------------
        # RIGHT EYE
        # -------------------------------------------------

        for index in RIGHT_EYE:

            landmark = face_landmarks[index]

            x = int(
                landmark.x * process_width_actual
            )

            y = int(
                landmark.y * process_height
            )

            right_eye.append(
                (x, y)
            )


        # -------------------------------------------------
        # EAR
        # -------------------------------------------------

        left_ear = calculate_ear(
            left_eye
        )

        right_ear = calculate_ear(
            right_eye
        )

        ear_value = (
            left_ear + right_ear
        ) / 2.0


        # =================================================
        # CONVERT LANDMARKS TO DISPLAY SIZE
        # =================================================

        x_scale = width / process_width_actual

        y_scale = height / process_height


        display_left_eye = []

        display_right_eye = []


        for x, y in left_eye:

            display_left_eye.append(
                (
                    int(x * x_scale),
                    int(y * y_scale)
                )
            )


        for x, y in right_eye:

            display_right_eye.append(
                (
                    int(x * x_scale),
                    int(y * y_scale)
                )
            )


        # =================================================
        # DRAW EYE POINTS
        # =================================================

        for point in display_left_eye:

            cv2.circle(
                img,
                point,
                2,
                (0, 255, 0),
                -1
            )


        for point in display_right_eye:

            cv2.circle(
                img,
                point,
                2,
                (0, 255, 0),
                -1
            )


        # =================================================
        # DROWSINESS DETECTION
        # =================================================

        if ear_value <= EAR_THRESHOLD:

            if eyes_closed_start is None:

                eyes_closed_start = time.time()


            closed_time = (
                time.time()
                -
                eyes_closed_start
            )


            if closed_time >= DROWSY_TIME:

                status = "DROWSY!"


                # -----------------------------------------
                # NEW DROWSINESS EVENT
                # -----------------------------------------

                if not was_drowsy:

                    latest_events += 1

                    was_drowsy = True


                    event_record = {
                        "Event": latest_events,
                        "Time": time.strftime("%H:%M:%S"),
                        "EAR": round(
                            ear_value,
                            3
                        ),
                        "Status": "DROWSY"
                    }


                    with state_lock:

                        event_history.append(
                            event_record
                        )


                # -----------------------------------------
                # START ALARM
                # -----------------------------------------

                start_alarm()


            else:

                status = "EYES CLOSED"


        # =================================================
        # EYES OPEN
        # =================================================

        else:

            status = "NORMAL"

            eyes_closed_start = None

            was_drowsy = False

            stop_alarm()


    # =====================================================
    # NO FACE
    # =====================================================

    else:

        status = "NO FACE"

        eyes_closed_start = None

        was_drowsy = False

        stop_alarm()


    # =====================================================
    # SHARED STATE
    # =====================================================

    with state_lock:

        latest_ear = ear_value

        latest_status = status

        ear_history.append(
            ear_value
        )


    # =====================================================
    # STATUS COLOR
    # =====================================================

    if status == "NORMAL":

        text_color = (
            0,
            255,
            0
        )

    elif (
        status == "EYES CLOSED"
        or
        status == "DROWSY!"
    ):

        text_color = (
            0,
            0,
            255
        )

    else:

        text_color = (
            0,
            255,
            255
        )


    # =====================================================
    # CAMERA TEXT
    # =====================================================

    cv2.putText(
        img,
        f"EAR: {ear_value:.3f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        img,
        f"STATUS: {status}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        img,
        f"DROWSINESS EVENTS: {latest_events}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # =====================================================
    # DROWSINESS WARNING
    # =====================================================

    if status == "DROWSY!":

        cv2.rectangle(
            img,
            (
                10,
                height - 75
            ),
            (
                width - 10,
                height - 10
            ),
            (0, 0, 255),
            -1
        )


        cv2.putText(
            img,
            "!!! WAKE UP !!!",
            (
                width // 2 - 130,
                height - 35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )


    # =====================================================
    # RETURN VIDEO
    # =====================================================

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# =========================================================
# LIVE CAMERA
# =========================================================

st.header(
    "📷 Live Camera Monitoring"
)


webrtc_streamer(

    key="drowsiness-camera",

    video_frame_callback=video_frame_callback,

    media_stream_constraints={

        "video": {

            "width": {
                "ideal": 640
            },

            "height": {
                "ideal": 480
            },

            "frameRate": {
                "ideal": 30
            }

        },

        "audio": False
    },

    async_processing=True
)


# =========================================================
# LIVE MONITORING
# =========================================================

st.header(
    "📊 Live Monitoring"
)


@st.fragment(run_every=0.5)
def live_monitoring():

    with state_lock:

        current_ear = latest_ear

        current_status = latest_status

        current_events = latest_events

        current_history = list(
            ear_history
        )

        current_event_history = list(
            event_history
        )


    # =====================================================
    # STATUS
    # =====================================================

    if current_status == "NORMAL":

        st.success(
            "🟢 SYSTEM STATUS: NORMAL"
        )

    elif current_status == "EYES CLOSED":

        st.error(
            "🔴 EYES CLOSED"
        )

    elif current_status == "DROWSY!":

        st.error(
            "🚨 DROWSINESS DETECTED — WAKE UP!"
        )

    else:

        st.warning(
            "🟡 NO FACE DETECTED"
        )


    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "CURRENT STATUS",
            current_status
        )


    with col2:

        st.metric(
            "EYE ASPECT RATIO",
            f"{current_ear:.3f}"
        )


    with col3:

        st.metric(
            "DROWSINESS EVENTS",
            current_events
        )


    # =====================================================
    # EAR GRAPH
    # =====================================================

    st.subheader(
        "📈 Eye Aspect Ratio"
    )


    if len(current_history) >= 2:

        st.line_chart(
            {
                "EAR": current_history
            },
            height=350
        )

    else:

        st.info(
            "Waiting for camera data..."
        )


    st.caption(
        "EAR Threshold: "
        + str(EAR_THRESHOLD)
        + " | Drowsiness Duration: "
        + str(DROWSY_TIME)
        + " seconds"
    )


    # =====================================================
    # EVENT HISTORY
    # =====================================================

    st.subheader(
        "📋 Drowsiness Event History"
    )


    if len(current_event_history) > 0:

        st.dataframe(
            current_event_history,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No drowsiness events recorded yet."
        )


live_monitoring()


# =========================================================
# HOW IT WORKS
# =========================================================

st.header(
    "🔍 How It Works"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.subheader(
        "👁 Face Tracking"
    )

    st.write(
        "MediaPipe detects facial landmarks "
        "around the driver's eyes."
    )


with col2:

    st.subheader(
        "📐 EAR Calculation"
    )

    st.write(
        "The Eye Aspect Ratio measures "
        "how open or closed the eyes are."
    )


with col3:

    st.subheader(
        "🚨 Alert System"
    )

    st.write(
        "If the eyes remain closed for "
        "the configured duration, "
        "an alarm is activated."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Driver Drowsiness AI • "
    "MediaPipe + OpenCV + Streamlit"
)