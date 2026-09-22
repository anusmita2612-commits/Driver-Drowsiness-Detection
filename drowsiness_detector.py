import argparse
import time
import platform

import cv2
import mediapipe as mp
import numpy as np

from utils import LEFT_EYE_IDX, RIGHT_EYE_IDX, eye_points_from_landmarks, compute_ear


def play_alarm(duration_ms: int = 700, freq_hz: int = 2000):
    """Play a short alarm sound. Tries winsound on Windows, otherwise uses simpleaudio if available.

    Non-blocking when using simpleaudio; winsound.Beep is blocking but short.
    """
    # Try Windows winsound first
    try:
        if platform.system() == "Windows":
            import winsound

            winsound.Beep(freq_hz, duration_ms)
            return
    except Exception:
        pass

    # Fallback to simpleaudio if installed
    try:
        import simpleaudio as sa

        fs = 44100
        t = np.linspace(0, duration_ms / 1000.0, int(fs * (duration_ms / 1000.0)), False)
        tone = np.sin(freq_hz * 2 * np.pi * t) * 0.5
        audio = (tone * (2 ** 15 - 1)).astype(np.int16)
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        # do not block; let it play in background
        return
    except Exception:
        # Last resort: no audio available
        return


def main(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    EAR_THRESHOLD = args.threshold
    CONSEC_FRAMES = args.consec_frames

    counter = 0
    alarm_on = False

    mp_face_mesh = mp.solutions.face_mesh

    # Create the FaceMesh within a context to avoid heavy initialization on import
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                h, w = frame.shape[:2]
                # Convert to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    # Only use the first face
                    face_landmarks = results.multi_face_landmarks[0].landmark

                    left_eye_pts = eye_points_from_landmarks(face_landmarks, LEFT_EYE_IDX, w, h)
                    right_eye_pts = eye_points_from_landmarks(face_landmarks, RIGHT_EYE_IDX, w, h)

                    ear_left = compute_ear(left_eye_pts)
                    ear_right = compute_ear(right_eye_pts)
                    ear = (ear_left + ear_right) / 2.0

                    # Draw eye contours
                    for p in left_eye_pts.astype(np.int32):
                        cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)
                    for p in right_eye_pts.astype(np.int32):
                        cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)

                    cv2.putText(frame, f"EAR: {ear:.3f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if ear < EAR_THRESHOLD:
                        counter += 1
                    else:
                        counter = 0
                        alarm_on = False

                    cv2.putText(frame, f"Consec: {counter}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if counter >= CONSEC_FRAMES:
                        if not alarm_on:
                            alarm_on = True
                            cv2.putText(frame, "DROWSINESS DETECTED", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (0, 0, 255), 3)
                            if args.alarm:
                                # play alarm in non-blocking manner where possible
                                play_alarm()
                        else:
                            # keep showing alert text
                            cv2.putText(frame, "DROWSINESS DETECTED", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (0, 0, 255), 3)

                # Show frame
                cv2.imshow('Drowsiness Detector (press q to exit)', frame)

                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Webcam drowsiness detector using EAR and MediaPipe Face Mesh')
    parser.add_argument('--threshold', type=float, default=0.25, help='EAR threshold to consider the eye closed')
    parser.add_argument('--consec_frames', type=int, default=48, help='Number of consecutive frames below threshold to trigger alarm')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    parser.add_argument('--alarm', action='store_true', help='Play a beep when drowsiness is detected')

    args = parser.parse_args()
    main(args)
