# Drowsiness Detector

A lightweight, local drowsiness detection toolkit that uses your webcam, MediaPipe Face Mesh for facial landmarks, and the Eye Aspect Ratio (EAR) to detect prolonged eye closure (a common drowsiness indicator).

Features

- Real-time webcam detection using OpenCV and MediaPipe
- Eye Aspect Ratio (EAR) computation from facial landmarks
- Configurable sensitivity (EAR threshold and consecutive-frame window)
- Optional audible alarm when drowsiness is detected
- Streamlit dashboard for quick visual monitoring and parameter tuning

Contents

- drowsiness_detector.py — Main OpenCV application that captures webcam frames, computes EAR, shows live video and plays an alarm when drowsiness is detected.
- streamlit_app.py — Streamlit dashboard for interactive monitoring and parameter tuning (useful for demos and remote viewing).
- utils.py — Utility functions for converting MediaPipe landmarks and computing EAR.
- requirements.txt — Python dependencies.

Installation

1. Clone or download this repository and open a terminal in the project folder.

2. Create and activate a virtual environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\activate

3. Install dependencies:

   pip install -r requirements.txt

Running the OpenCV detector

Usage:

   python drowsiness_detector.py [--threshold 0.25] [--consec_frames 48] [--camera 0] [--alarm]

Options:

- --threshold: EAR threshold below which an eye is considered "closed". Default: 0.25
- --consec_frames: Number of consecutive frames below threshold to trigger alarm. Default: 48
- --camera: Camera index to use (default 0)
- --alarm: Enable audible alarm when drowsiness is detected

Notes:

- Tweak the threshold and consecutive frames to suit your camera and lighting. Typical values: threshold 0.20–0.30, consec_frames 15–60.
- The script uses MediaPipe Face Mesh and does not require any external model files.

Running the Streamlit dashboard

Start the dashboard with:

   streamlit run streamlit_app.py

The dashboard allows interactive tuning of threshold and consecutive-frame window and displays the live camera feed with EAR and status.

Troubleshooting

- If the camera cannot be opened, ensure no other application (e.g., Zoom) is using the camera and the correct camera index is selected.
- If audio alarm does not play, ensure the optional simpleaudio dependency is installed and your system audio is available.

Privacy & Safety

All processing is done locally — no frames or data are uploaded.

License

MIT License

Contributing

Contributions are welcome — please open issues or pull requests with improvements, e.g., a better alarm, logging, or evaluation on recorded videos.

Contact

For questions, reach out via GitHub issues on the repository.
