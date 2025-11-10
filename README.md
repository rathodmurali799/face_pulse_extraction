💓 Face Pulse Wave Extraction
Using MediaPipe, OpenCV, and Tkinter GUI
This project detects a person’s face using a webcam or video file and extracts a pulse wave signal by tracking subtle color changes on the skin.
It uses MediaPipe Face Mesh for facial landmark detection, YCgCo color conversion for green-channel analysis, and a Tkinter GUI for real-time visualization.

🧠 Overview

Human skin slightly changes color with every heartbeat.
This project captures those small variations from the face and converts them into a visible pulse wave.
It also estimates the phase difference and Pulse Transit Time (PTT) between two regions (left and right cheeks).
Key Features

🎥 Works with Webcam or Uploaded Video
🧍 Real-time Face Mesh Detection using MediaPipe
💚 Extracts Cg component from YCgCo color space
📊 Displays Live Pulse Waveform (left & right cheeks)
📁 Option to Save & Analyze extracted data
🔍 Computes Phase Difference and PTT
🧰 Simple and responsive Tkinter GUI

Requirements
opencv-python
mediapipe
numpy
pandas
pillow
matplotlib
scipy

How It Works

Face Detection
MediaPipe locates facial landmarks and automatically selects two cheek regions.
Signal Extraction
Each region’s color is converted to YCgCo space, and the Cg (green chroma) signal is averaged over time.
Filtering & Smoothing
The signal passes through a band-pass filter (0.7–3.0 Hz) to isolate the heart rate range.
Visualization
The filtered signals from both cheeks are plotted in real time on a custom Tkinter canvas.
Analysis
After saving, the program computes dominant frequency, phase difference, and Pulse Transit Time (PTT) between both ROIs.
