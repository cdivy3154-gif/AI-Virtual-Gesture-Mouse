# AI Virtual Gesture Mouse

This repository contains a **single-file, beginner-friendly Python project** that lets you control your computer mouse with hand gestures using a webcam.

## What is included

- `gesture_mouse.py`: main application with a Tkinter GUI and real-time gesture tracking.
- `hand_landmarker.task`: MediaPipe hand landmark model used for gesture detection.

## Features

- Start/Stop GUI controls (Tkinter)
- Live webcam preview with hand landmarks
- Cursor movement based on index fingertip tracking
- Left click using thumb–index pinch
- Right click gesture (index+middle up, ring+pinky down)
- One-Euro Filter smoothing for more stable cursor motion
- MediaPipe `LIVE_STREAM` mode for low-latency detection

## Requirements

Install dependencies:

```bash
pip install opencv-python mediapipe pyautogui Pillow pywin32 numpy
```

> Note: `pywin32` indicates this project is primarily targeted at **Windows**.

## How to run

From the repository root:

```bash
python gesture_mouse.py
```

The app will open a desktop window where you can start/stop gesture mouse control.

## Gesture controls (default)

- **Move cursor:** move your index fingertip in front of the camera
- **Left click:** thumb + index pinch
- **Right click:** index and middle fingers up while ring and pinky are down

## Tuning

You can edit constants near the top of `gesture_mouse.py` to adjust behavior, such as:

- camera resolution and margin
- smoothing (`MIN_CUTOFF`, `BETA`, `D_CUTOFF`)
- click sensitivity, frame thresholds, and cooldowns

## Notes

- Good lighting and a clear background improve tracking quality.
- Webcam positioning affects comfort and accuracy.
- If the model file is missing, the script can download it automatically.
