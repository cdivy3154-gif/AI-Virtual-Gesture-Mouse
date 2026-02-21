"""
AI Virtual Gesture Mouse (GUI, single-file, beginner-friendly)
---------------------------------------------------------------
Features:
- Tkinter GUI with Start / Stop controls
- Live camera preview with MediaPipe landmarks
- Cursor movement from index fingertip
- Left click from thumb-index pinch
- Right click from index+middle up, ring+pinky down
- One-Euro Filter for ultra-smooth cursor movement
- LIVE_STREAM async detection for minimal latency

Requirements (pip install):
    pip install opencv-python mediapipe pyautogui Pillow pywin32

Model file:
    This script auto-downloads "hand_landmarker.task" on first run.
"""

import math
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import win32api
from PIL import Image, ImageTk


# ---------------------------------------------------------
# Auto-download the hand_landmarker model if not present
# ---------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")


def ensure_model():
    """Download the hand-landmarker model file if it does not exist."""
    if os.path.isfile(MODEL_PATH):
        return
    print(f"[INFO] Downloading hand_landmarker.task to {MODEL_PATH} ...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Download complete.")


# -----------------------------
# Basic settings (easy to tweak)
# -----------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_MARGIN = 70

# -- One-Euro Filter tuning --
# Lower MIN_CUTOFF  → smoother at rest (less jitter)
# Higher BETA       → faster reaction to sudden moves
MIN_CUTOFF = 1.5
BETA = 0.05
D_CUTOFF = 1.0            # derivative cutoff (rarely needs changing)

LEFT_CLICK_RATIO = 0.33
LEFT_CLICK_FRAMES = 3
LEFT_CLICK_COOLDOWN = 0.30

RIGHT_CLICK_FRAMES = 4
RIGHT_CLICK_COOLDOWN = 0.55


# -----------------------------------------------
# Convenience aliases for the new MediaPipe API
# -----------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
RunningMode = mp.tasks.vision.RunningMode
draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
DrawingSpec = mp.tasks.vision.drawing_utils.DrawingSpec


# =============================================================
# ONE-EURO FILTER  (gold-standard cursor smoother)
# =============================================================
# Reference: Géry Casiez et al., "1€ Filter: A Simple Speed-Based
# Low-Pass Filter for Noisy Input in Interactive Systems", 2012
# =============================================================
class LowPassFilter:
    """Simple exponential low-pass (EMA) filter."""

    def __init__(self):
        self.prev = None

    def apply(self, value, alpha):
        if self.prev is None:
            self.prev = value
        else:
            self.prev = alpha * value + (1 - alpha) * self.prev
        return self.prev

    def reset(self):
        self.prev = None


class OneEuroFilter:
    """
    One-Euro Filter for a single axis (x or y).
    - min_cutoff : lower → smoother when still (reduces jitter)
    - beta       : higher → snappier reaction to fast moves
    """

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, value, timestamp=None):
        now = timestamp if timestamp is not None else time.monotonic()
        if self.last_time is None:
            self.last_time = now
            self.x_filter.apply(value, 1.0)    # seed the filter
            self.dx_filter.apply(0.0, 1.0)
            return value

        dt = max(now - self.last_time, 1e-6)   # seconds since last call
        self.last_time = now

        # Estimate derivative (speed)
        dx = (value - (self.x_filter.prev or value)) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_smooth = self.dx_filter.apply(dx, alpha_d)

        # Adapt cutoff: fast motion → higher cutoff → less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        alpha = self._alpha(cutoff, dt)

        return self.x_filter.apply(value, alpha)

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


# -----------------------------
# Math + gesture helpers
# -----------------------------
def distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def fingers_up(landmarks):
    """
    Check which fingers are "up" (extended).
    Each finger is up if its tip is ABOVE (lower y value) its PIP joint.
    landmarks: list of (x, y) tuples, indexed 0-20.
    """
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    pips = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
    return {name: landmarks[tips[name]][1] < landmarks[pips[name]][1] for name in tips}


def map_to_screen(index_tip, frame_w, frame_h, screen_w, screen_h):
    """
    Map the index fingertip position (in the camera frame) to
    the full screen, clamping to FRAME_MARGIN on each side.
    """
    clamped_x = min(max(index_tip[0], FRAME_MARGIN), frame_w - FRAME_MARGIN)
    clamped_y = min(max(index_tip[1], FRAME_MARGIN), frame_h - FRAME_MARGIN)

    raw_x = (clamped_x - FRAME_MARGIN) * screen_w / (frame_w - 2 * FRAME_MARGIN)
    raw_y = (clamped_y - FRAME_MARGIN) * screen_h / (frame_h - 2 * FRAME_MARGIN)
    return raw_x, raw_y


def normalized_to_pixel(hand_landmarks, frame_w, frame_h):
    """
    Convert a list of NormalizedLandmarks (new API) to a list of
    (x_px, y_px) tuples in pixel coordinates.
    """
    return [
        (int(lm.x * frame_w), int(lm.y * frame_h))
        for lm in hand_landmarks
    ]


# ----------------------------------------------------------
# Draw hand landmarks on the frame (new Tasks-API helper)
# ----------------------------------------------------------
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

LANDMARK_SPEC = DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
CONNECTION_SPEC = DrawingSpec(color=(255, 0, 255), thickness=2)


def draw_hand_on_frame(frame, hand_landmarks_normalized):
    """
    Draw landmarks and connections onto a BGR frame using the
    new MediaPipe Tasks drawing utilities.
    """
    draw_landmarks(
        image=frame,
        landmark_list=hand_landmarks_normalized,
        connections=HAND_CONNECTIONS,
        landmark_drawing_spec=LANDMARK_SPEC,
        connection_drawing_spec=CONNECTION_SPEC,
    )


# -----------------------------
# GUI app
# -----------------------------
class GestureMouseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Virtual Gesture Mouse")
        self.root.geometry("900x640")

        self.status_var = tk.StringVar(value="Status: Idle")

        # ---------- header ----------
        header = ttk.Label(root, text="AI Virtual Gesture Mouse",
                           font=("Segoe UI", 18, "bold"))
        header.pack(pady=(10, 4))

        help_text = (
            "Move: index finger  |  Left click: thumb-index pinch  |  "
            "Right click: index+middle up, ring+pinky down"
        )
        ttk.Label(root, text=help_text).pack(pady=(0, 10))

        # ---------- buttons ----------
        btn_row = ttk.Frame(root)
        btn_row.pack(pady=6)

        self.start_btn = ttk.Button(btn_row, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = ttk.Button(btn_row, text="Stop", command=self.stop,
                                   state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6)

        # ---------- video preview ----------
        self.video_label = ttk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        self.status_label = ttk.Label(root, textvariable=self.status_var,
                                      font=("Segoe UI", 10))
        self.status_label.pack(pady=(0, 10))

        # ---------- runtime resources ----------
        self.cap = None
        self.hand_landmarker = None

        # ---------- LIVE_STREAM result (shared between threads) ----------
        self.latest_result = None
        self.result_lock = threading.Lock()

        # ---------- One-Euro Filters for X and Y ----------
        self.filter_x = OneEuroFilter()
        self.filter_y = OneEuroFilter()

        # ---------- gesture state ----------
        self.running = False
        self.prev_mouse_x = 0
        self.prev_mouse_y = 0
        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0
        self.left_pinch_frames = 0
        self.right_gesture_frames = 0

        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- LIVE_STREAM callback (runs on MediaPipe's thread) ----
    def _on_result(self, result, output_image, timestamp_ms):
        """
        Called asynchronously by MediaPipe when detection is done.
        We just store the latest result; the GUI loop picks it up.
        """
        with self.result_lock:
            self.latest_result = result

    # ---- set up the hand tracker (LIVE_STREAM for low latency) ----
    def setup_tracker(self):
        """
        Create HandLandmarker in LIVE_STREAM mode.
        Detection runs async — never blocks the main loop.
        """
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
            result_callback=self._on_result,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(options)

    # ---- set up the webcam ----
    def setup_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Minimise internal buffer to reduce frame staleness
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    # ---- start tracking ----
    def start(self):
        if self.running:
            return

        self.setup_tracker()
        if not self.setup_camera():
            self.status_var.set("Status: Camera open failed. Check permissions/index.")
            self.cleanup_resources()
            return

        self.prev_mouse_x, self.prev_mouse_y = pyautogui.position()
        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0
        self.left_pinch_frames = 0
        self.right_gesture_frames = 0

        # Reset the One-Euro Filters for a fresh session
        self.filter_x.reset()
        self.filter_y.reset()
        self.latest_result = None

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Status: Running")

        self.update_frame()

    # ---- stop tracking ----
    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Status: Stopped")
        self.cleanup_resources()

    # ---- window close ----
    def on_close(self):
        self.running = False
        self.cleanup_resources()
        self.root.destroy()

    # ---- release camera + model ----
    def cleanup_resources(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()
            self.hand_landmarker = None
        self.video_label.configure(image="")
        self.video_label.image = None

    # ---- click detection logic ----
    def process_clicks(self, finger_state, pinch_ratio, frame):
        # ---------- Left click (thumb-index pinch) ----------
        if pinch_ratio < LEFT_CLICK_RATIO:
            self.left_pinch_frames += 1
        else:
            self.left_pinch_frames = 0

        if self.left_pinch_frames >= LEFT_CLICK_FRAMES:
            now = time.time()
            if now - self.last_left_click_time > LEFT_CLICK_COOLDOWN:
                pyautogui.click(button="left")
                self.last_left_click_time = now
                cv2.putText(frame, "LEFT CLICK", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            self.left_pinch_frames = 0

        # ---------- Right click (index+middle up, ring+pinky down) ----------
        right_condition = (
            finger_state["index"]
            and finger_state["middle"]
            and not finger_state["ring"]
            and not finger_state["pinky"]
            and pinch_ratio > LEFT_CLICK_RATIO
        )

        if right_condition:
            self.right_gesture_frames += 1
        else:
            self.right_gesture_frames = 0

        if self.right_gesture_frames >= RIGHT_CLICK_FRAMES:
            now = time.time()
            if now - self.last_right_click_time > RIGHT_CLICK_COOLDOWN:
                pyautogui.click(button="right")
                self.last_right_click_time = now
                cv2.putText(frame, "RIGHT CLICK", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            self.right_gesture_frames = 0

    # ---- main per-frame processing loop ----
    def update_frame(self):
        if not self.running or self.cap is None or self.hand_landmarker is None:
            return

        # Flush stale frames: grab() discards the buffered frame,
        # then read() gets the freshest one from the camera.
        self.cap.grab()
        success, frame = self.cap.read()
        if not success:
            self.status_var.set("Status: Failed to read camera frame")
            self.stop()
            return

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape

        # --- Send frame to MediaPipe (non-blocking) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Use real monotonic timestamp for accurate tracking
        timestamp_ms = int(time.monotonic() * 1000)
        self.hand_landmarker.detect_async(mp_image, timestamp_ms)

        # --- Draw the active-zone rectangle ---
        cv2.rectangle(
            frame,
            (FRAME_MARGIN, FRAME_MARGIN),
            (frame_w - FRAME_MARGIN, frame_h - FRAME_MARGIN),
            (255, 0, 255),
            2,
        )

        # --- Grab the latest async result (thread-safe) ---
        with self.result_lock:
            result = self.latest_result

        # --- If a hand is detected, process gestures ---
        if result is not None and result.hand_landmarks:
            hand_lm_normalized = result.hand_landmarks[0]

            # Draw landmarks on the frame
            draw_hand_on_frame(frame, hand_lm_normalized)

            # Convert normalised landmarks → pixel coords for gesture math
            lm = normalized_to_pixel(hand_lm_normalized, frame_w, frame_h)
            thumb_tip, index_tip, middle_tip = lm[4], lm[8], lm[12]

            # --- Move cursor (One-Euro Filter + win32api) ---
            raw_x, raw_y = map_to_screen(
                index_tip, frame_w, frame_h, self.screen_w, self.screen_h
            )

            # Apply One-Euro Filter for buttery-smooth movement
            now = time.monotonic()
            mouse_x = self.filter_x.apply(raw_x, now)
            mouse_y = self.filter_y.apply(raw_y, now)

            # win32api.SetCursorPos is ~10× faster than pyautogui.moveTo
            try:
                win32api.SetCursorPos((int(mouse_x), int(mouse_y)))
            except Exception:
                pass   # ignore if coords are out of bounds momentarily
            self.prev_mouse_x, self.prev_mouse_y = mouse_x, mouse_y

            # --- Detect clicks ---
            finger_state = fingers_up(lm)
            hand_size = max(distance(lm[0], lm[9]), 1.0)
            pinch_ratio = distance(thumb_tip, index_tip) / hand_size

            self.process_clicks(finger_state, pinch_ratio, frame)

            # --- Draw key points and info ---
            cv2.circle(frame, thumb_tip, 8, (255, 255, 0), cv2.FILLED)
            cv2.circle(frame, index_tip, 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, middle_tip, 8, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, f"PinchRatio: {pinch_ratio:.2f}",
                        (20, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Show the frame in the Tkinter window ---
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=display)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule next frame as fast as possible (1ms idle)
        self.root.after(1, self.update_frame)


# -----------------------------
# Entry point
# -----------------------------
def main():
    ensure_model()          # auto-download model if needed
    root = tk.Tk()
    app = GestureMouseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
