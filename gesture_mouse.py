"""
AI Virtual Gesture Mouse (GUI, single-file, beginner-friendly)
---------------------------------------------------------------
Features:
- Tkinter GUI with Start / Stop controls
- Live camera preview with MediaPipe landmarks
- Cursor movement from index fingertip
- Left click from thumb-index pinch
- Right click from index+middle up, ring+pinky down
- Adaptive smoothing for lower lag + better stability
"""

import time
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
import pyautogui
from PIL import Image, ImageTk


# -----------------------------
# Basic settings (easy to tweak)
# -----------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_MARGIN = 70

MIN_SMOOTHING = 2.0
MAX_SMOOTHING = 7.0
FAST_MOVEMENT_PIXELS = 55

LEFT_CLICK_RATIO = 0.33
LEFT_CLICK_FRAMES = 3
LEFT_CLICK_COOLDOWN = 0.30

RIGHT_CLICK_FRAMES = 4
RIGHT_CLICK_COOLDOWN = 0.55


# -----------------------------
# Math + gesture helpers
# -----------------------------
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def fingers_up(landmarks):
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    pips = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
    return {name: landmarks[tips[name]][1] < landmarks[pips[name]][1] for name in tips}


def adaptive_smoothing(raw_x, raw_y, prev_x, prev_y):
    motion = distance((raw_x, raw_y), (prev_x, prev_y))

    if motion >= FAST_MOVEMENT_PIXELS:
        smoothing = MIN_SMOOTHING
    else:
        ratio = motion / FAST_MOVEMENT_PIXELS
        smoothing = MAX_SMOOTHING - (MAX_SMOOTHING - MIN_SMOOTHING) * ratio

    x = prev_x + (raw_x - prev_x) / smoothing
    y = prev_y + (raw_y - prev_y) / smoothing
    return x, y


def map_to_screen(index_tip, frame_w, frame_h, screen_w, screen_h):
    clamped_x = min(max(index_tip[0], FRAME_MARGIN), frame_w - FRAME_MARGIN)
    clamped_y = min(max(index_tip[1], FRAME_MARGIN), frame_h - FRAME_MARGIN)

    raw_x = (clamped_x - FRAME_MARGIN) * screen_w / (frame_w - 2 * FRAME_MARGIN)
    raw_y = (clamped_y - FRAME_MARGIN) * screen_h / (frame_h - 2 * FRAME_MARGIN)
    return raw_x, raw_y


def extract_landmarks(hand_landmarks, frame_w, frame_h):
    return [
        (int(landmark.x * frame_w), int(landmark.y * frame_h))
        for landmark in hand_landmarks.landmark
    ]


# -----------------------------
# GUI app
# -----------------------------
class GestureMouseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Virtual Gesture Mouse")
        self.root.geometry("900x640")

        self.status_var = tk.StringVar(value="Status: Idle")

        header = ttk.Label(root, text="AI Virtual Gesture Mouse", font=("Segoe UI", 18, "bold"))
        header.pack(pady=(10, 4))

        help_text = (
            "Move: index finger | Left click: thumb-index pinch | "
            "Right click: index+middle up, ring+pinky down"
        )
        ttk.Label(root, text=help_text).pack(pady=(0, 10))

        btn_row = ttk.Frame(root)
        btn_row.pack(pady=6)

        self.start_btn = ttk.Button(btn_row, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = ttk.Button(btn_row, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6)

        self.video_label = ttk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        self.status_label = ttk.Label(root, textvariable=self.status_var, font=("Segoe UI", 10))
        self.status_label.pack(pady=(0, 10))

        # Runtime resources
        self.cap = None
        self.hands = None
        self.mp_hands = None
        self.mp_draw = None

        # Gesture runtime state
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

    def setup_tracker(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

    def setup_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return self.cap.isOpened()

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

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Status: Running")

        self.update_frame()

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Status: Stopped")
        self.cleanup_resources()

    def on_close(self):
        self.running = False
        self.cleanup_resources()
        self.root.destroy()

    def cleanup_resources(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        self.video_label.configure(image="")
        self.video_label.image = None

    def process_clicks(self, finger_state, pinch_ratio, frame):
        # Left click
        if pinch_ratio < LEFT_CLICK_RATIO:
            self.left_pinch_frames += 1
        else:
            self.left_pinch_frames = 0

        if self.left_pinch_frames >= LEFT_CLICK_FRAMES:
            now = time.time()
            if now - self.last_left_click_time > LEFT_CLICK_COOLDOWN:
                pyautogui.click(button="left")
                self.last_left_click_time = now
                cv2.putText(frame, "LEFT CLICK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 3)
            self.left_pinch_frames = 0

        # Right click
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
                cv2.putText(frame, "RIGHT CLICK", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 3)
            self.right_gesture_frames = 0

    def update_frame(self):
        if not self.running or self.cap is None or self.hands is None:
            return

        success, frame = self.cap.read()
        if not success:
            self.status_var.set("Status: Failed to read camera frame")
            self.stop()
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        frame_h, frame_w, _ = frame.shape

        cv2.rectangle(
            frame,
            (FRAME_MARGIN, FRAME_MARGIN),
            (frame_w - FRAME_MARGIN, frame_h - FRAME_MARGIN),
            (255, 0, 255),
            2,
        )

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            lm = extract_landmarks(hand_landmarks, frame_w, frame_h)
            thumb_tip, index_tip, middle_tip = lm[4], lm[8], lm[12]

            raw_x, raw_y = map_to_screen(index_tip, frame_w, frame_h, self.screen_w, self.screen_h)
            mouse_x, mouse_y = adaptive_smoothing(raw_x, raw_y, self.prev_mouse_x, self.prev_mouse_y)
            pyautogui.moveTo(mouse_x, mouse_y, duration=0)
            self.prev_mouse_x, self.prev_mouse_y = mouse_x, mouse_y

            finger_state = fingers_up(lm)
            hand_size = max(distance(lm[0], lm[9]), 1.0)
            pinch_ratio = distance(thumb_tip, index_tip) / hand_size

            self.process_clicks(finger_state, pinch_ratio, frame)

            cv2.circle(frame, thumb_tip, 8, (255, 255, 0), cv2.FILLED)
            cv2.circle(frame, index_tip, 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, middle_tip, 8, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, f"PinchRatio: {pinch_ratio:.2f}", (20, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=display)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # schedule next frame (~60fps max; actual depends on camera/CPU)
        self.root.after(15, self.update_frame)


def main():
    root = tk.Tk()
    app = GestureMouseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
