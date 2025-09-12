import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from mss import mss
from pynput.mouse import Listener, Button
from pynput.keyboard import Listener as KeyboardListener, Key
import threading
import ctypes
import math
import win32api
import win32con
import requests
import os
from termcolor import colored
import random

# =========================
# R6 MODEL CONFIGURATION
# =========================
R6_MODEL_CONFIG = {
    "workspace": "project-fwhag",
    "project": "r6-jymu7",
    "version": 1,
    "api_key": None,  # Set your Roboflow API key here or in environment variable
    "confidence": 0.23
}

# =========================
# CONFIG
# =========================
CONFIDENCE = 0.23
IMG_SIZE = 640
OVERLAY_COLOR = (0, 255, 0)
HEAD_COLOR = (0, 0, 255)

# Display settings - ENABLED for visibility
SHOW_DISPLAY_WINDOW = True
SAVE_DEBUG_IMAGES = False

# Performance Optimization
FOCUSED_CAPTURE = True  # Set to True for a huge FPS boost
CAPTURE_BOX_SIZE = 800   # Size of the square capture box (e.g., 800x800 pixels)

# R6-Optimized parameters
DEADZONE_RADIUS = 275
ONLY_AIM_IN_DEADZONE = True
SHOW_DEADZONE = True
RAINBOW_DEADZONE = True
RAINBOW_SPEED = 1.7

# R6-Specific auto-aim parameters
AUTO_AIM_ENABLED = True
TRIGGER_BOT_ENABLED = True
TRIGGER_THRESHOLD = 15
AIM_SPEED = 0.7
CONTINUOUS_TRACKING = True
TRACKING_UPDATE_RATE = 0.1
HEAD_HITBOX_EXPANSION = 15
PRIORITIZE_HEAD_SHOTS = True

MANUAL_AIM_KEY = 'v'

# =========================
# DEVICE
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# =========================
# MODEL LOADING
# =========================
def load_r6_model():
    """Load R6 model or fallback to object detection"""
    global model
    
    try:
        model_path = "r6_model.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"[INFO] Loaded local R6 model: {model_path}")
            return "r6_custom"
        
        api_key = R6_MODEL_CONFIG.get("api_key") or os.getenv("ROBOFLOW_API_KEY")
        if api_key:
            try:
                from roboflow import Roboflow
                rf = Roboflow(api_key=api_key)
                project = rf.workspace(R6_MODEL_CONFIG["workspace"]).project(R6_MODEL_CONFIG["project"])
                dataset = project.version(R6_MODEL_CONFIG["version"])
                model = dataset.model
                print(f"[INFO] Loaded Roboflow R6 model")
                return "roboflow_r6"
            except Exception as e:
                print(f"[WARNING] Failed to load Roboflow model: {e}")
        
        print(f"[INFO] Using YOLOv8 object detection model for R6 (fallback)")
        model = YOLO("yolov8n.pt")
        return "yolov8_object"
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model = YOLO("yolov8n.pt")
        return "yolov8_object"

# Load the model
current_model_type = load_r6_model()
model.to(device)
if hasattr(model, 'fuse'):
    model.fuse()

# =========================
# Screen Capture Setup
# =========================
sct = mss()
main_monitor = sct.monitors[1]
current_monitor = main_monitor.copy()

if FOCUSED_CAPTURE:
    capture_x = main_monitor['width'] // 2 - CAPTURE_BOX_SIZE // 2
    capture_y = main_monitor['height'] // 2 - CAPTURE_BOX_SIZE // 2
    current_monitor = {
        "top": capture_y,
        "left": capture_x,
        "width": CAPTURE_BOX_SIZE,
        "height": CAPTURE_BOX_SIZE,
    }
    SCREEN_CENTER_X = CAPTURE_BOX_SIZE // 2
    SCREEN_CENTER_Y = CAPTURE_BOX_SIZE // 2
else:
    SCREEN_CENTER_X = current_monitor['width'] // 2
    SCREEN_CENTER_Y = current_monitor['height'] // 2


# =========================
# Mouse Input Functions
# =========================
SendInput = ctypes.windll.user32.SendInput

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def move_mouse_direct_input(dx, dy):
    """Move mouse using direct input"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(dx, dy, 0, 0x0001, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def mouse_down():
    """Simulate a left mouse button press (hold)"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def mouse_up():
    """Simulate a left mouse button release"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def get_cursor_position():
    """
    Always returns the screen center to simulate a fixed cursor.
    """
    return SCREEN_CENTER_X, SCREEN_CENTER_Y

# =========================
# R6 Smooth Aim System
# =========================
class R6SmoothAim:
    def __init__(self):
        self.last_target = None
        self.tracking_start_time = None
        self.last_click_time = 0
        
    def calculate_distance_to_cursor(self, target_x, target_y):
        """Calculate distance from current cursor position to target"""
        cursor_x, cursor_y = get_cursor_position()
        distance = math.sqrt((target_x - cursor_x)**2 + (target_y - cursor_y)**2)
        return distance, cursor_x, cursor_y
    
    def calculate_smooth_movement(self, target_x, target_y, use_aim_speed=False):
        """Calculate smooth movement towards target"""
        cursor_x, cursor_y = get_cursor_position()
        
        raw_move_x = target_x - cursor_x
        raw_move_y = target_y - cursor_y
        distance = (raw_move_x**2 + raw_move_y**2)**0.5
        
        if distance == 0:
            return 0, 0, 0
        
        if use_aim_speed:
            if distance <= 20:
                speed_factor = 0.8
            elif distance <= 50:
                speed_factor = 0.5
            elif distance <= 100:
                speed_factor = 0.3
            else:
                speed_factor = 0.2
                
            move_x = int(raw_move_x * speed_factor)
            move_y = int(raw_move_y * speed_factor)
        else:
            sensitivity = 0.5 if distance <= 30 else 0.4 if distance <= 80 else 0.3
            move_x = round(raw_move_x * sensitivity)
            move_y = round(raw_move_y * sensitivity)
        
        return move_x, move_y, distance
    
    def auto_aim_to_target(self, target_x, target_y):
        """Auto aim to target with smooth movement"""
        move_x, move_y, distance = self.calculate_smooth_movement(target_x, target_y, use_aim_speed=True)
        
        if distance <= TRIGGER_THRESHOLD + 5:
            return False
            
        max_move = 15
        if abs(move_x) > max_move:
            move_x = max_move if move_x > 0 else -max_move
        if abs(move_y) > max_move:
            move_y = max_move if move_y > 0 else -max_move
            
        if abs(move_x) > 1 or abs(move_y) > 1:
            move_mouse_direct_input(move_x, move_y)
            return True
            
        return False

# Initialize aim system
smooth_aim = R6SmoothAim()

# =========================
# Detection Processing
# =========================
def process_detections(results):
    """
    Process YOLO detections for R6 targets.
    Only includes detections classified as 'person' and prioritizes their heads.
    """
    detected_targets = []
    
    for r in results:
        if r.boxes is None:
            continue
            
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            
            if hasattr(model, 'names') and class_id in model.names:
                class_name = model.names[class_id]
            else:
                class_name = f"object_{class_id}"
            
            if 'person' not in class_name.lower():
                continue
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            head_y = y1 + int((y2 - y1) * 0.25)
            target_x, target_y = center_x, head_y
            target_type = "head"
            
            detected_targets.append({
                'center': (target_x, target_y),
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_name': class_name,
                'class_id': class_id,
                'target_type': target_type
            })
    
    return detected_targets

# =========================
# Rainbow Effect
# =========================
class RainbowGenerator:
    def __init__(self):
        self.hue = 0
    
    def get_color(self, speed=0.1):
        self.hue += speed
        self.hue %= 180
        hsv_color = np.uint8([[[self.hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

rainbow = RainbowGenerator()
# Text Rainbow Gen
from termcolor import colored

def rainbow_text(text):
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    result = ''
    for i, line in enumerate(text.splitlines()):
        color = colors[i % len(colors)]  # cycle through colors
        result += colored(line, color) + '\n'
    return result

#text
text = r"""
/$$   /$$  /$$$$$$  /$$     /$$ /$$$$$$  /$$$$$$$$
| $$  /$$/ /$$__  $$|  $$   /$$//$$__  $$|_____ $$ 
| $$ /$$/ | $$  \ $$ \  $$ /$$/| $$  \ $$     /$$/ 
| $$$$$/  | $$$$$$$$  \  $$$$/ | $$$$$$$$    /$$/  
| $$  $$  | $$__  $$   \  $$/  | $$__  $$   /$$/   
| $$\  $$ | $$  | $$    | $$   | $$  | $$  /$$/    
| $$ \  $$| $$  | $$    | $$   | $$  | $$ /$$$$$$$$
|__/  \__/|__/  |__/    |__/   |__/  |__/|________/
"""
textori = r"""
 /$$      /$$                 /$$                 /$$$$$$$               
| $$$    /$$$                | $$                | $$__  $$              
| $$$$  /$$$$  /$$$$$$   /$$$$$$$  /$$$$$$       | $$  \ $$ /$$   /$$ /$$
| $$ $$/$$ $$ |____  $$ /$$__  $$ /$$__  $$      | $$$$$$$ | $$  | $$|__/
| $$  $$$| $$  /$$$$$$$| $$  | $$| $$$$$$$$      | $$__  $$| $$  | $$    
| $$\  $ | $$ /$$__  $$| $$  | $$| $$_____/      | $$  \ $$| $$  | $$ /$$
| $$ \/  | $$|  $$$$$$$|  $$$$$$$|  $$$$$$$      | $$$$$$$/|  $$$$$$$|__/
|__/     |__/ \_______/ \_______/ \_______/      |_______/  \____  $$    
                                                            /$$  | $$    
                                                           |  $$$$$$/    
                                                            \______/     
"""




# =========================
# Utility Functions
# =========================
def is_target_in_deadzone(target_x, target_y):
    """Check if target is within deadzone"""
    distance_from_center = np.sqrt((target_x - SCREEN_CENTER_X)**2 + (target_y - SCREEN_CENTER_Y)**2)
    return distance_from_center <= DEADZONE_RADIUS

def find_best_target(detected_targets):
    """Find best target prioritizing proximity"""
    if not detected_targets:
        return None
    
    valid_targets = []
    
    for target in detected_targets:
        target_x, target_y = target['center']
        
        if ONLY_AIM_IN_DEADZONE and not is_target_in_deadzone(target_x, target_y):
            continue
            
        distance = np.sqrt((target_x - SCREEN_CENTER_X)**2 + (target_y - SCREEN_CENTER_Y)**2)
        
        priority = 100 + target['confidence'] * 30 + max(0, 300 - distance)
        
        valid_targets.append((target, priority, distance))
    
    if not valid_targets:
        return None
    
    valid_targets.sort(key=lambda x: x[1], reverse=True)
    return valid_targets[0][0]

# =========================
# Global Variables
# =========================
manual_aim_pressed = False
mouse_is_down = False
current_target = None
auto_aim_active = False
trigger_bot_active = False
scoping = False

def on_click(x, y, button, pressed):
    global scoping
    if button == Button.right:
        scoping = pressed
        # Note: The 'scoping' state is captured here but the screen scan
        # remains unaffected, ensuring full-screen detection.

def on_key_press(key):
    global manual_aim_pressed
    try:
        if hasattr(key, 'char') and key.char and key.char.lower() == MANUAL_AIM_KEY:
            manual_aim_pressed = True
        elif key == Key.esc:
            return False
    except AttributeError:
        if key == Key.esc:
            return False

def on_key_release(key):
    global manual_aim_pressed
    try:
        if hasattr(key, 'char') and key.char and key.char.lower() == MANUAL_AIM_KEY:
            manual_aim_pressed = False
        elif key == Key.esc:
            return False
    except AttributeError:
        if key == Key.esc:
            return False

def auto_aim_and_trigger_system(detected_targets):
    global current_target, auto_aim_active, trigger_bot_active, mouse_is_down
    
    if not detected_targets:
        current_target = None
        auto_aim_active = False
        trigger_bot_active = False
        
        if mouse_is_down:
            mouse_up()
            mouse_is_down = False
        return
    
    best_target = find_best_target(detected_targets)
    
    if best_target:
        target_x, target_y = best_target['center']
        current_target = (target_x, target_y)
        
        distance_to_cursor, _, _ = smooth_aim.calculate_distance_to_cursor(target_x, target_y)
        
        if TRIGGER_BOT_ENABLED and distance_to_cursor <= TRIGGER_THRESHOLD:
            trigger_bot_active = True
            if not mouse_is_down:
                mouse_down()
                mouse_is_down = True
            auto_aim_active = False
            return
        
        if mouse_is_down:
            mouse_up()
            mouse_is_down = False
        
        trigger_bot_active = False
        
        if AUTO_AIM_ENABLED and distance_to_cursor > TRIGGER_THRESHOLD:
            auto_aim_active = True
            moved = smooth_aim.auto_aim_to_target(target_x, target_y)
        else:
            auto_aim_active = False
    else:
        current_target = None
        auto_aim_active = False
        trigger_bot_active = False
        
        if mouse_is_down:
            mouse_up()
            mouse_is_down = False

# =========================
# Main Loop
# =========================
frame_count = 0
last_time = time.time()
window_name = f"R6 Enhanced Aimbot + Triggerbot - {current_model_type.upper()} - Made By KAYAZ"

mouse_listener = Listener(on_click=on_click)
keyboard_listener = KeyboardListener(on_press=on_key_press, on_release=on_key_release)
mouse_listener.start()
keyboard_listener.start()

display_window_active = False
if SHOW_DISPLAY_WINDOW:
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 900)
        display_window_active = True
        print(f"[INFO] Display window created successfully!")
    except Exception as e:
        print(f"[WARNING] Could not create display window: {e}")
        display_window_active = False

try:
    print(rainbow_text(textori))
    print(rainbow_text(text))
    print(f"\n[INFO] Starting R6-Enhanced Aimbot + Triggerbot...")
    print(f"[INFO] Model: {current_model_type}")
    print(f"[INFO] Screen resolution: {current_monitor['width']}x{current_monitor['height']}")
    print(f"[INFO] Screen center: ({SCREEN_CENTER_X}, {SCREEN_CENTER_Y})")
    print(f"[INFO] Deadzone radius: {DEADZONE_RADIUS} pixels")
    print(f"[INFO] Auto-aim: {AUTO_AIM_ENABLED} | Trigger bot: {TRIGGER_BOT_ENABLED}")
    print(f"[INFO] Head shot priority: {PRIORITIZE_HEAD_SHOTS}")
    print(f"[INFO] Display window: {display_window_active}")
    print(f"[INFO] Focused Capture: {FOCUSED_CAPTURE} ({current_monitor['width']}x{current_monitor['height']})")
    print(f"[INFO] Confidence threshold: {CONFIDENCE}")
    print(f"[INFO] Trigger threshold: {TRIGGER_THRESHOLD} pixels")
    print("\n[🎮 CONTROLS - R6 OPTIMIZED]")
    print(f"- '{MANUAL_AIM_KEY.upper()}' key: Manual aim at nearest target")
    print("- RIGHT-CLICK: FREE for scoping! 🔍")
    print("- LEFT-CLICK: Manual shooting (or trigger bot)")
    print("- ESC key: Exit program")
    print("- Auto-aim: Automatically tracks heads within deadzone")
    print("- Trigger bot: Automatically shoots continuously when crosshair is on target")
    print("testie")
    print("=" * 50)
    
    while True:
        img = np.array(sct.grab(current_monitor))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        results = model(
            img_rgb,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device=device,
            verbose=False,
            half=True
        )

        detected_targets = process_detections(results)
        auto_aim_and_trigger_system(detected_targets)
        
        if display_window_active:
            display_img = img_rgb.copy()
            cursor_x, cursor_y = get_cursor_position()

            for target in detected_targets:
                x1, y1, x2, y2 = target['bbox']
                target_x, target_y = target['center']
                class_name = target['class_name']
                conf = target['confidence']
                target_type = target['target_type']
                
                color = (0, 255, 0)
                
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 3)
                
                target_color = (0, 0, 255)
                cv2.circle(display_img, (target_x, target_y), 8, target_color, -1)
                cv2.circle(display_img, (target_x, target_y), 12, (255, 255, 255), 3)
                
                if TRIGGER_BOT_ENABLED:
                    trigger_color = (255, 0, 255)
                    if trigger_bot_active and current_target and current_target == (target_x, target_y):
                        trigger_color = (0, 255, 255)
                    cv2.circle(display_img, (target_x, target_y), TRIGGER_THRESHOLD, trigger_color, 3)
                    
                    cv2.putText(display_img, "TRIGGER ZONE", (target_x - 50, target_y + TRIGGER_THRESHOLD + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, trigger_color, 2)
                
                label = f"{class_name} ({target_type}) {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_img, (x1, y1 - label_size[1] - 15),
                            (x1 + label_size[0] + 10, y1), (0, 0, 0), -1)
                cv2.putText(display_img, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if is_target_in_deadzone(target_x, target_y):
                    distance_to_cursor = math.sqrt((target_x - cursor_x)**2 + (target_y - cursor_y)**2)
                    
                    if distance_to_cursor <= TRIGGER_THRESHOLD:
                        dist_color = (0, 255, 255)
                        dist_text = f"{distance_to_cursor:.0f}px [🎯 TRIGGER READY]"
                    else:
                        dist_color = (255, 255, 255)
                        dist_text = f"{distance_to_cursor:.0f}px [TRACKING]"
                    
                    cv2.putText(display_img, dist_text, (target_x + 20, target_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)
            
            crosshair_color = (255, 255, 255)
            if trigger_bot_active:
                crosshair_color = (0, 255, 255)
            cv2.line(display_img, (SCREEN_CENTER_X - 25, SCREEN_CENTER_Y),
                    (SCREEN_CENTER_X + 25, SCREEN_CENTER_Y), crosshair_color, 4)
            cv2.line(display_img, (SCREEN_CENTER_X, SCREEN_CENTER_Y - 25),
                    (SCREEN_CENTER_X, SCREEN_CENTER_Y + 25), crosshair_color, 4)
            
            cursor_color = (255, 0, 255)
            if manual_aim_pressed:
                cursor_color = (0, 255, 0)
            cv2.circle(display_img, (SCREEN_CENTER_X, SCREEN_CENTER_Y), 10, cursor_color, 4)
            
            if SHOW_DEADZONE:
                deadzone_color = rainbow.get_color(RAINBOW_SPEED) if RAINBOW_DEADZONE else (255, 255, 0)
                cv2.circle(display_img, (SCREEN_CENTER_X, SCREEN_CENTER_Y),
                        DEADZONE_RADIUS, deadzone_color, 5)
                
                cv2.putText(display_img, "DEADZONE", (SCREEN_CENTER_X - 50, SCREEN_CENTER_Y - DEADZONE_RADIUS - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, deadzone_color, 2)
            
            if manual_aim_pressed and detected_targets:
                best_target = find_best_target(detected_targets)
                if best_target:
                    target_x, target_y = best_target['center']
                    if not ONLY_AIM_IN_DEADZONE or is_target_in_deadzone(target_x, target_y):
                        move_x, move_y, distance = smooth_aim.calculate_smooth_movement(target_x, target_y, use_aim_speed=False)
                        if move_x != 0 or move_y != 0:
                            move_mouse_direct_input(move_x, move_y)

            status_y = 30
            status_items = [
                f"MODEL: {current_model_type.upper()}",
                f"TARGETS: {len(detected_targets)}",
                f"FPS: {frame_count if time.time() - last_time < 1 else int(frame_count / (time.time() - last_time + 0.001))}"
            ]
            
            if AUTO_AIM_ENABLED:
                status_items.append("AUTO-AIM: ON")
                
            status_items.append(f"TRIGGER BOT: {'ON' if TRIGGER_BOT_ENABLED else 'OFF'}")
                
            if auto_aim_active:
                status_items.append("TRACKING ACTIVE")
            if trigger_bot_active:
                status_items.append("FIRING ACTIVE")

            for item in status_items:
                cv2.putText(display_img, item, (10, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                status_y += 30

            try:
                cv2.imshow(window_name, display_img)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            except Exception as e:
                print(f"[ERROR] Display error: {e}")
                break

        frame_count += 1
        
        current_time = time.time()
        if current_time - last_time >= 2:
            fps = frame_count / (current_time - last_time)
            print(f"[INFO] FPS: {fps:.1f} | Targets detected: {len(detected_targets)} | Model: {current_model_type}")
            if auto_aim_active:
                print(f"[INFO] Currently tracking target")
            frame_count = 0
            last_time = current_time

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C pressed - Exiting...")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
finally:
    if mouse_is_down:
        mouse_up()
    if display_window_active:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    mouse_listener.stop()
    keyboard_listener.stop()

    print("[INFO] R6-Enhanced Aimbot stopped cleanly")

