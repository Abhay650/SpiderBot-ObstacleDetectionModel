import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time
import pyttsx3
import os
from datetime import datetime
import pandas as pd
from telegram import Bot
import atexit

# --- CONFIGURATION ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_USER_ID = int(os.getenv("TELEGRAM_USER_ID", ""))

try:
    bot = Bot(token=TELEGRAM_TOKEN)
except Exception as e:
    st.error(f"Failed to initialize Telegram Bot: {e}")

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="Spider Bot Obstacle Detection", layout="wide")
st.title("🤖 Spider Bot Obstacle Detection")

# --- LOAD YOLO MODEL ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# --- SPEECH ENGINE CONFIGURATION ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
speech_queue = queue.Queue(maxsize=10)

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
        speech_queue.task_done()

if "speech_thread" not in st.session_state:
    threading.Thread(target=speech_worker, daemon=True).start()
    st.session_state.speech_thread = True

def speak_bot(msg):
    if not speech_queue.full():
        speech_queue.put(msg)

# --- THREAD-SAFE FRAME CAPTURE ---

frame_queue = queue.Queue(maxsize=1)

def capture_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Keep only latest frame in queue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

# Initialize camera
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

cap = st.session_state.cap

if not cap.isOpened():
    st.error("Cannot open webcam")
    st.stop()

# Start frame capture thread once
if "capture_thread" not in st.session_state:
    t = threading.Thread(target=capture_frames, args=(cap,), daemon=True)
    t.start()
    st.session_state.capture_thread = t

# --- TELEGRAM IMAGE SEND FUNCTION ---
def send_image_to_telegram(image_path, caption="Obstacle detected"):
    try:
        if TELEGRAM_TOKEN:
            bot.send_photo(chat_id=TELEGRAM_USER_ID, photo=open(image_path, 'rb'), caption=caption)
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram image: {e}")

# --- UI ELEMENTS ---
start_button = st.button("▶️ Start Detection")
stop_button = st.button("⏹ Stop Detection")
frame_placeholder = st.empty()
info_placeholder = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False

if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False
    if "cap" in st.session_state:
        st.session_state.cap.release()
        del st.session_state["cap"]

# --- SETUP DATA STORAGE ---
if not os.path.exists("obstacle_logs"):
    os.makedirs("obstacle_logs")

if "detected_data" not in st.session_state:
    st.session_state.detected_data = []
if "previous_obstacles" not in st.session_state:
    st.session_state.previous_obstacles = set()

def estimate_distance(area):
    # Simplified example estimation (you can improve this)
    return max(50 - area / 1500, 10)

def process_frame(frame):
    # Resize frame for faster detection (optional)
    small_frame = cv2.resize(frame, (640, 360))

    results = model(small_frame)

    if not results or results[0].boxes is None:
        return frame, [], set()

    detections = results[0].boxes
    info_lines = []
    current_obstacles = set()

    # Scale boxes back to original frame size
    scale_x = frame.shape[1] / small_frame.shape[1]
    scale_y = frame.shape[0] / small_frame.shape[0]

    for box in detections:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0]
        # Scale back
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        area = (x2 - x1) * (y2 - y1)
        dist = estimate_distance(area)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {dist:.0f}cm", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current_obstacles.add(label)

        if label not in st.session_state.previous_obstacles:
            speak_bot(f"{label} detected at approximately {dist:.0f} centimeters")
            speak_bot(f"Size is approximately {round(area * 0.05, 1)} centimeters")

        if dist < 30:
            speak_bot("Obstacle too close. Halting Spider Bot.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"obstacle_logs/{label}_{timestamp}.jpg"
        crop_img = frame[max(y1, 0):min(y2, frame.shape[0]), max(x1, 0):min(x2, frame.shape[1])]

        if crop_img.size > 0:
            try:
                cv2.imwrite(img_path, crop_img)
                if time.time() - st.session_state.get("last_sent_time", 0) > 10:
                    send_image_to_telegram(img_path, caption=f"{label.capitalize()} | Size: {round(area * 0.05,1)}cm | Distance: {dist:.0f}cm")
                    st.session_state.last_sent_time = time.time()
            except Exception as e:
                print(f"Error saving/sending image: {e}")

        info_lines.append(f"**{label.capitalize()}** - Size: {area}px, Distance: {dist:.0f} cm")

        st.session_state.detected_data.append({
            'label': label,
            'size_px': area,
            'size_cm': round(area * 0.05, 1),
            'distance': round(dist, 1),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'location': "Lab Zone A",
            'image_path': img_path
        })

    # Announce obstacles leaving the frame
    for obs in st.session_state.previous_obstacles - current_obstacles:
        speak_bot(f"{obs} has exited the view.")

    st.session_state.previous_obstacles = current_obstacles
    return frame, info_lines, current_obstacles

# --- DETECTION LOOP ---
if st.session_state.running:
    if not frame_queue.empty():
        frame = frame_queue.get()
        processed_frame, info_lines, _ = process_frame(frame)

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, caption="Live Camera Feed", use_container_width=True)

        if info_lines:
            info_placeholder.markdown(
                "<div style='background:#112233; padding:10px; border-radius:8px; color:#aaffcc;'>"
                + "<br>".join(info_lines) + "</div>", unsafe_allow_html=True)
        else:
            info_placeholder.write("No obstacles detected.")
    else:
        info_placeholder.write("Waiting for camera frame...")

    # No explicit sleep to keep it responsive

else:
    st.info("Click 'Start Detection' to begin.")

# --- DETECTION LOG DISPLAY ---
detected_data = st.session_state.get("detected_data", [])
if not st.session_state.running and detected_data:
    st.markdown("---")
    st.markdown("### 📝 Obstacle Detection Log")

    for entry in detected_data:
        st.image(entry['image_path'], width=200, caption=f"{entry['label']} @ {entry['time']}")
        st.write(f"""
        **Obstacle:** {entry['label'].capitalize()}  
        **Size:** {entry['size_cm']} cm  
        **Distance:** {entry['distance']} cm  
        **Location:** {entry['location']}  
        **Time:** {entry['time']}
        """)

    try:
        df = pd.DataFrame(detected_data)
        csv_path = "obstacle_logs/obstacle_log.csv"
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            st.download_button(
                label="📅 Download Obstacle Log (CSV)",
                data=f,
                file_name="obstacle_log.csv",
                mime="text/csv"
            )
        st.success(f"Detection log saved to: {csv_path}")
    except Exception as e:
        st.error(f"Error saving log: {e}")

# --- CLEANUP ON EXIT ---
@atexit.register
def cleanup():
    if "cap" in st.session_state:
        st.session_state.cap.release()
