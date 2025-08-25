🕷️🤖 Spider Bot – Obstacle Detection & Voice Feedback

An AI-powered Spider Bot that detects obstacles in real time using YOLOv5/YOLOv8, estimates their distance and size, provides voice alerts, and logs detection history.
Supports both CLI mode (OpenCV + TTS) and an interactive Streamlit web app with Telegram integration.

✨ Features

📷 Real-time Obstacle Detection using YOLOv5 / YOLOv8

🎙️ Voice Feedback with pyttsx3 (alerts when an obstacle is too close)

🏛️ Object Tracking (enter/exit view notifications)

📏 Distance & Size Estimation from bounding box

💾 Detection Logs – Saves obstacle images & data to CSV

📲 Telegram Integration – Sends obstacle images to your phone

🖥️ Streamlit Web App – Start/stop detection, view live feed, logs & downloads

📦 Requirements

Ensure you have Python 3.8+ installed.

Install dependencies
pip install -r requirements.txt

requirements.txt
torch
opencv-python
pyttsx3
streamlit
ultralytics
pandas
numpy
python-telegram-bot

▶️ Usage
🔹 CLI Mode (YOLOv5 + OpenCV + TTS)
python spider_cli.py


Opens webcam feed

Detects obstacles (person, car, chair, etc.)

Announces detections with voice alerts

Stops when obstacle is too close

Press q to quit

🔹 Streamlit Web App (YOLOv8 + Advanced UI)
streamlit run spider_app.py


Features in UI:

▶️ Start/Stop Detection

📷 Live Camera Feed with bounding boxes

🎙️ Voice alerts for new obstacles & close warnings

📲 Send obstacle snapshots to Telegram (set TELEGRAM_TOKEN & TELEGRAM_USER_ID as environment variables)

📝 Obstacle Log with size, distance, location & timestamp

📅 Download detection logs as CSV

📂 Project Structure
spider_bot/
│── spider_cli.py          # CLI version (YOLOv5 + OpenCV + TTS)
│── spider_app.py          # Streamlit version (YOLOv8 + UI + Telegram)
│── obstacle_logs/         # Stores snapshots & CSV logs
│── requirements.txt       # Dependencies
│── README.md              # Documentation

⚙️ Configuration

YOLOv5 Model (CLI mode)

Place YOLOv5 repo at:

/Users/abhaygupta/Spider_Bot/yolov5


The script loads pretrained yolov5s.pt

Telegram Alerts (Optional, Streamlit)

Set environment variables:

export TELEGRAM_TOKEN="your-telegram-bot-token"
export TELEGRAM_USER_ID="your-chat-id"

📊 Detection Logs

Each detected obstacle is logged with:

Label (e.g., person, car)

Size in pixels & cm

Estimated distance (cm)

Timestamp

Location (customizable)

Snapshot image

CSV export is available via Streamlit.
