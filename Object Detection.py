import cv2
import torch
import pyttsx3
import threading
import queue

# Load YOLOv5 model from local path
try:
    model = torch.hub.load('/Users/abhaygupta/Spider_Bot/yolov5', 'yolov5s', source='local', pretrained=True)
    print("[INFO] YOLOv5 model loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load YOLOv5 model:", e)
    exit(1)

# COCO classes considered as obstacles
obstacle_classes = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck', 'chair', 'bottle', 'tvmonitor', 'bench']

# Initialize TTS engine
engine = pyttsx3.init()

# Optional: adjust volume or rate if needed
engine.setProperty('rate', 150)  # Speed
engine.setProperty('volume', 1.0)  # Volume max

# Thread-safe queue for speech
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        print(f"[TTS] Speaking: {text}")
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("[ERROR] pyttsx3 failed:", e)
        speech_queue.task_done()

# Start speech thread
threading.Thread(target=speech_worker, daemon=True).start()

# Estimate fake distance based on bounding box area
def estimate_distance(area):
    return max(50 - area / 1000, 10)

# For enter/exit tracking
previous_detections = set()

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb)
        detections = results.xyxy[0]

        current_detections = set()

        for *xyxy, conf, cls in detections:
            if conf < 0.4:
                continue

            label = model.names[int(cls)]
            if label not in obstacle_classes:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            area = (x2 - x1) * (y2 - y1)
            distance = estimate_distance(area)
            current_detections.add(label)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Speak on new detection
            if label not in previous_detections:
                msg = f"{label} entered view. Size {area:.0f} pixels. Estimated distance {distance:.0f} centimeters."
                speech_queue.put(msg)

            # Speak when too close
            if distance < 30:
                speech_queue.put("Obstacle too close. Halting Spider Bot.")

        # Speak exit
        for label in previous_detections - current_detections:
            speech_queue.put(f"{label} exited view.")

        previous_detections = current_detections

        cv2.imshow("Spider Bot Camera", frame)

        # Break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    speech_queue.join()
