import cv2
import torch
import numpy as np

# Load YOLOv5 model
MODEL_PATH = "best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# Use webcam or video file
USE_LIVE = False  # Set to True for webcam
VIDEO_PATH = "Video Traffic Light8.mp4"
cap = cv2.VideoCapture(0 if USE_LIVE else VIDEO_PATH)

# HSV Ranges
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([35, 255, 255])
green_lower = np.array([40, 100, 100])
green_upper = np.array([85, 255, 255])

# Function to count colors in a region
def dominant_color(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    return {
        "Red": cv2.countNonZero(red_mask),
        "Yellow": cv2.countNonZero(yellow_mask),
        "Green": cv2.countNonZero(green_mask)
    }

# Zone-based traffic light color detection
def detect_color(crop):
    if crop.size == 0:
        return "Unknown"
    
    crop = cv2.resize(crop, (50, 150))  # Normalize size
    h = crop.shape[0]

    top = crop[0:h//3, :]
    middle = crop[h//3:2*h//3, :]
    bottom = crop[2*h//3:, :]

    top_counts = dominant_color(top)
    mid_counts = dominant_color(middle)
    bot_counts = dominant_color(bottom)

    if top_counts["Red"] > 50:
        return "Red"
    elif mid_counts["Yellow"] > 50:
        return "Yellow"
    elif bot_counts["Green"] > 50:
        return "Green"
    else:
        return "Unknown"

# Main detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxyn[0]

    for *box, conf, cls in detections:
        if conf < 0.3:  # Higher confidence threshold
            continue

        x1 = int(box[0] * frame.shape[1])
        y1 = int(box[1] * frame.shape[0])
        x2 = int(box[2] * frame.shape[1])
        y2 = int(box[3] * frame.shape[0])

        crop = frame[y1:y2, x1:x2]
        color = detect_color(crop)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{color} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Traffic Light Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
