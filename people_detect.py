import cv2
import numpy as np

# Load YOLO model
WEIGHTS_PATH = "/home/ellakiya/Opencv/yolov4.weights"
CFG_PATH = "/home/ellakiya/Opencv/yolov4.cfg"
COCO_PATH = "coco.names"

# Check if model files exist
try:
    net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
except Exception as e:
    print("Error loading YOLO model. Check file paths!")
    print(e)
    exit()

# Load class names
with open(COCO_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Check if "person" is in classes
PERSON_CLASS_ID = classes.index("person") if "person" in classes else 0
print(f"Person class ID: {PERSON_CLASS_ID}")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# People Tracking
trackers = []
tracked_ids = set()
person_id_counter = 0


# Function to detect objects using YOLO
def detect_people(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == PERSON_CLASS_ID and confidence > 0.3:  # Lowered threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detections.append((x, y, w, h))
    return detections


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    detections = detect_people(frame)
    current_count = len(detections)

    # Draw bounding boxes and display counts
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update total count
    total_people_detected = len(detections) if current_count > 0 else 0

    # Display count
    cv2.putText(frame, f"Current Count: {current_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Count: {total_people_detected}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("People Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
