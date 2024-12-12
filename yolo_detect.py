import cv2

# Paths to YOLO model files
cfg_path = "yolov4.cfg"
weights_path = "yolov4.weights"
names_path = "coco.names"

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNet(weights_path, cfg_path)

# Use GPU if available (optional)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set camera screen to larger size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Prepare input blob for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output layers
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    # Loop over detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":  # Threshold and detect only 'person'
                # Get bounding box coordinates
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {int(confidence * 100)}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the result in a larger window
    cv2.namedWindow("YOLO Person Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLO Person Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
