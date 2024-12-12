# import cv2
# import numpy as np

# # Default camera parameters (approximate values for a typical webcam)
# camera_matrix = np.array([[800, 0, 320],
#                           [0, 800, 240],
#                           [0, 0, 1]], dtype=np.float32)
# dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # No distortion assumed

# # ArUco Dictionary and Marker Size
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# MARKER_SIZE = 0.05  # Marker size in meters (50 mm)

# # Start video capture from the default camera (laptop's built-in camera)
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Cannot access camera")
#         break

#     # Convert to grayscale for marker detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect ArUco markers
#     corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

#     if ids is not None:
#         # Estimate pose of each marker
#         rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

#         for i in range(len(ids)):
#             # Draw axes to visualize pose estimation
#             cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], MARKER_SIZE / 2)
#             print(f"Marker ID: {ids[i][0]}, Translation: {tvecs[i].flatten()}, Rotation: {rvecs[i].flatten()}")

#         # Draw detected markers on the frame
#         cv2.aruco.drawDetectedMarkers(frame, corners, ids)

#     # Display the result
#     cv2.imshow("Pose Estimation with Laptop Camera", frame)

#     # Exit loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load the camera
# cap = cv2.VideoCapture(0)

# # Load the dictionary and parameters
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# parameters = cv2.aruco.DetectorParameters()

# # Create detector
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame")
#         break

#     # Detect ArUco markers
#     corners, ids, rejected = detector.detectMarkers(frame)

#     # Draw markers if detected
#     if ids is not None:
#         cv2.aruco.drawDetectedMarkers(frame, corners, ids)

#     cv2.imshow("ArUco Marker Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()

import cv2
import numpy as np

# Load ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Marker size (in meters for pose estimation)
marker_size = 0.05  # 5 cm marker size

# Fake camera calibration (replace with real calibration if available)
camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Generate the ArUco marker
marker_id = 0  # ID of the marker
marker_size_px = 300  # Size of the marker in pixels
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

# Simulate a camera view (use the generated marker as the "camera frame")
simulated_image = marker_image.copy()  # This would be the frame from a webcam in a real scenario

# Detect the marker in the simulated image
corners, ids, rejected = cv2.aruco.detectMarkers(simulated_image, aruco_dict, parameters=parameters)

# Perform pose estimation if markers are detected
if ids is not None:
    for i in range(len(ids)):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, dist_coeffs)

        # Draw the marker's axis in the simulated image
        cv2.drawFrameAxes(simulated_image, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

    # Draw detected markers
    cv2.aruco.drawDetectedMarkers(simulated_image, corners, ids)

# Display the simulated image with pose estimation
cv2.imshow("Simulated Pose Estimation", simulated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


