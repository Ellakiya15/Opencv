import numpy as np
import cv2

ARUCO_DICT= {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5" : cv2.aruco.DICT_APRILTAG_16h5,
}

aruco_type = "DICT_APRILTAG_16h5"
id = 1

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

print("Aruco type '{}' with ID '{}'".format(aruco_type,id))
tag_size = 500

tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
cv2.aruco.drawMarker(arucoDict,id,tag_size,tag, 1)

tag_name = "arucoMarkers/" + aruco_type + "_" + str(id) + ".png"
cv2.imwrite(tag_name,tag)
cv2.imshow("Aruco Tag", tag)

cv2.waitKey(0)

cv2.destroyAllWindows()
