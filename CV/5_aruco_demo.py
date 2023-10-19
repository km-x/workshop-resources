import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

camera_matrix = np.load('mtx.npy')
dist_coeffs = np.load('dist.npy')
print(camera_matrix)
print(dist_coeffs)

while True:
    ret, img = cap.read()

    if not ret:
        break

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100) # change to the CV2 definnitions such as arucoget predeifnided dictionary (cv.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv.aruco.DetectorParameters()
    (corners_list, ids, rejected) = cv.aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)

    if corners_list:
        rvecs = []
        tvecs = []
        for corners, id in zip(corners_list, ids):
            pts = np.array(corners,dtype=np.int32)
            cv.polylines(img, pts, True, (0, 0, 255), 10)
            markerLength = 0.010 # 28 mm
            rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs)
            rvecs.append(rvec)
            tvecs.append(tvec)
        if len(ids) == 2:
            p1 = tvecs[0][0][0]
            p2 = tvecs[1][0][0]
            dist = np.linalg.norm(p1 - p2)
            x1, y1 = np.mean(corners_list[0][0], axis=0).astype(np.int64)
            x2, y2 = np.mean(corners_list[1][0], axis=0).astype(np.int64)
            x3, y3 = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(img, '{:.3f}'.format(dist), (x3, y3), cv.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 4)
            



    cv.imshow('frame', img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
