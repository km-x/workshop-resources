import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((8 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

gray = None
while True:
    ret, img = cap.read()
    
    if not ret:
        break
    cv.imshow('img', img)
    cv.waitKey(1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (8, 5), None)

    if ret:
        c = cv.waitKey(1)
        if c == ord(' '):
            objpoints.append(objp)
            imgpoints.append(corners)

        cv.drawChessboardCorners(img, (8, 5), corners_draw, ret)
        cv.imshow('corners', img)
        cv.waitKey(1)

    c = cv.waitKey(1)
    if c == ord('q'):
        break
if gray is not None:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.save("ret.npy", ret)
np.save("mtx.npy", mtx)
np.save("dist.npy", dist)
np.save("rvecs.npy", rvecs)
np.save("tvecs.npy", tvecs)

while True:
    ret, img = cap.read()
    
    if not ret:
        break

    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x, y, w, h = roi
    img_undistorted = cv.undistort(img, mtx, dist, None, newcameramtx)[y:y+h, x:x+w]
    cv.imshow('img', img)
    cv.imshow('img_undistorted', img_undistorted)
    c = cv.waitKey(1)

    if c == ord('q'):
        break


cv.destroyAllWindows()


