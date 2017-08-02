"""
camera calibration methods
compute camera matrix method should be called under the root directory
"""

import cv2
import os
import numpy as np
import glob

def find_corner(file_path, nx=9, ny=6, visualize = False):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    return ret, corners

def generate_calibration_data(directory_path, nx=9, ny=6):
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    paths = os.path.join(directory_path, 'calibration*.jpg')
    images = glob.glob(paths)
    del images[0]
    objpoints = list()
    imgpoints = list()
    for idx, fname in enumerate(images):
        ret, corners = find_corner(fname, visualize=False)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)         
    return objpoints, imgpoints

def compute_camera_matrix(root_directory=os.getcwd()):
    objpoints, imgpoints = generate_calibration_data(os.path.join(root_directory, 'camera_cal'))
    img = cv2.imread(os.path.join(root_directory, 'camera_cal', 'calibration1.jpg'))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)
    return mtx, dist

if __name__ == '__main__':
    mtx, dist = compute_camera_matrix()
    print('camera matrix: ', mtx)
    print('distortion coefficient: ', dist)
