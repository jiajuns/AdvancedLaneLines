## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project is one step furthur to the [Lane Detection Basic](https://jiajuns.github.io/LaneDetectionBasic). This project dose not require identify lane but also produce a measurement of lane`s curvature and vehicle position relative to the lane. Therefore, it requires some knowledge of camera calibration and image rectification. Here's a [link to my video result](./output_video.mp4).

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_test.png "Road Transformed"
[image3]: ./examples/binary_example.png "Binary Example"
[image4]: ./examples/recitfied_result.png "Warp Example"
[image5]: ./examples/fit_line.png "Fit Visual"
[image6]: ./examples/output_image.png "Output"
[video1]: ./project_video.mp4 "Video"

steps of this project are :
---
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames. `output_images` contain output from each stage of the pipeline. `project_video.mp4` is the input video for this project. The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions. The `harder_challenge.mp4` video is another optional challenge and is brutal!


Camera Calibration
---

Camera calibration is to solve camera intrinsics and distortion parameters. Most important intrinsics include `focal length` and `principal points`. Here, I take the advantage of OpenCV `cv2.calibrateCamera()` function, which requires two parameters: (1) `objpoints` and (2) `imgpoints`. The code for this step is contained in the first code cell of the IPython notebook located in `advanced_lane_detection.ipynb`

I start by preparing `object points`, which will be the (x, y, z) coordinates of the chessboard corners in the real world. I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.

`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the focal plane. OpenCV has predefined a helper function that assists identify chessboard corner location from grayscale image:

```python
img = cv2.imread(file_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
```

After preparing `objpoints` and `imgpoints`, you can directly use `cv2.calibrateCamera()`:

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)
```


Image undistortion
---

Given the distortion coefficient from camera calibration, I can apply distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

```python
undist = cv2.undistort(img, mtx, dist, None, mtx)
```

![alt text][image1]

Below is an example of undistorted image for the road.

![alt text][image2]


Create a Thresholded Binary Image
---

I used a combination of color and gradient thresholds and angle to generate a binary image (thresholding steps at lines 30 through 65 in `/code/image_pipeline.py`).
One safe assumption for lane lines are the angle of lines are oritentated around 90 degrees. In order to extract angle information, I apply sobel filter on x and y direction:

```python
sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
sobely = cv2.Sobel(r_channel, cv2.CV_64F, 0, 1)

# Threshold magnitude
magnitude = np.sqrt(sobelx**2 + sobely**2)
magnitude = np.uint8(255*magnitude/np.max(magnitude))
m_binary = np.zeros_like(magnitude)
m_binary[(magnitude >= m_thresh[0]) & (magnitude <= m_thresh[1])] = 1

# Threshold direction
angle = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
anglebinary = np.zeros_like(angle)
anglebinary[(angle >= angle_thresh[0]) & (angle <= angle_thresh[1])] = 1
```

![alt text][image3]

Rectify Image
---
The code for my perspective transform includes a function called `rectify()`, which appears in lines 11 through 28 in the file `/code/image_pipeline.py`.  The `rectify()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32(
        [[288, 660],
         [1015, 660],
         [703, 460],
         [578, 460]])

    dst = np.float32(
        [[400, 700],
         [900, 700],
         [900, 0],
         [400, 0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Detect lane pixels
---
I use sliding window to identify points on the rectified images and then fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

Determine the curvature
---
I use below equation to calculated the curvature:

![equation](http://www.sciweavers.org/tex2img.php?eq=%5B1%2B%282Ay%2BB%29%5E2%5D%5E%7B3%2F2%7D%2F%7C2A%7C&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)

The position of the vehical is defined as the distance from the center pixel to the average of left lane and right lane. I did this in lines 156 through 175 in my code in `/code/image_pipeline.py`

Plot boundary back onto original images
---
I implemented this step in lines 178 through 195 in my code in `/code/image_pipeline.py` in the function `wrap_back()`.  Here is an example of my result on a test image:

![alt text][image6]
