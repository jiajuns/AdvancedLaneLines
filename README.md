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

The code for this step is contained in the first code cell of the IPython notebook located in "advanced_lane_detection.ipynb" (or in lines # through # of the file called `code/camera_cal.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  


Image undistortion
---

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

Below is an example of undistorted image for the road.

![alt text][image2]


Create a Thresholded Binary Image
---

I used a combination of color and gradient thresholds and angle to generate a binary image (thresholding steps at lines 30 through 65 in `/code/image_pipeline.py`).  Here's an example of my output for this step.

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
