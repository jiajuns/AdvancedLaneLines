### Writeup / README

**Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_test.png "Road Transformed"
[image3]: ./examples/binary_example.png "Binary Example"
[image4]: ./examples/recitfied_result.png "Warp Example"
[image5]: ./examples/fit_line.png "Fit Visual"
[image6]: ./examples/output_image.png "Output"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "advanced_lane_detection.ipynb" (or in lines # through # of the file called `code/camera_cal.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds and angle to generate a binary image (thresholding steps at lines 30 through 65 in `/code/image_pipeline.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

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

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use sliding window to identify points on the rectified images and then fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I use below equation to calculated the curvature:
$$
[1 + (2Ay + B)^2]^{3/2}/|2A|
$$

The position of the vehical is defined as the distance from the center pixel to the average of left lane and right lane. I did this in lines 156 through 175 in my code in `/code/image_pipeline.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 178 through 195 in my code in `/code/image_pipeline.py` in the function `wrap_back()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is likely to fail when the image thresholding is not done properly. The image thresholding part should be robust to different lighting and road condition in order to make the pipeline working properly.
