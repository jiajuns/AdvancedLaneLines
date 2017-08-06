import cv2
import numpy as np
import os

from code.camera_cal import compute_camera_matrix

def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def rectify(img):
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

    image_shape = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    rectified_img = cv2.warpPerspective(img, M, image_shape, flags=cv2.INTER_LINEAR)
    return rectified_img, Minv

def thresholding(img, r_thresh=(190, 255), m_thresh=(5, 255), sx_thresh=(4, 255), angle_thresh=(-0.5, 0.5)):
    img = np.copy(img)
    r_channel = img[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobely = cv2.Sobel(r_channel, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255*magnitude/np.max(magnitude))
    m_binary = np.zeros_like(magnitude)
    m_binary[(magnitude >= m_thresh[0]) & (magnitude <= m_thresh[1])] = 1
    
    # Threshold direction
    angle = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    anglebinary = np.zeros_like(angle)
    anglebinary[(angle >= angle_thresh[0]) & (angle <= angle_thresh[1])] = 1
        
    # Threshold color channel
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    
    combined = np.zeros_like(magnitude)
    combined[(anglebinary == 1) & ((m_binary == 1) & (sxbinary == 1)) & (r_binary == 1)] = 1
    
    return combined

def sliding_window_detector(binary_rectified, left_fit=None, right_fit=None):
    histogram = np.sum(binary_rectified[int(binary_rectified.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_rectified, binary_rectified, binary_rectified))*255

    # Choose the number of sliding windows
    nwindows = 9
    window_height = np.int(binary_rectified.shape[0]/nwindows)
    nonzero = binary_rectified.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    if (left_fit is not None) and (right_fit is not None):
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_rectified.shape[0]-1, binary_rectified.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    else:
        # Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_rectified.shape[0] - (window+1)*window_height
            win_y_high = binary_rectified.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_rectified.shape[0]-1, binary_rectified.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    return left_fit, right_fit, left_fitx, right_fitx

def calculate_curvature(left_fit, right_fit, rectified_binary):
    ploty =  np.linspace(0, rectified_binary.shape[0]-1, rectified_binary.shape[0])
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720       # meters per pixel in y dimension
    xm_per_pix = 3.7/500    # meters per pixel in x dimension
    
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    distance_to_center = (-rectified_binary.shape[1]/2+(leftx[-1] + rightx[-1])/2) * xm_per_pix

    return left_curverad, right_curverad, distance_to_center


def warp_back(undist, rectified_binary, left_fitx, right_fitx, Minv):
    warp_zero = np.zeros_like(rectified_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, rectified_binary.shape[0]-1, rectified_binary.shape[0])
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0) 
    return result

def pipeline(img, mtx, dist, previous_left_fit=None, previous_right_fit=None):
    undistorted_img = cal_undistort(img, mtx, dist)  
    rectified_img, Minv = rectify(undistorted_img)
    rectified_binary = thresholding(rectified_img)
    left_fit, right_fit, left_fitx, right_fitx = sliding_window_detector(rectified_binary, previous_left_fit, previous_right_fit)
    left_curverad, right_curverad, distance_to_center = calculate_curvature(left_fit, right_fit, rectified_binary)
    result = warp_back(undistorted_img, rectified_binary, left_fitx, right_fitx, Minv)
    return result, left_fit, right_fit, left_curverad, right_curverad, distance_to_center

if __name__ == '__main__':
    mtx, dist = compute_camera_matrix()
    directory_path = os.path.join(os.getcwd(), 'test_images')
    img_path = os.path.join(directory_path, 'test4.jpg')
    image = cv2.imread(img_path)

    final_img, _, _, left_curverad, right_curverad = pipeline(image, mtx, dist)
    print('camera matrix: ', mtx)
    print('distortion coefficient: ', dist)
    print('left curverad {}m; right curverad {}m'.format(left_curverad, right_curverad))