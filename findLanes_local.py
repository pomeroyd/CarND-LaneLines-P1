#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline



import os
os.listdir("test_images/")

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

plt.show()

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    return img

def draw_lines_other(img, lines, color=[255, 0, 0], thickness=10):
    imshape = img.shape
    left_x1 = []
    left_x2 = []
    right_x1 = []
    right_ximg2 = [] 
    y_min = img.shape[0]
    y_max = int(img.shape[0]*0.611)
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1)) < 0:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                left_x1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                left_x2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
#           cv2.line(img, (xone, imshape[0]), (xtwo, 330), color, thickness)
            elif ((y2-y1)/(x2-x1)) > 0:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                right_x1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                right_x2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
#           cv2.line(img, (xone, imshape[0]), (xtwo, 330), color, thickness)
    l_avg_x1 = np.int(np.nanmean(left_x1))
    l_avg_x2 = np.int(np.nanmean(left_x2))
    r_avg_x1 = np.int(np.nanmean(right_x1))
    r_avg_x2 = np.int(np.nanmean(right_x2))
#     print([l_avg_x1, l_avg_x2, r_avg_x1, r_avg_x2])
    cv2.line(img, (l_avg_x1, y_min), (l_avg_x2, y_max), color, thickness)
    cv2.line(img, (r_avg_x1, y_min), (r_avg_x2, y_max), color, thickness) 


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #use draw_lines function
    draw_lines_other(line_img, lines)
    
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def consolidated_lines(lines, ylen):
    
    
    #"""
    #step 1: identify left and right lane segments using slope info
    
    #step 2: find mean value of slope and intercept for left and right lanes
    
    #step 3: Given slope and intercept, find the x cordinates of lane lines  after fixing 
    #y cordinates at certain heights (y=mx+c)
    
    #This function returns left and right lanes given hough lines identified inside ROI
    
    #"""

    left_slope = 0
    left_intercept = 0
    cnt_left_lines =0
   
    right_slope = 0
    right_intercept = 0
    cnt_right_lines =0

    #xl1 = xl2 = xr1 = xr2 = 0 # xl1-->left lane segment: first x coordinate  
 
    #fix y cordinates of the lane segments
    yl1 = yr1 = ylen
    yl2 = yr2 = 0.575*ylen
    
    xl1_prev = xl2_prev = xr1_prev =xr2_prev = 0

    for line in lines:
        
        for x1,y1,x2,y2 in line:
            
            slope = (y1-y2)/(x1-x2)
            intercept = y1-slope*x1
            
            # identify left lane segments based on slope
            if slope < -0.1 :
                 
                left_slope += slope
                left_intercept += intercept
                cnt_left_lines += 1

            if slope > 0.1:
                
                right_slope += slope
                right_intercept += intercept
                cnt_right_lines += 1
                


   
    if(cnt_left_lines !=0):
        
        #take avearge of the slopes and intercepts
        left_slope = left_slope/cnt_left_lines
        left_intercept = left_intercept/cnt_left_lines
        xl1 = (yl1-left_intercept)/left_slope
        xl2 = (yl2 - left_intercept)/left_slope
        
        xl1_prev = xl1
        xl2_prev = xl2
        
    else:
        
        print("div by 0...",xl1_prev) 
        xl1 = xl1_prev
        xl2 = xl2_prev
            
       

    if(cnt_right_lines !=0):
        right_slope = right_slope/cnt_right_lines
        right_intercept = right_intercept/cnt_right_lines
        xr1 = (yr1-right_intercept)/right_slope
        xr2 = (yr2 -right_intercept)/right_slope
        
        xr1_prev = xr1
        xr2_prev = xr2
        
    else:
        xr1 = xr1_prev
        xr2 = xr2_prev
        
    cons_lines = np.zeros(shape = (1,2,4),dtype = np.int) 
    cons_lines[0][0] = [xl1,yl1,xl2,yl2]
    cons_lines[0][1] = [xr1,yr1,xr2,yr2]
    return cons_lines

    #return ([[[xl1,yl1,xl2,yl2],[xr1,yr1,xr2,yr2]]])

#take original image and make grayscale

image_gray = grayscale(image)
plt.imshow(image_gray)
plt.show()

#take grayscale image and apply Canny Filter 
#with low threshold =50, high threshold = 150

low_threshold = 50
high_threshold = 150
image_gray_canny =canny(image_gray, low_threshold, high_threshold)
plt.imshow(image_gray_canny)
plt.show()

#applyt Gaussian blur with a Kernal = 5

kernel = 5
image_gray_canny_gauss = gaussian_blur(image_gray_canny, kernel)
plt.imshow(image_gray_canny_gauss)
plt.show()

#Apply region of interest
ysize = image_gray_canny_gauss.shape[0]
xsize = image_gray_canny_gauss.shape[1]
vertices = np.array([[(0,ysize),(450, 290), (490, 290), (xsize,ysize)]], dtype=np.int32)
image_gray_canny_gauss_region = region_of_interest(image_gray_canny_gauss, vertices)
plt.imshow(image_gray_canny_gauss_region)
plt.show()

#Apply Haugh Filter
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20
image_gray_canny_gauss_region_haugh,my_lines = hough_lines(image_gray_canny_gauss_region, rho, theta, threshold, min_line_length, max_line_gap)
###temp work
#lines = cv2.HoughLinesP(image_gray_canny_gauss_region_haugh, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

####
plt.imshow(image_gray_canny_gauss_region_haugh)
plt.show()

#Add drawn lines to original image

image_orig_with_lines = weighted_img(image_gray_canny_gauss_region_haugh, image, α=0.8, β=1., λ=0.)
plt.imshow(image_orig_with_lines)
plt.show()

#Draw lines

#image_orig_with_lines_draw_lines = draw_lines(image_orig_with_lines, lines, color=[255, 200, 0], thickness=2)
#plt.imshow(image_orig_with_lines_draw_lines)
#plt.show()
###############################################


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    return result

#####
#Apply to video
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time white_clip.write_videofile(white_output, audio=False)



