# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

1. Grayscale
2. Gaussian Blur
3. Canny edge detection
4. Region of interest
5. Haugh Transfer
6. Build lane Lines - draw lines
7. Merge lane lines with original image

My pipeline consisted of 7 steps. First, I converted the images to grayscale, then I 
applied a gaussian filter to the image to smooth the edges. This avoids false edges being detected.

Third, I applied the canny filter to detect high rate of changes in the image. this finds
the edges in the image

SHOW CANNY IMAGE

In order to find the lane lines only and ignore the rest of the image, a mask was created for the region
of interest. A set of vertices was created using the np.array function. This was applied to
the cv2.fillpoly function. This mask was combined with the canny image using the bitwise_and function.

SHOW REGION OF INTEREST CANNY IMAGE

Once I had the lane in focus, I applied the haugh transform to the canny image region of interest.
This created line segments along the lane lines. These line segments needed to be filtered and merged into a left and right lane road markings. 

SHOW ALL LINES FOUND FROM HAUGH TRANSFORM

I left the draw_lines() function in tact and created a new function, make_lines().
Make lines() takes all the line segments and measures the slope of each line. If the slope is less than
zero, it is counted as a left lane. the slope is accumulated in the left_slope variable and then divided by the number of lines on the left side.
This gives the average slope of all the lines on the left.
This is repeated for all lines with a slope greater than zero. this gives the average slope for the right lane.
The intercept point for each line is calculated and averaged. From the average slope and intercept, the final left and right lines are drawn.

SHOW LEFT AND RIGHT LANE LINES

Finally, the new lane lines image is merged with the original image of the road, showing the identified lane from the road markings.  

SHOW ORIGINAL ROAD IMAGE WITH LANE LINES ADDED




![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the road curves round a bend. The haugh transform only builds straight lines.
It would need to be modified to add a polynomial line model.

At the moment there is a lot of variance in the video from one line to the next.

If the road goes down a hill or up a hill, the region of interest would not be correct and the lane would not be correctly identified.
To handle this scenario, the horizon would need to be identified and the region of interest would be derived from the horizon.


### 3. Suggest possible improvements to your pipeline

THe drawn lane lines vary a lot from frame to frame in the video. These could be averaged from frame
to frame to smooth the position of the lane line and hopefully give them a more stable position.

A possible improvement would be to manage error handling. If the pipeline could not correctly identify the lanes, or drew lanes
with a widely incorrect, slope, then it should draw a predefined set of lines representing a straigh lane ahead. It should
also give an error message in the lane, saying it could not find the lane. The driver needs to know the system is not working.

If the there are no road markings, the pipeline should draw the default lane lines. Again it should warn the driver that no
lanes were found.


