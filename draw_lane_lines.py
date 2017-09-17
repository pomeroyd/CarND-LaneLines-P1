
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def build_my_lane_lines(lines):


	slope_left =range(0)
	b_left = range(0)
	slope_right = range(0)
	b_right = range(0)
	count_left = 0
	count_right =0
	x1_left = range(0)
	x2_left = range(0)
	x1_right = range(0)
	x2_right = range(0)
	number_of_lines = len(lines)
	line_no = 0
	slope = range(0)


	for line in lines:
	
		#print(line)
		for x1,y1,x2,y2 in line:
		
			#print("x1 = ",x1)
			#print("y1 = ",y1)
			#print("x2 = ",x2)
			#print("y2 = ",y2)


			slope.append(float(y1-y2)/(x2-x1))
			#print("slope = ",slope[line_no])
		
			b1 = y1 - slope[line_no]*x1
			b2 = y2 - slope[line_no]*x2

			#print("b1 = ",b1)
			#print("y1 = ",round(y1),"slope[line_no]*x1+b1 = ",round(slope[line_no]*x1+b1))
			assert(round(y1) == round(slope[line_no]*x1+b1))
		
			if(slope[line_no] >0):
				#print("slope is left lane")
				slope_left.append(slope[line_no])
				b_left.append(b1)
				count_left = count_left+1
				x1_left.append(x1)
				x2_left.append(x2)
			else:
				#print("slop is right lane")
				slope_right.append(slope[line_no])
				b_right.append(b1)
				count_right = count_right +1
				x1_right.append(x1)
				x2_right.append(x2)
		line_no = line_no +1
		
	if(count_left != 0):
		slope_left_ave = sum(slope_left)/len(slope_left)
		b_left_ave = sum(b_left)/len(b_left)
		x1_left_ave = sum(x1_left)/len(x1_left)
		x2_left_ave = sum(x2_left)/len(x2_left)
	else:
		slope_left_ave = 0
		b_left_ave = 0
	print("slope_left_ave = ",slope_left_ave)
	print("b_left_ave = ",b_left_ave)

	if(count_right !=0):
		slope_right_ave = sum(slope_right)/len(slope_right)
		b_right_ave = sum(b_right)/len(b_right)
		x1_right_ave = sum(x1_right)/len(x1_right)
		x2_right_ave = sum(x2_right)/len(x1_right)
	else:
		slope_right_ave = 0
		b_right_ave = 0
	print("slope_right_ave = ",slope_right_ave)
	print("b_right_ave = ",b_right_ave)

	#ylength should equal image.shape[0] Y length
	# Need to handle topleft corner start point when building lines.
	ylength = 500
	y1_left = ylength
	x1_left = round((y1_left-b_left_ave)/slope_left_ave)
	y2_left = ylength*0.5
	x2_left = round((y2_left+ylength*0.7-b_left_ave)/slope_left_ave)
	print("left lane made up of (x1, y1), (x2,y2)",x1_left,y1_left,x2_left,y2_left)
	
	y1_right = ylength
	x1_right = round((y1_right-b_right_ave)/slope_right_ave)
	y2_right = ylength*0.5
	x2_right = round((y2_right+ylength*0.0-b_right_ave)/slope_right_ave)
	print("right lane made up of (x1, y1), (x2,y2)",x1_right,y1_right,x2_right,y2_right)

	print("average x1,x2,right x1,x2 = ",x1_left_ave,x2_left_ave,x1_right_ave,x2_right_ave)
	lane_lines = np.zeros(shape = (1,2,4),dtype = np.int) 
	lane_lines[0][0] = [x1_left,y1_left,x2_left,y2_left]
	lane_lines[0][1] = [x1_right,y1_right,x2_right,y2_right]

	#retrun my_lane_lines
	return lane_lines




lines = [[[176, 539, 478, 311]],[[313, 428, 476, 310]], [[229, 486, 471, 311]], [[249, 483, 318, 423]]]
lines = [[[176,539,478,311]],[[313,428,476,310]],[[229,486,471,311]],[[473,310,794,495]],[[154,538,491,310]],[[474,310,647,418]],[[175,538,315,425]],[[307,441,481,310]],[[192,526,280,450]],[[492,314,654,419]],[[495,310,782,496]],[[356,398,457,325]],[[237,492,291,442]],[[477,310,772,494]],[[217,507,341,407]],[[155,539,238,478]],[[453,319,519,318]],[[481,311,609,391]],[[204,517,287,445]],[[435,338,494,310]],[[501,317,616,400]],[[497,312,559,362]],[[249,483,318,423]]]

my_lane_lines = build_my_lane_lines(lines)

print("my lane lines = ",my_lane_lines)
print(" ")
#print("x1_left",x1_left

#Dont use
def make_lines(lines, ylength):
    
    
    """
    step 1: identify left and right lane segments using slope info
    
    step 2: find mean value of slope and intercept for left and right lanes
    
    step 3: Given slope and intercept, find the x cordinates of lane lines  after fixing 
    y cordinates at certain heights (y=mx+c)
    
    This function returns left and right lanes given hough lines identified inside ROI
    
    """

    left_slope = 0
    left_intercept = 0
    cnt_left_lines =0
   
    right_slope = 0
    right_intercept = 0
    cnt_right_lines =0

    #xl1 = xl2 = xr1 = xr2 = 0 # xl1-->left lane segment: first x coordinate  
 
    #fix y cordinates of the lane segments
    yl1 = yr1 = ylength
    yl2 = yr2 = 0.575*ylength
    
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

