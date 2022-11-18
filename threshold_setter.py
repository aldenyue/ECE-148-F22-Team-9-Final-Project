import cv2 as cv2 #importing the library
import numpy as np
import matplotlib.pyplot as plt8

def myCallBack1(val):
    global lower_thresh
    lower_thresh = val
    print('lower', lower_thresh)

def myCallBack2(val):
    global upper_thresh
    upper_thresh = val
    print('upper', upper_thresh)

def find_canny(img,thresh_low,thresh_high): #function for implementing the canny
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # show_image('gray',img_gray)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    # show_image('blur',img_blur)
    img_canny = cv2.Canny(img_blur,thresh_low,thresh_high)
    return img_canny

def region_of_interest(image): #function for extracting region of interest
    #bounds in (x,y) format
    bounds = np.array([[[0,600],[0,200],[150,200],[500,200],[650,300],[650,650]]],dtype=np.int32)
    
    # bounds = np.array([[[0,image.shape[0]],[0,image.shape[0]/2],[900,image.shape[0]/2],[900,image.shape[0]]]],dtype=np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,bounds,[255,255,255])
    # show_image('inputmask',mask)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def draw_lines(img,lines): #function for drawing lines on black mask
    mask_lines=np.zeros_like(img)
    for points in lines:
        x1,y1,x2,y2 = points[0]
        cv2.line(mask_lines,(x1,y1),(x2,y2),[0,0,255],2)

    return mask_lines

def get_coordinates(img,line_parameters): #functions for getting final coordinates
    slope=line_parameters[0]
    intercept = line_parameters[1]
    # y1 =300
    # y2 = 120
    y1=img.shape[0]
    y2 = 0.6*img.shape[0]
    x1= int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    print(x1,int(y1),x2,int(y2),'final_line')
    return [x1,int(y1),x2,int(y2)]

def compute_average_lines(img,lines):
    left_lane_lines=[]
    right_lane_lines=[]
    left_weights=[]
    right_weights=[]
    for points in lines:
        x1,y1,x2,y2 = points[0]
        if x2==x1:
            continue     
        parameters = np.polyfit((x1,x2),(y1,y2),1) #implementing polyfit to identify slope and intercept
        slope,intercept = parameters
        length = np.sqrt((y2-y1)**2+(x2-x1)**2)
        if slope <0:
            left_lane_lines.append([slope,intercept])
            left_weights.append(length)         
        else:
            right_lane_lines.append([slope,intercept])
            right_weights.append(length)
    # Computing average slope and intercept
    left_average_line = np.average(left_lane_lines,axis=0)
    right_average_line = np.average(right_lane_lines,axis=0)
    print(left_average_line,right_average_line)
    # #Computing weigthed sum
    # if len(left_weights)>0:
    #     left_average_line = np.dot(left_weights,left_lane_lines)/np.sum(left_weights)
    # if len(right_weights)>0:
    #     right_average_line = np.dot(right_weights,right_lane_lines)/np.sum(right_weights)
    left_fit_points = get_coordinates(img,left_average_line)
    right_fit_points = get_coordinates(img,right_average_line) 
    # print(left_fit_points,right_fit_points)
    return [[left_fit_points],[right_fit_points]] #returning the final coordinates



## Initialization
image = cv2.imread('track_image2.jpg') #Reading the image file
lane_image = np.copy(image)
lane_image = cv2.resize(lane_image,(650,500))

## Trackbars
lower_thresh = 100; upper_thresh = 200
cv2.namedWindow('My Trackbars')
cv2.resizeWindow('My Trackbars', 400, 100)
cv2.moveWindow('My Trackbars', 400, 0)
cv2.createTrackbar('Lower','My Trackbars',0,250, myCallBack1)
cv2.createTrackbar('Upper','My Trackbars',0,250, myCallBack2)

while True:
    lane_canny = find_canny(lane_image,lower_thresh,upper_thresh)
    lane_roi = region_of_interest(lane_canny)
    lane_lines = cv2.HoughLinesP(lane_roi,1,np.pi/180,50,40,5)
    # print(lane_lines)
    lane_lines_plotted = draw_lines(lane_image,lane_lines)
    # #show_image('lines',lane_lines_plotted)
    # result_lines = compute_average_lines(lane_image,lane_lines)
    # #print(result_lines)
    # final_lines_mask = draw_lines(lane_image,result_lines)
    # for points in result_lines:
    #     x1,y1,x2,y2 = points[0]
    #     cv2.line(lane_image,(x1,y1),(x2,y2),(0,0,255),5)

    cv2.imshow('track', lane_canny)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break

cv2.destroyAllWindows()