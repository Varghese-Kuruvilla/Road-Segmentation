#Python script to perform road segmentation
import cv2
import numpy as np
import glob 
import imutils
from matplotlib import pyplot as plt
from statistics import mean 

#Measure compute time
import timeit

#Utils
def display_image(winname,frame):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,frame)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        exit(0)

def breakpoint():
    inp = input("Waiting for input...")

class road_seg_utils():
    def __init__(self,img_path=None):
        if(img_path!=None):
            self.rgb_img = cv2.imread(img_path)
            self.grayscale_img = cv2.imread(img_path,0)
        else:
            self.rgb_img = None
            self.grayscale_img = None


    def texture_segment(self):
        '''
        Function to segment the road region based on texture
        '''
        
    def colour_segment(self,frame=None):
        #TODO: Improve segmentation based on colour
        #NOTE: Because of the tree canopy along the main road the colours are not consistent
        # and exhibit a lot of variance
        if(frame != None):
            self.rgb_img = frame
            self.grayscale_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #Mask top half of the image
        mask_img = np.zeros_like(self.rgb_img[:,:,0])
        mask_img[int(self.rgb_img.shape[0]/2):int(self.rgb_img.shape[0]),:] = [255]
        self.rgb_img = cv2.bitwise_and(self.rgb_img,self.rgb_img,mask=mask_img)
        # display_image('mask_image',self.rgb_img)

        rgb_color_seg = np.copy(self.rgb_img)
        line_img = np.copy(self.rgb_img)

        hsv_img = cv2.cvtColor(self.rgb_img,cv2.COLOR_BGR2HSV)
        yuv_img = cv2.cvtColor(self.rgb_img,cv2.COLOR_BGR2YUV)
        s_img = hsv_img[:,:,1]
        # v_img = hsv_img[:,:,2]

        # display_image("Hue Image",h_img)
        # display_image("Sat Image",s_img)
        # display_image("Value Image",v_img)

        road_color_range_hsv = [(0,0,40),(180,50,255)]
        # road_color_range_yuv = [(90,0,0),(140,255,255)]

        road_mask_hsv = cv2.inRange(hsv_img,road_color_range_hsv[0],road_color_range_hsv[1])
        # display_image("road_mask_hsv",road_mask_hsv)
        # road_mask_yuv = cv2.inRange(yuv_img,road_color_range_yuv[0],road_color_range_yuv[1])
        # display_image("road_mask_yuv",road_mask_yuv)
        road_mask = road_mask_hsv 
        # display_image('road_mask:',road_mask)
        # rgb_color_seg = cv2.bitwise_and(rgb_color_seg,rgb_color_seg,mask=road_mask)
        # display_image("rgb_color_seg",rgb_color_seg)


        edge_img = np.copy(road_mask)
        edge_img = cv2.bilateralFilter(edge_img,15,75,75)

        #Canny edge detection thresholds
        sigma = 0.33
        v = np.median(road_mask)
        lower = int(max(0,(1.0-sigma)*v))
        upper = int(min(255,(1.0+sigma)*v))
        edge_img = cv2.Canny(edge_img,lower,upper)
        display_image('Edge Image',edge_img)

        #Hough Lines
        votes = 0.25 * self.grayscale_img.shape[0]
        lines = cv2.HoughLines(self.grayscale_img,1,1*(np.pi/180),int(votes/2))
        left_lane_line = []
        right_lane_line = []
        #Draw lines on the image
        if(lines.any() != None):
            for line in lines:
                rho,theta = line[0]
                # print('rho,theta:',rho,theta)
                if((abs(theta) >= 0.349) and (abs(theta) <= 1.22)):
                    #Left Lane: Find the average slope of all these lines
                    left_lane_line.append([rho,theta])
                elif((abs(theta) >= 0.349 and abs(theta) <= 1.22) or (abs(theta) >= 1.92 and abs(theta) <= 2.79)):
                    right_lane_line.append([rho,theta])
                else:
                    continue
                # a = np.cos(theta)
                # b = np.sin(theta)
                # x0 = a*rho
                # y0 = b*rho
                # x1 = int(x0 + 1500*(-b))
                # y1 = int(y0 + 1500*(a))
                # x2 = int(x0 - 1500*(-b))
                # y2 = int(y0 - 1500*(a))
                # cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)                
                # display_image("line_img",line_img)

        np_left_lane_line = np.asarray(left_lane_line)
        np_right_lane_line = np.asarray(right_lane_line)

        for i in range(0,2):
            if(i==0 and np.size(np_left_lane_line) != 0):
                rho,theta = np.mean(np_left_lane_line,axis=0)
            elif(i==1 and np.size(np_right_lane_line) != 0):
                rho,theta = np.mean(np_right_lane_line,axis=0)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1500*(-b))
            y1 = int(y0 + 1500*(a))
            x2 = int(x0 - 1500*(-b))
            y2 = int(y0 - 1500*(a))
            cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)                
        display_image("Line Image",line_img)

        
        #Gabor Filter
        # ksize = 31
        # for angle in np.arange(0,np.pi,np.pi/16):
        #     start = timeit.default_timer()
        #     kern = cv2.getGaborKernel((ksize,ksize),sigma = 4.0, theta = angle , lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        #     fimg = cv2.filter2D(self.rgb_img,cv2.CV_8UC3,kern)
        #     stop = timeit.default_timer()
        #     print('Time:',stop-start)
        #     breakpoint()
        #     display_image("fimg",fimg)



    def test_1(self,frame=None):

        if(np.size(frame) != 0):
            self.rgb_img = frame
            self.grayscale_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # display_image('grayscale_img',self.grayscale_img)

        mask_img = np.zeros_like(self.rgb_img[:,:,0])
        mask_img[int(self.rgb_img.shape[0]/2):int(self.rgb_img.shape[0]),:] = [255]
        edge_img = np.copy(self.grayscale_img)

        self.rgb_img = cv2.bitwise_and(self.rgb_img,self.rgb_img,mask=mask_img)
        line_img = np.copy(self.rgb_img)
        self.grayscale_img = cv2.bitwise_and(self.grayscale_img,self.grayscale_img,mask=mask_img)
        display_image("grayscale Image",self.grayscale_img)
        #Edge detection
        self.grayscale_img = cv2.bilateralFilter(self.grayscale_img,15,75,75)
        display_image("Blurred Image",self.grayscale_img)
        #Canny edge detection thresholds
        sigma = 0.33
        v = np.median(edge_img)
        lower = int(max(0,(1.0-sigma)*v))
        upper = int(min(255,(1+sigma)*v))
        self.grayscale_img = cv2.Canny(self.grayscale_img,lower,upper)
        display_image('Edge Image',self.grayscale_img)

        #Hough Lines
        #Min number of votes required
        votes = 0.25 * self.grayscale_img.shape[0]
        lines = cv2.HoughLines(self.grayscale_img,1,1*(np.pi/180),int(votes/2))
        left_lane_line = []
        right_lane_line = []
        #Draw lines on the image
        if(lines.any() != None):
            for line in lines:
                rho,theta = line[0]
                print('rho,theta:',rho,theta)
                if((abs(theta) >= 0.349) and (abs(theta) <= 1.22)):
                    #Left Lane: Find the average slope of all these lines
                    left_lane_line.append([rho,theta])
                elif((abs(theta) >= 0.349 and abs(theta) <= 1.22) or (abs(theta) >= 1.92 and abs(theta) <= 2.79)):
                    right_lane_line.append([rho,theta])
                # if((abs(theta) >= 0.349 and abs(theta) <= 1.22) or (abs(theta) >= 1.92 and abs(theta) <= 2.79)):
                    # a = np.cos(theta)
                    # b = np.sin(theta)
                    # x0 = a*rho
                    # y0 = b*rho
                    # x1 = int(x0 + 1500*(-b))
                    # y1 = int(y0 + 1500*(a))
                    # x2 = int(x0 - 1500*(-b))
                    # y2 = int(y0 - 1500*(a))
            
            #Draw left and right lines on the image
            np_left_lane_line = np.asarray(left_lane_line)
            np_right_lane_line = np.asarray(right_lane_line)

            for i in range(0,2):
                if(i==0 and np.size(np_left_lane_line) != 0):
                    rho,theta = np.mean(np_left_lane_line,axis=0)
                elif(i==1 and np.size(np_right_lane_line) != 0):
                    rho,theta = np.mean(np_right_lane_line,axis=0)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1500*(-b))
                y1 = int(y0 + 1500*(a))
                x2 = int(x0 - 1500*(-b))
                y2 = int(y0 - 1500*(a))
                cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)                
            display_image("Line Image",line_img)

if __name__ == '__main__':
    for img_path in glob.glob('/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/*.jpeg'):
        # img_path = '/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/ZED_image825.jpeg'
        print('img_path:',img_path)
        road_seg_utils_obj = road_seg_utils(img_path)
        road_seg_utils_obj.colour_segment()

    #Testing
    # road_seg_utils_obj = road_seg_utils()
    # cap = cv2.VideoCapture('/home/varghese/Transvahan/demo/autonomous_parking_vision/data_ece_23rd_August/Varghese/second/out.avi')
    # while(True):
    #     ret,frame = cap.read()
    #     road_seg_utils_obj.test_1(frame)
    
    # cap.release()
    # cv2.destroyAllWindows()

