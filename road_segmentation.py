#Python script to perform road segmentation
import cv2
import numpy as np
import glob 
import imutils
from matplotlib import pyplot as plt

#Utils
def display_image(winname,frame):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,frame)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()

def breakpoint():
    inp = input("Waiting for input...")

class road_seg_utils():
    def __init__(self,img_path):
        self.rgb_img = cv2.imread(img_path)
        self.grayscale_img = cv2.imread(img_path,0)


    def test(self):
        #Mask top half of the image
        mask_img = np.zeros_like(self.rgb_img[:,:,0])
        mask_img[int(self.rgb_img.shape[0]/2):int(self.rgb_img.shape[0]),:] = [255]
        # mask_img = np.zeros((int(self.rgb_img.shape[0]/2),self.rgb_img.shape[1]),dtype = np.uint8)
        # display_image("mask_img",mask_img)
         
        edge_img = np.copy(self.rgb_img)
        self.rgb_img = cv2.bitwise_and(self.rgb_img,self.rgb_img,mask=mask_img)
        cnt_img = np.copy(self.rgb_img)
        extreme_pt_img = np.copy(self.rgb_img)
        rgb_seg = np.copy(self.grayscale_img)
        
        #ColourSpaces
        #NOTE
        # HSV- Might be useful
        # LAB- Cannot be used
        # YUV- Cannot be used
        hsv_img = cv2.cvtColor(self.rgb_img,cv2.COLOR_BGR2HSV)
        h_img = hsv_img[:,:,0]
        s_img = hsv_img[:,:,1]
        # v_img = hsv_img[:,:,2]
        # display_image('Hue Image',h_img)
        # display_image('Sat Image',s_img)
        # display_image('Value Image',v_img)
       

        #TODO: Initial Estimate: To be improved
        road_color_range = [(50,5,0),(150,20,255)]
        road_mask = cv2.inRange(hsv_img,road_color_range[0],road_color_range[1])
        road_seg = cv2.bitwise_and(rgb_seg,rgb_seg,mask=road_mask)
        # display_image("Road Segmentation",road_seg)
    
        #Thresholding
        # road_seg = cv2.GaussianBlur(road_seg,(5,5),0)
        ret,thresh_img_1 = cv2.threshold(road_seg,0,255,cv2.THRESH_BINARY)

        #Erosion
        kernel = np.ones((5,5),np.uint8)
        thresh_img_1 = cv2.morphologyEx(thresh_img_1, cv2.MORPH_OPEN, kernel)
        # display_image("thresh_image",thresh_img_1)


        #Find largest contour
        cnts = cv2.findContours(thresh_img_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if(len(cnts) != 0):
            c = max(cnts,key=cv2.contourArea)
            cnt_img = cv2.drawContours(cnt_img,[c],0,(0,0,255),5)
            display_image("Contour Image",cnt_img)

            #Find extreme points along the contour
            hull = cv2.convexHull(c,clockwise=True,returnPoints=True)
            cnt_img = cv2.drawContours(cnt_img,[hull],0,(0,255,0),5)
            display_image("Contour Image",cnt_img)

            cv2.fillPoly(self.rgb_img,pts = [hull], color=(0,0,0))
            display_image("Masked Image",self.rgb_img)

            #Edge detection
            self.rgb_img = cv2.bilateralFilter(self.rgb_img,15,75,75)
            display_image("Blurred Image",self.rgb_img)

            #Canny edge detection thresholds
            sigma = 0.20
            v = np.median(edge_img)
            lower = int(max(0,(1.0-sigma)*v))
            upper = int(min(255,(1+sigma)*v))
            self.rgb_img = cv2.Canny(self.rgb_img,lower,upper)
            display_image('Edge Image',self.rgb_img)
        
            

        















if __name__ == '__main__':
    for img_path in glob.glob('/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/*.jpeg'):
    # img_path = '/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/ZED_image673.jpeg'
        print('img_path:',img_path)
        road_seg_utils_obj = road_seg_utils(img_path)
        road_seg_utils_obj.test()