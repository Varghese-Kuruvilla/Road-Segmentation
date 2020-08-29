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


    def test_1(self):
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
        #Draw lines on the image
        for line in lines:
            rho,theta = line[0]
            if((abs(theta) >= 0.349 and abs(theta) <= 1.22) or (abs(theta) >= 1.92 and abs(theta) <= 2.79)):
                print('rho,theta:',rho,theta)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)
                display_image("line Image",line_img)
                


if __name__ == '__main__':
    for img_path in glob.glob('/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/*.jpeg'):
        # img_path = '/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/ZED_image159.jpeg'
        print('img_path:',img_path)
        road_seg_utils_obj = road_seg_utils(img_path)
        road_seg_utils_obj.test_1()