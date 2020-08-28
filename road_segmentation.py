#Python script to perform road segmentation
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Utils
def display_image(winname,frame):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,frame)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()

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
         
         
        self.rgb_img = cv2.bitwise_and(self.rgb_img,self.rgb_img,mask=mask_img)
        rgb_seg = np.copy(self.grayscale_img)
        
        #Construct histogram in HSV space
        hsv_img = cv2.cvtColor(self.rgb_img,cv2.COLOR_BGR2HSV)
        h_img = hsv_img[:,:,0]
        s_img = hsv_img[:,:,1]
        v_img = hsv_img[:,:,2]
        display_image('Hue Image',h_img)
        display_image('Sat Image',s_img)
        display_image('Value Image',v_img)

        #TODO: Initial Estimate: To be improved
        road_color_range = [(50,5,0),(150,25,255)]
        road_mask = cv2.inRange(hsv_img,road_color_range[0],road_color_range[1])
        road_seg = cv2.bitwise_and(rgb_seg,rgb_seg,mask=road_mask)
        display_image("Road Segmentation",road_seg)
    
        #Thresholding
        # road_seg = cv2.GaussianBlur(road_seg,(5,5),0)
        ret,thresh_img_1 = cv2.threshold(road_seg,0,255,cv2.THRESH_BINARY)
        display_image("thresh_image",thresh_img_1)
        















if __name__ == '__main__':
    img_path = '/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/ZED_image673.jpeg'
    road_seg_utils_obj = road_seg_utils(img_path)
    road_seg_utils_obj.test()