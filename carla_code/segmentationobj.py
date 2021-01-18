import glob
import numpy as np
import cv2 
import os

import time
from statistics import mean
import pickle


import torch
import torchvision.transforms as transforms
import cv2 

from PIL import Image
import numpy as np 
import glob
import sys

from auto_park_utils import auto_park_vision
#Utils
def display_image(winname,img):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,img)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        sys.exit(0)

def breakpoint():
    inp = input("Waiting for input...")

#For testing
intrinsic_mat = np.array([[1.13119617e+03, 0.00000000e+00, 6.40000000e+02],
                        [0.00000000e+00, 1.13119617e+03, 3.60000000e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

cam_to_car = np.array([[-6.12323400e-17 ,-6.12323400e-17,  1.00000000e+00,  3.00000000e-01],
                       [-1.00000000e+00 , 3.74939946e-33, -6.12323400e-17,  0.00000000e+00],
                       [ 0.00000000e+00 , 1.00000000e+00,  6.12323400e-17,  1.30000000e+00],
                       [ 0.00000000e+00 , 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

car_to_world = np.array([[ 2.38548938e-06,  1.05023635e-04, -9.99999994e-01,  2.70740145e+02],
                        [ 9.99999994e-01 , 1.08124080e-04 , 2.39684497e-06 , 1.29490238e+02],
                        [-1.08124331e-04 , 9.99999989e-01 , 1.05023376e-04 , 3.94027384e+01],
                        [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

world_transform = np.array([[-9.99999994e-01, -2.38548938e-06,  1.05023635e-04,  2.71040009e+02],
                            [ 2.39684497e-06, -9.99999994e-01,  1.08124080e-04,  1.29490097e+02],
                            [ 1.05023376e-04,  1.08124331e-04,  9.99999989e-01,  3.81027069e+01],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


def ret_line_eq(pt1,pt2):
    '''
    Returns m1,c1 given 2 points
    '''
    points = [pt1,pt2]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m1, c1 = np.linalg.lstsq(A, y_coords)[0] #TODO: Improve this, it computes the least square solution
    return m1,c1

def find_midpoint(pt_ls):
    '''
    Find the midpoint(image coordinates) given 4 corners of the contour
    '''
    m1,c1 = ret_line_eq(pt_ls[0],pt_ls[2])
    m2,c2 = ret_line_eq(pt_ls[1],pt_ls[3])

    #Solve the 2 eqns to obtain the midpoint
    A = np.array([[-m1,1],
                 [-m2,1]],dtype=np.float64)
    B = np.array([c1,c2])
    midpoint = np.linalg.inv(A).dot(B)
    return midpoint 

def pot_parking_spot(orig_img,inf_img,points_2d_ls):
    '''
    Function to detect potential parking spot
    '''
    #TODO: Optimize the code
    xcoords_ls = []
    ycoords_ls = []
    for point in points_2d_ls:
        xcoords_ls.append(point[0][0])
        ycoords_ls.append(point[1][0])
    
    #Line equations of the top and bottom line
    coeff_top = np.polyfit(xcoords_ls[0:2],ycoords_ls[0:2],1)
    line_top = np.poly1d(coeff_top)

    coeff_bottom = np.polyfit(xcoords_ls[2:4],ycoords_ls[2:4],1)
    line_bottom = np.poly1d(coeff_bottom)
    # print("line_bottom",line_bottom)
    # print("line_top",line_top)
    flag_top = 0
    flag_bottom = 0
    #Points for potential parking spot
    pt_tl = []
    pt_tr = []
    pt_bl = []
    pt_br = []
    for x in range(0,inf_img.shape[1]):##inf_img.shape[1]-1,0,-1
        if(inf_img[int(line_top(x)),x] == 255 and flag_top == 0):
            pt_tl = [int(line_top(x)),x]
            pt_tr = [int(line_top(x+300)),int(x+300)]
            #cv2.circle(orig_img,(pt_tl[1],pt_tl[0]), 5, (0,0,255), -1)
            #cv2.circle(orig_img,(pt_tr[1],pt_tr[0]), 5, (255,0,255), -1)
            #cv2.imshow("skdksjds,",orig_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            flag_top = 1

        if(inf_img[int(line_bottom(x)),x] == 255 and flag_bottom == 0):
            pt_bl = [int(line_bottom(x)),x]
            pt_br = [int(line_bottom(x+200)),int(x+200)]
            # cv2.circle(orig_img,(pt_bl[1],pt_bl[0]), 5, (0,0,255), -1)
            # cv2.circle(orig_img,(pt_br[1],pt_br[0]), 5, (0,0,255), -1)
            # cv2_imshow(orig_img)
            flag_bottom = 1
        
        if(flag_top == 1 and flag_bottom ==1):
            # cv2_imshow(orig_img)
            break

    if(flag_top == 1 and flag_bottom == 1):
        cv2.line(orig_img,(pt_tl[1],pt_tl[0]),(pt_tr[1],pt_tr[0]),(0,0,255),2)
        cv2.line(orig_img,(pt_tr[1],pt_tr[0]),(pt_br[1],pt_br[0]),(0,0,255),2)
        cv2.line(orig_img,(pt_br[1],pt_br[0]),(pt_bl[1],pt_bl[0]),(0,0,255),2)
        cv2.line(orig_img,(pt_bl[1],pt_bl[0]),(pt_tl[1],pt_tl[0]),(0,0,255),2)
    
    pt_ls = [pt_bl,pt_br,pt_tr,pt_tl]
    return orig_img,pt_ls


class carla_utils():

    def __init__(self,intrinsic_mat):
        self.intrinsic_mat = intrinsic_mat
        weights_path = '/media/smart/5f69d8cc-649e-4c33-a212-c8bc4f16fc67/CARLA_0.8.4/PythonClient/ShelfNet-lw-cityscapes/weights/model_final_final_iisc_idd_16kweights.pth'
        self.auto_park_obj = auto_park_vision(weights_path)
        self.ext_mat = None
        self.points_2d_ls = None
        self.window_width = 1280
        self.window_height = 720
        self.seg_img = None
        self.rect_pts = None
        self.count=0
        # print("self.points_2d_ls",self.points_2d_ls)
        # print("self.ext_mat",self.ext_mat)
    
   
    def carla_world_2d(self,world_points):
        '''
        Takes an array of world points and converts them into corresponding image points
        '''
        
        pts_2d_ls = [] #List of 2d points
        for point in world_points:
            world_coord = np.asarray(point).reshape(4,-1)
            # print("world_coord.shape",world_coord.shape)
            # world_coord = np.array([[250.0 ,129.0 ,38.0 ,1.0]]).reshape(4,-1)
            cam_coord = np.linalg.inv(self.ext_mat) @ world_coord
            img_coord = self.intrinsic_mat @ cam_coord[:3]
            img_coord = np.array([img_coord[0]/img_coord[2],
                                img_coord[1]/img_coord[2],
                                img_coord[2]])

            if(img_coord[2] > 0):
                x_2d = self.window_width - img_coord[0]
                y_2d = self.window_height - img_coord[1]
                pts_2d_ls.append([x_2d,y_2d,1.0])
            
        return pts_2d_ls


    def run_seg(self,frame,ext_mat,car_pos):
        '''
        Function to run segmentation using carla image
        '''
        self.ext_mat = ext_mat
        
        car_pos = [[car_pos[0]-10,car_pos[1]-10,car_pos[2],1.0],[car_pos[0]-10,car_pos[1]+10,car_pos[2],1.0],
                    [car_pos[0]-20,car_pos[1]-10,car_pos[2],1.0],[car_pos[0]-20,car_pos[1]+10,car_pos[2],1.0]]
        self.points_2d_ls = self.carla_world_2d(car_pos)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        self.seg_img = self.auto_park_obj.forward_pass(frame_pil,img_path=None)
        res_img,self.rect_pts = pot_parking_spot(frame,self.seg_img,self.points_2d_ls)
        # display_image("Image",res_img)
        # print("self.rect_pts",self.rect_pts)
        #Error handling
        for i in range(0,len(self.rect_pts)):
            if(len(self.rect_pts[i]) == 0):
                return [0,0] #Failure
        # breakpoint()
        midpoint = find_midpoint(self.rect_pts)
        res_img=cv2.circle(res_img,(int(midpoint[1]),int(midpoint[0])),3,(255, 0, 0),-1)
        
        cv2.imwrite('./seg_out/{}_seg.png'.format(str(self.count)),res_img)
        #self.count=self.count+1
        return midpoint,self.rect_pts,res_img
        #Overlay segmented image on the original frame
        # overlay = np.copy(self.seg_img)
        # overlay = cv2.cvtColor(overlay,cv2.COLOR_GRAY2RGB)
        # alpha = 0.5
        # cv2.addWeighted(overlay, alpha, frame, 1 - alpha,0, frame)
        # display_image("Image",frame)
        # cv2_imshow(frame)







if __name__ == '__main__':
    # img = cv2.imread('/content/homography_computation/data/rbc_data_00061.png')
    carla_utils_obj = carla_utils(intrinsic_mat)
    ext_mat = np.array([[-2.58691501e-05, -2.12096950e-03, -9.99997750e-01,  2.49117455e+02],
                [ 1.00000000e+00, -1.72580384e-05, -2.58326045e-05,  1.29494645e+02],
                [ 1.72032094e-05,  9.99997751e-01, -2.12096994e-03,  3.94030672e+01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # points_ls = [[251.0,129.4,38.1,1.0],[251.0,139.4,38.1,1.0],
    #              [231.0,129.4,38.1,1.0],[231.0,139.4,38.1,1.0]]
    # points_2d_ls = carla_world_2d(points_ls) #[967,427],[295,438],[438,309],[817,300]
    # print("points_2d_ls",points_2d_ls)
   
    
    #For debug
    # count = 0
    # ret = True

    for img_path in glob.glob('/media/smart/5f69d8cc-649e-4c33-a212-c8bc4f16fc67/CARLA_0.8.4/PythonClient/Course1FinalProject/_out/*.png'):
 
        frame = cv2.imread(img_path)
        midpoint = carla_utils_obj.run_seg (frame,ext_mat,[229,129,38])
        print("midpoint",midpoint)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame_pil = Image.fromarray(frame)

    #     seg_img = auto_park_obj.forward_pass(frame_pil,img_path=None)

    #     #Overlay segmented image on the original frame
    #     overlay = np.copy(seg_img)
    #     overlay = cv2.cvtColor(overlay,cv2.COLOR_GRAY2RGB)
    #     alpha = 0.5
    #     cv2.addWeighted(overlay, alpha, frame, 1 - alpha,0, frame)
    #     # cv2_imshow(frame)
    #     res_img,pt_ls = pot_parking_spot(frame,seg_img,points_2d_ls)
    #     cv2_imshow(res_img)
    #     midpoint = find_midpoint(pt_ls)
    #     print(midpoint)
    #     cv2.circle(res_img,(int(midpoint[1]),int(midpoint[0])),radius=4,color=(0,0,255),thickness = 2)
    #     cv2_imshow(res_img)
