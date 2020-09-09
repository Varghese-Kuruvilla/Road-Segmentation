#Python script to implement paper 'Road Scence Segmentation from a Single Image'
#by J.M Alvarez et al

import cv2 
import numpy as np  
import matplotlib.pyplot as plt
import glob

#Utils
def breakpoint():
  inp = input("Waiting for input...")


def display_image(winname,frame):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,frame)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        exit(0)


def proto_seg(img_path):
    bgr_img = cv2.imread(img_path)
    #Mask top half of the image
    mask_img = np.zeros_like(bgr_img[:,:,0])
    mask_img[int(bgr_img.shape[0]/2):int(bgr_img.shape[0]),:] = [255]
    bgr_img = cv2.bitwise_and(bgr_img,bgr_img,mask=mask_img)

    lab_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2LAB)
    #Colour channels used
    r_img = bgr_img[:,:,2]
    g_img = bgr_img[:,:,1]
    b_img = bgr_img[:,:,0]
    l_img = lab_img[:,:,0]
    a_img = lab_img[:,:,1]
    b_img = lab_img[:,:,2]

    train_region_range = [range(690,720),range(600,680)]
    roi = [690,720,600,680]
    # display_image("RGB Image",rgb_img)

    #Selecting 30*80 patch from the bottom middle of the image
    #Dim(train_region): 6*30*80
    N = 6 #represents the number of color planes used
    train_region = np.array([r_img[roi[0]:roi[1],roi[2]:roi[3]],\
                            g_img[roi[0]:roi[1],roi[2]:roi[3]],\
                            b_img[roi[0]:roi[1],roi[2]:roi[3]],\
                            l_img[roi[0]:roi[1],roi[2]:roi[3]],\
                            a_img[roi[0]:roi[1],roi[2]:roi[3]],\
                            b_img[roi[0]:roi[1],roi[2]:roi[3]]])
    train_region = train_region.reshape(6,-1) 
    cov = np.cov(train_region)
    cov_inv = np.linalg.inv(cov)
    ones = np.ones((N,1))
    #Calculate w(weights) based on the formula : w = cov_inv*I*((I.transpose()*cov_inv*I))^-1
    w = cov_inv @ ones @ np.linalg.inv(ones.transpose() @ cov_inv @ ones)

    #Compute y as follows:
    #y(i) = summation from j=1 to N (w_j @ x_j(i))
    color_plane_img = np.array([r_img,\
                                g_img,\
                                b_img,\
                                l_img,\
                                a_img,\
                                b_img])
    y = np.zeros_like(r_img)
    for rows in range(0,r_img.shape[0]):
        for cols in range(0,r_img.shape[1]):
            # compute = w.transpose() @ color_plane_img[:,rows,cols]
            y[rows,cols] = w.transpose() @ color_plane_img[:,rows,cols]
    
    # display_image("y:",y)

    #Split the image into 4 images of equal size
    num_rows = 2
    num_cols =  3
    U_ls = [] #Store the list of histogram uniformity values
    for i in range(1,2):
        for j in range(0,num_cols):
            split_img = y[int(i*y.shape[0]/num_rows):int((i*y.shape[0]/num_rows)+(y.shape[0]/num_rows)),\
                            int(j*y.shape[1]/num_cols):int((j*y.shape[1]/num_cols)+(y.shape[1]/num_cols))]
            #Compute image histogram- 10 bins , range:0-256
            img_hist = cv2.calcHist([split_img],[0],None,[256],[0,256])
            #Normalize img_hist
            img_hist = np.true_divide(img_hist,np.sum(img_hist,axis=0))
            img_hist = np.power(img_hist,2)
            U = np.sum(img_hist,axis=0)
            U_ls.append(U)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    #demarcate regions on the input image 
    for i in range(0,(num_cols)):
        print("i*(bgr_img.shape[1]/num_cols",i*(bgr_img.shape[1]/num_cols))
        cv2.line(bgr_img,(int(i*(bgr_img.shape[1]/num_cols)),int(bgr_img.shape[0]/num_rows)),\
                        (int(i*(bgr_img.shape[1]/num_cols)),int(bgr_img.shape[0])),(0,0,255),2)
        cv2.putText(bgr_img,'U'+str(U_ls[i]),(int(i*(bgr_img.shape[1]/(1.0*num_cols))),540), font, 1,(255,255,255),2)
    display_image("Uniformity values",bgr_img)




























if __name__ == '__main__':
    for img_path in glob.glob('/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/*.jpeg'):
        # img_path = '/home/varghese/Transvahan/demo/autonomous_parking_vision/Road_scence_segmentation_alvarez/ZED_image811.jpeg'
        proto_seg(img_path)