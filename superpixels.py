#Superpixels SLIC
import cv2
import numpy as np 
from fast_slic import Slic 
# from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

#For timing information
import time 

#Utils
def display_image(winname,frame):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,frame)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        exit(0)


def test_slic(img_path):
    '''
    Test superpixel segmentation SLIC algorithm on images
    '''
    img = cv2.imread(img_path)
    mask_img = np.zeros_like(img[:,:,0])
    mask_img[int(img.shape[0]/2):int(img.shape[0]),:] = [255]
    img = cv2.bitwise_and(img,img,mask=mask_img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    display_image("img",img)

    
    slic = Slic(num_components=10,compactness=0.1)
    assignment = slic.iterate(img)
    assignment = np.uint8(assignment)
    img_seg = np.zeros_like(img)
    final_seg = np.zeros_like(img)
    mask_img = np.zeros_like(img[:,:,0])
    #Assign colours to segmentation map
    searchval = np.min(assignment)
    for searchval in range(np.min(assignment),np.max(assignment)+1):
        #Generate random colour for seg map
        color_val = list(np.random.choice(range(256),size=3))
        color = [int(color_val[0]),int(color_val[1]),int(color_val[2])]
        img_seg[:,:,:] = color
       
        mask_img = np.uint8(np.where(assignment == searchval,255,0))
        img_seg = cv2.bitwise_and(img_seg,img_seg,mask=mask_img)
        final_seg = final_seg + img_seg
    display_image("final_seg",final_seg)       
            
    
    # print('assignment:',assignment)
    # print('slic.slic_model.clusters:',slic.slic_model.clusters)




if __name__ == '__main__':
    import timeit
    # img_path = '/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/ZED_image378.jpeg'
    img_path = '/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/ZED_image735.jpeg'
    # start_time = time.time()
    test_slic(img_path)
    # print("Time taken for completion:",time.time()-start_time)
    # time = (timeit.timeit(stmt='test_slic(img_path)',setup='from __main__ import test_slic,img_path',number=1))
    # print("Time taken:",time)
    # print(timeit.timeit("test_slic(img_path)", setup="from __main__ import test_slic,img_path"))