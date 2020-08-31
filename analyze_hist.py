#Script to analyze colour ranges of road views
import cv2 
import numpy as np
import numpy.ma as ma 
import glob



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


def color_range(img_path):
    '''
    Function to find out colour range of the image
    '''
    rgb_img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2HSV)
    s_img = hsv_img[:,:,1]
    v_img = hsv_img[:,:,2]
    # display_image('s_img:',s_img)
    # display_image('v_img',v_img)
    #Assuming that the ROI is pasted on a white background
    #Saturation values for white: 0, Value channel values for white:255

    sat_mask_array = np.ones_like(s_img)
    val_mask_array = np.ones_like(v_img)

    for i in range(0,sat_mask_array.shape[0]):
        for j in range(0,sat_mask_array.shape[1]):
            if(s_img[i][j] != 0):
                sat_mask_array[i][j] = 0
            if(v_img[i][j] <= 240):
                val_mask_array[i][j] = 0
            
                # cv2.circle(v_img,(i,j),3,(0,0,255),thickness=1,lineType=8,shift=0)
                # display_image("v_img",v_img)
                # breakpoint()

                
    
                
    sat_masked = ma.masked_array(s_img,sat_mask_array)
    val_masked = ma.masked_array(v_img,val_mask_array)

    
    # display_image('s_img',sat_masked)
    # display_image('v_img',val_masked)
    sat_max_ls = []
    sat_min_ls = []
    val_max_ls = []
    val_min_ls = []
    #Find out colour range
    sat_max_ls.append(np.max(sat_masked))
    sat_min_ls.append(np.min(sat_masked))
    val_max_ls.append(np.max(val_masked))
    val_min_ls.append(np.min(val_masked))

    sat_range = [max(sat_max_ls),min(sat_min_ls)]
    val_range = [max(val_max_ls),min(val_min_ls)]

    return sat_range,val_range




if __name__ == '__main__':
    for img_path in glob.glob('/home/varghese/Transvahan/demo/autonomous_parking_vision/histogram_analysis/*_road.jpeg'):
        # img_path = '/home/varghese/Transvahan/demo/autonomous_parking_vision/python_scripts/ZED_image159.jpeg'
        print('img_path:',img_path)
        sat_range,val_range = color_range(img_path)
        print('val_range:',val_range)
        print("breakpoint")
    
    print("sat_range:",sat_range)
    print("val_range:",val_range)