#Script to verify homography computation
import numpy as np
import cv2 
import sys

#Utils
def display_image(winname,frame):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,frame)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        exit(0)

def homography_verify(img_path):
    '''
    Function to verfiy estimated homography
    '''
    h = np.array([[1.780416746375038, -2.206070668840309,585.355926253062421],
                [0.886650069935394,-0.165293312304243,647.776136409074638],
                [0.002787095054000,-0.000173568950461,1.000000000000000]])
    world_coords = np.array([[0,0],[181.5,0],[181.5,-121],[0,-121]],dtype='float32')

    img_coords = []
    for i in range(0,world_coords.shape[0]):
        test = np.array([[world_coords[i]]])
        img_coords.append(cv2.perspectiveTransform(test,h))
    # print('img_coords:',img_coords)

    #Draw predictions on the image
    test_img = cv2.imread(img_path)
    for coord in img_coords:
        # print('coord[0]:',coord[0][0])
        cv2.circle(test_img,(int(coord[0][0][0]),int(coord[0][0][1])),5,(0,0,255),-1)
        

    display_image("test_image",test_img)        







if __name__ == '__main__':
    img_path = sys.argv[1]
    homography_verify(img_path)