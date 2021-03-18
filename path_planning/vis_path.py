#Helper function to visualize path
import numpy as np 
import cv2
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

def world_2d(points_ls):
    '''
    Convert a list of 3d points to corresponding image points using homography
    '''
    H = np.array([[23.051566426191190, 38.859138964854033, 626.971467292441275],
            [5.111361028987754, -1.474310992555276, 7144.855081929711559],
            [0.036814327356280, -0.001812769201662, 1.0]])
    points_2d_ls = []
    for point in points_ls:
        point_3d = np.asarray(point)
        point_3d = point_3d.reshape(3,1)
        point_2d = np.dot(H,point_3d)
        point_2d = point_2d // point_2d[2,0]
        points_2d_ls.append(point_2d)
        # print("point_2d",point_2d)
    return points_2d_ls


def visualize_path(img,pt_list_3d):

    """ Visualize path
    :type pt_list_3d:
    :param pt_list_3d:

    :raises:

    :rtype:
    """
    # pt_ls_2d = world_2d(pt_list_3d)
    #TODO: Parallelize this computation
    pt_2d = np.asarray(pt_list_3d)
    for i in range(0,len(pt_2d)):
        pt_2d[i,2] = 1
    
    pt_ls_2d = world_2d(pt_2d)
    print("pt_ls_2d",pt_ls_2d)
    breakpoint()

    for i in range(0,len(pt_2d)):
        # try:
        print(pt_ls_2d[i][1])
        print(pt_ls_2d[i][0])
        # breakpoint()
        img=cv2.circle(img,(int(pt_ls_2d[i][1]),int(pt_ls_2d[i][0])),3,(255, 0, 0),-1)
        display_image("image",img)
        # breakpoint()
        # except:
        # print("E")
    
    display_image("Path planning",img)


