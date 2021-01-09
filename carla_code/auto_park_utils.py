import torch
import torchvision.transforms as transforms
import cv2 
from PIL import Image
import numpy as np 
import glob
import sys

sys.path.insert(1, '/media/smart/5f69d8cc-649e-4c33-a212-c8bc4f16fc67/CARLA_0.8.4/PythonClient/ShelfNet-lw-cityscapes/ShelfNet18_realtime/')
from evaluate import MscEval
from shelfnet import ShelfNet

class auto_park_vision():
    def __init__(self,weights_path):

        self.n_classes = 19
        #Homography matrix
        self.H = np.array([[1.132372123443505,-3.359375305984110, 843.835898580380217],
                [0.604347080571571 ,-0.324814498163494 ,955.938833468783173],
                [0.001378608429514 ,-0.000531862797747, 1.000000000000000]])
        #Point indicating potential parking spot
        self.midpoint = []
        self.point_3d = None #3D point corresponding to self.midpoint
        self.eval_define(weights_path) #Define Object of class MscEval
        #self.evaluator is an object of the class MscEval

    def forward_pass(self,frame=None,img_path=None):

        if(img_path != None):
            img = Image.open(img_path)
        else:
            img = frame 
        orig_img = np.array(img)
        # orig_img = cv2.imread(img_path)
        # cv2_imshow(img)
        #Preprocess Image
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        img = to_tensor(img)
        

        _, H, W = img.shape
        # print("H,W:",H,W)
        #Change image size to the form NCHW from CHW
        img = img.unsqueeze(0)

        probs = torch.zeros((1, self.n_classes, H, W))
        probs.requires_grad = False
        img = img.cuda()        

        # for sc in self.scales:
        prob = self.evaluator.scale_crop_eval(img, scale=1.0) #prob.type torch.cuda.FloatTensor
        prob = prob.detach().cpu()
        prob = prob.data.numpy()
        preds = np.argmax(prob, axis=1) #preds.dtype int64
        # palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        # pred = palette[preds.squeeze()]

        #Changed 
        preds = preds.squeeze().astype(np.uint8)
        preds[preds == 0] = 255
        preds = preds.astype(np.uint8)
        return preds


    def image_to_world(self,point_2d):
        '''
        Function to find out world coordinates
        given an image point
        '''
        #TODO: Check homography matrix
        # print("point.shape",point.shape)
        #Conversion to homogenous coordinates
        point_2d = np.append(point_2d,1).reshape(-1,1)

        H_inv = np.linalg.inv(self.H)
        self.point_3d = np.dot(H_inv,point_2d)
        self.point_3d = self.point_3d / self.point_3d[2,0]
        return self.point_3d




    def eval_define(self,weights_path):

        n_classes = self.n_classes
        net = ShelfNet(n_classes=n_classes)

        net.load_state_dict(torch.load(weights_path))
        net.cuda()
        net.eval()
        self.evaluator = MscEval(net, dataloader=None, scales=[1.0],flip=False)