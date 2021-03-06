#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB), and the INTEL Visual Computing Lab.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client to generate point cloud in PLY format that you
   can visualize with MeshLab (meshlab.net) for instance. Please
   refer to client_example.py for a simpler and more documented example."""

from __future__ import print_function

import argparse
import logging
import os
import random
import time
import sys
import glob
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import carla
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line, StopWatch
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array
from carla.transform import Transform
import cv2
import numpy as np
from numpy.linalg import pinv, inv
import math
import re
import segmentation
def run_carla_client(host, port, far):
    # Here we will run a single episode with 300 frames.
    number_of_frames = 3000
    frame_step = 10  # Save one image every 100 frames
    output_folder = '_out'
    image_size = [1280, 720]
    camera_local_pos = [0.7, 0.0, 1.3] # [X, Y, Z]
    camera_local_rotation = [0, 0, 0]  # [pitch(Y), yaw(Z), roll(X)]
    fov = 59
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    MINI_WINDOW_WIDTH = 320
    MINI_WINDOW_HEIGHT = 180

    WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
    WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2

    k = np.identity(3)
    k[0, 2] = WINDOW_WIDTH_HALF
    k[1, 2] = WINDOW_HEIGHT_HALF
    k[0, 0] = k[1, 1] = WINDOW_WIDTH / \
        (2.0 * math.tan(59.0 * math.pi / 360.0))
    intrinsic=k
    # Connect with the server
    with make_carla_client(host, port) as client:
        print('CarlaClient connected')

        # Here we load the settings.
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=20,
            NumberOfPedestrians=40,
            WeatherId=1)
        settings.randomize_seeds()

        camera1 = Camera('CameraDepth', PostProcessing='Depth', FOV=fov)
        camera1.set_image_size(*image_size)
        camera1.set_position(*camera_local_pos)
        camera1.set_rotation(*camera_local_rotation)
        settings.add_sensor(camera1)

        camera2 = Camera('CameraRGB', PostProcessing='SceneFinal', FOV=fov)
        camera2.set_image_size(*image_size)
        camera2.set_position(*camera_local_pos)
        camera2.set_rotation(*camera_local_rotation)
        settings.add_sensor(camera2)
        #scene = client.load_settings(settings)
        client.load_settings(settings)
        #print("sjdsjhdjshdjshdjshgds",scene.player_start_spots[0].location.x)
        # Start at location index id '0'
        client.start_episode(0)

        # Compute the camera transform matrix
        camera_to_car_transform = camera2.get_unreal_transform()
        print("camera_to_car_transform",camera_to_car_transform)
        carla_utils_obj = segmentation.carla_utils(intrinsic)
        
        
        # Iterate every frame in the episode except for the first one.
        for frame in range(1, number_of_frames):
            # Read the data produced by the server this frame.
            measurements, sensor_data = client.read_data()
            

            # Save one image every 'frame_step' frames
            if not frame % frame_step:
                # Start transformations time mesure.
                

                # RGB image [[[r,g,b],..[r,g,b]],..[[r,g,b],..[r,g,b]]]
                image_RGB = to_rgb_array(sensor_data['CameraRGB'])
                image_depth=to_rgb_array(sensor_data['CameraDepth'])
                #measurements.player_measurements.transform.location.x-=40
                #measurements.player_measurements.transform.location.z-=2
                world_transform = Transform(
                    measurements.player_measurements.transform
                )

                im_bgr = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)
                
                pos_vector=([
                   measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                
                
                
                fdfd=str(world_transform)
                sdsd=fdfd[1:-1].split('\n')
                new=[]
                for i in sdsd:
                    dd=i[:-1].lstrip('[ ')
                    ff=re.sub("\s+", ",", dd.strip())
                    gg=ff.split(',')
                    new.append([float(i) for i in gg])
                newworld=np.array(new)
                fdfd=str(camera_to_car_transform)
                sdsd=fdfd[1:-1].split('\n')
                new=[]
                for i in sdsd:
                    dd=i[:-1].lstrip('[ ')
                    ff=re.sub("\s+", ",", dd.strip())
                    gg=ff.split(',')
                    new.append([float(i) for i in gg])
                newcam=np.array(new)
                extrinsic = np.matmul(newworld,newcam)
                get_2d_point=carla_utils_obj.run_seg(im_bgr,extrinsic,pos_vector)
                print(get_2d_point)
                depth=image_depth[int(get_2d_point[0]),int(get_2d_point[1])]
                scale=depth[0]+depth[1]*256+depth[2]*256*256
                scale=scale/((256*256*256) - 1 )
                depth=scale*1000
                pos2d=np.array([(WINDOW_WIDTH-get_2d_point[1])*depth,(WINDOW_HEIGHT-get_2d_point[0])*depth,depth])
                
                first=np.dot(np.linalg.inv(intrinsic),pos2d)
                first=np.append(first,1)
                sec=np.matmul((extrinsic),first)
                print("present",pos_vector)
                print('goal-',sec)
            client.send_control(
                measurements.player_measurements.autopilot_control
            )



def check_far(value):
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(
            "{} must be a float between 0.0 and 1.0")
    return fvalue


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-f', '--far',
        default=0.2,
        type=check_far,
        help='The maximum save distance of camera-point '
             '[0.0 (near), 1.0 (far)] (default: 0.2)')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:
            run_carla_client(host=args.host, port=args.port, far=args.far)
            print('\nDone!')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nClient stoped by user.')
