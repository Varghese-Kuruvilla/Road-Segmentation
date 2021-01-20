# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA waypoint follower assessment client script.

A controller assessment to follow a given trajectory, where the trajectory
can be defined using way-points.

STARTING in a moment...
"""
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
#import csv
import random
import cv2


import live_plotter as lv   # Custom live plotting library
import controller2d
import configparser 
import segmentationobj
import darknet_proper_fps
import module_7_clean
from module_7_clean import *


sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import carla
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
from carla.sensor import Camera
from carla.util import print_over_same_line, StopWatch
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array
from carla.transform import Transform


"""
Configurable params
"""

WAIT_TIME_BEFORE_START = 5.00   # game seconds (time before controller start)


PLAYER_START_INDEX = 0      # spawn index for player (keep to 1)

DIST_THRESHOLD_TO_LAST_WAYPOINT = 0.5  # some distance from last position before
                                       # simulation ends
                                       
# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying


number_of_frames = 3000
frame_step = 10  # Save 10 image every 100 frames
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


def exec_waypoint_nav_demo(args):
    """ Executes waypoint navigation demo.
    """
    with make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')

        settings,camera2 = make_carla_settings(args)

        scene = client.load_settings(settings)

        player_start = PLAYER_START_INDEX #which model car (player) to choose

        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        #locate the options file
        config = configparser.ConfigParser()
        config.read(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))         
        demo_opt = config['Demo Parameters']

        # Get options
        enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        # Set options
        live_plot_timer = Timer(live_plot_period)
        #get data from carla client
        measurement_data, sensor_data = client.read_data()

        #read the images from camera
        image_RGB = to_rgb_array(sensor_data['CameraRGB'])
        image_depth=to_rgb_array(sensor_data['CameraDepth'])
        im_bgr = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)        

        #initial 3d world location of the car
        ini_location=([
                   measurement_data.player_measurements.transform.location.x,
                    measurement_data.player_measurements.transform.location.y,
                    measurement_data.player_measurements.transform.location.z])

        #compute extrinsic matrix 
        extrinsic,newcam= get_extrinsic(measurement_data,camera2)

        #initiate the segmentation pipeline
        carla_utils_obj = segmentationobj.carla_utils(intrinsic)
        get_2d_point,pointsmid,res_img=carla_utils_obj.run_seg(im_bgr,extrinsic,ini_location)

        #convert the 2d midpoint to 3d world location-as the goal location
        sec=module_7_clean._2d_to_3d_roi([get_2d_point[0],get_2d_point[1]],image_depth,extrinsic)

        print("present location",ini_location)
        print('destination location',sec)

        #generate the waypoints using initial and goal points
        waypoints_np = np.array(get_discrete_waypts(ini_location,sec))
        
        #interpolate the waypoints for smoother trajectory
        waypoints,wp_interp_hash,wp_distance,wp_interp=dis_to_continous(waypoints_np)

        #pass the waypoints to the controller algorithm
        controller = controller2d.Controller2D(waypoints)

        #few initiallizations for live plotting,object detection
        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        TOTAL_EPISODE_FRAMES=total_frames(client,sim_start_stamp)

        
        start_x, start_y, start_yaw = get_current_pose(measurement_data)
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history     = [start_x]
        y_history     = [start_y]
        yaw_history   = [start_yaw]
        time_history  = [0]
        speed_history = [0]
        #liveplotting object initialization
        liveobj=live_all(waypoints_np,start_x,start_y,TOTAL_EPISODE_FRAMES,INTERP_MAX_POINTS_PLOT,enable_live_plot)

        #get the 3d location of the ROI corners which is to be tracked for object detection 
        sec1=module_7_clean._2d_to_3d_roi([pointsmid[0][0],pointsmid[0][1]],image_depth,extrinsic)
        sec2=module_7_clean._2d_to_3d_roi([pointsmid[1][0],pointsmid[1][1]],image_depth,extrinsic)
        sec3=module_7_clean._2d_to_3d_roi([pointsmid[2][0],pointsmid[2][1]],image_depth,extrinsic)
        sec4=module_7_clean._2d_to_3d_roi([pointsmid[3][0],pointsmid[3][1]],image_depth,extrinsic)
        update_pts=[list(sec1),list(sec2),list(sec3),list(sec4)]

        reached_the_end = False
        skip_first_frame = True
        closest_index    = 0  # Index of waypoint that is currently closest to
                              # the car (assumed to be the first index)
        closest_distance = 0  # Closest distance of closest waypoint to car
        counter=0
        
        for frame in range(TOTAL_EPISODE_FRAMES):
            # Gather current data from the CARLA server
            measurement_data, sensor_data = client.read_data()
            image_RGB = to_rgb_array(sensor_data['CameraRGB'])
            im_bgr = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)
            image_depth=to_rgb_array(sensor_data['CameraDepth'])
            # Update pose, timestamp
            current_x, current_y, current_yaw = \
                get_current_pose(measurement_data)
            current_speed = measurement_data.player_measurements.forward_speed
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0

            # Wait for some initial time before starting the demo
            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)

                continue
            else:
                current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START
            #log the data 
            x_history.append(current_x)
            y_history.append(current_y)
            yaw_history.append(current_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp)

            #check if there is a object in ROI or approaching the ROI,defined in 2d by pts_2d_ls
            pts_2d_ls=find_pts(update_pts,measurement_data,newcam)
            flagbox=chk_obj(im_bgr,pts_2d_ls,image_depth,pointsmid)

            #flag to check if obstacle is in ROI,if no object control the car to goal point  
            if flagbox==False:

                new_waypoints=subset_waypoints(closest_index,current_x,current_y,waypoints_np,wp_interp_hash,wp_distance,wp_interp)
                controller.update_waypoints(new_waypoints)

            # Update the other controller values and controls
                controller.update_values(current_x, current_y, current_yaw, 
                                     current_speed,
                                     current_timestamp, frame)
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()

            # Skip the first frame (so the controller has proper outputs)
                

                if skip_first_frame and frame == 0:
                    pass
                else:
                    plot_live_plot(skip_first_frame,frame,liveobj,live_plot_timer,current_x,current_y,new_waypoints,current_timestamp,current_speed,controller,cmd_throttle, cmd_steer, cmd_brake,enable_live_plot)
                    

            # Output controller command to CARLA server
                send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake)

                dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,
                waypoints[-1][1] - current_y]))
                #check if the car has reached the end by using a distance threshold
                if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                    reached_the_end = True
                if reached_the_end:
                    break
            else:
                #if there is a obstacle in ROI,brake immediately
                send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
                break

        # End of demo - Stop vehicle and Store outputs to the controller output
        # directory.
        if reached_the_end:
            print("Reached the end of path. Writing to controller_output...")
            send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
        else:
            print("stop!!!!.")
        # Stop the car
        #send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
        # Store the various outputs
        store_trajectory_plot(liveobj.trajectory_fig.fig, 'trajectory.png')
        store_trajectory_plot(liveobj.forward_speed_fig.fig, 'forward_speed.png')
        store_trajectory_plot(liveobj.throttle_fig.fig, 'throttle_output.png')
        store_trajectory_plot(liveobj.brake_fig.fig, 'brake_output.png')
        store_trajectory_plot(liveobj.steer_fig.fig, 'steer_output.png')
        write_trajectory_file(x_history, y_history, speed_history, time_history)
def main():
    """Main function.

    Args:
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
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
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    # Execute when server connection is established
    while True:
        try:
            exec_waypoint_nav_demo(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')