#!/usr/bin/env python3

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
import csv
import matplotlib.pyplot as plt
import controller2d
import configparser 


import sys
import glob
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import carla

import live_plotter as lv   # Custom live plotting library

from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
import random
import segmentation
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line, StopWatch
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array
from carla.transform import Transform
import cv2
import re
import segmentationobj
import darknet_proper_fps
"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 5.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 200.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 0      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

PLAYER_START_INDEX = 0      # spawn index for player (keep to 1)
FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1    
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

WAYPOINTS_FILENAME = 'racetrack_waypoints.txt'  # waypoint file to load
DIST_THRESHOLD_TO_LAST_WAYPOINT = 0.5  # some distance from last position before
                                       # simulation ends
                                       
# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20   # lookahead in meters
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

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
thresh= 0.55
configPath = "/media/smart/My Passport/darknet/cfg/yolo-obj.cfg"
weightPath = "/media/smart/My Passport/darknet/backup/yolo-obj_best.weights"
metaPath= "/media/smart/My Passport/darknet/obj.data"
showImage= False
makeImageOnly = False
initOnly= False
def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()
    #settings = CarlaSettings()
    settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=2,
            NumberOfPedestrians=10,
            WeatherId=1)
    #settings.randomize_seeds()
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
    
    return settings,camera2
class Timer(object):
    """ Timer Class
    
    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements
    
    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, yaw)

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene
    
    Obtains the player x,y, and yaw pose from the scene.

    Args:
        scene: The CARLA scene object

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

def send_control_command(client, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.
    
    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i]))

def exec_waypoint_nav_demo(args):
    """ Executes waypoint navigation demo.
    """

    with make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')

        settings,camera2 = make_carla_settings(args)

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.

        scene = client.load_settings(settings)

        # Refer to the player start folder in the WorldOutliner to see the 
        # player start information
        player_start = PLAYER_START_INDEX

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        #############################################
        # Load Configurations
        #############################################

        # Load configuration file (options.cfg) and then parses for the various
        # options. Here we have two main options:
        # live_plotting and live_plotting_period, which controls whether
        # live plotting is enabled or how often the live plotter updates
        # during the simulation run.
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

        #############################################
        # Load Waypoints
        #############################################
        # Opens the waypoint file and stores it to "waypoints"
        camera_to_car_transform = camera2.get_unreal_transform()
        
        carla_utils_obj = segmentationobj.carla_utils(intrinsic)
        measurement_data, sensor_data = client.read_data()
        measurements=measurement_data
        image_RGB = to_rgb_array(sensor_data['CameraRGB'])
        image_depth=to_rgb_array(sensor_data['CameraDepth'])
                
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
        #print("dfjdfjkjfdkf",extrinsic)
        get_2d_point,pointsmid,res_img=carla_utils_obj.run_seg(im_bgr,extrinsic,pos_vector)
        #print(get_2d_point)
        #dis1=((get_2d_point[1]-pointsmid[0][1]),(get_2d_point[0]-pointsmid[0][0]))
        #dis2=((get_2d_point[1]-pointsmid[1][1]),(get_2d_point[0]-pointsmid[1][0]))
        #dis3=((get_2d_point[1]-pointsmid[2][1]),(get_2d_point[0]-pointsmid[2][0]))
        #dis4=((get_2d_point[1]-pointsmid[3][1]),(get_2d_point[0]-pointsmid[3][0]))
        #flagbox=darknet_proper_fps.performDetect(im_bgr,thresh,configPath, weightPath, metaPath, showImage, makeImageOnly, initOnly,pointsmid)
        depth1=image_depth[int(pointsmid[0][0]),int(pointsmid[0][1])]
        depth2=image_depth[int(pointsmid[1][0]),int(pointsmid[1][1])]
        depth3=image_depth[int(pointsmid[2][0]),int(pointsmid[2][1])]
        depth4=image_depth[int(pointsmid[3][0]),int(pointsmid[3][1])]
        
        scale1=depth1[0]+depth1[1]*256+depth1[2]*256*256
        scale1=scale1/((256*256*256) - 1 )
        depth1=scale1*1000
        pos2d1=np.array([(WINDOW_WIDTH-pointsmid[0][1])*depth1,(WINDOW_HEIGHT-pointsmid[0][0])*depth1,depth1])
        first1=np.dot(np.linalg.inv(intrinsic),pos2d1)
        first1=np.append(first1,1)
        sec1=np.matmul((extrinsic),first1)
        
        scale2=depth2[0]+depth2[1]*256+depth2[2]*256*256
        scale2=scale2/((256*256*256) - 1 )
        depth2=scale2*1000
        pos2d2=np.array([(WINDOW_WIDTH-pointsmid[1][1])*depth2,(WINDOW_HEIGHT-pointsmid[1][0])*depth2,depth2])
        first2=np.dot(np.linalg.inv(intrinsic),pos2d2)
        first2=np.append(first2,1)
        sec2=np.matmul((extrinsic),first2)

        scale3=depth3[0]+depth3[1]*256+depth3[2]*256*256
        scale3=scale3/((256*256*256) - 1 )
        depth3=scale3*1000
        pos2d3=np.array([(WINDOW_WIDTH-pointsmid[2][1])*depth3,(WINDOW_HEIGHT-pointsmid[2][0])*depth3,depth3])
        first3=np.dot(np.linalg.inv(intrinsic),pos2d3)
        first3=np.append(first3,1)
        sec3=np.matmul((extrinsic),first3)

        scale4=depth4[0]+depth4[1]*256+depth4[2]*256*256
        scale4=scale4/((256*256*256) - 1 )
        depth4=scale4*1000
        pos2d4=np.array([(WINDOW_WIDTH-pointsmid[3][1])*depth4,(WINDOW_HEIGHT-pointsmid[3][0])*depth4,depth4])
        first4=np.dot(np.linalg.inv(intrinsic),pos2d4)
        first4=np.append(first4,1)
        sec4=np.matmul((extrinsic),first4)
     
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
        pos_vector[2]=4.0
        ini=pos_vector
        goal=list(sec)
        goal[2]=1.08
        aa=ini[0]
        dec=abs(ini[1]-goal[1])/abs(aa-(goal[0]))
        f1=open(WAYPOINTS_FILENAME,'a+')
        for i in range(int(goal[0]),int(aa)):
    
    
    # print(int(goal[0])-i)
            if abs((int(aa)-i))<10:
        
                ini=[ini[0]-1,ini[1]+dec,ini[2]-0.03]
            else:
                ini=[ini[0]-1,ini[1]+dec,ini[2]]
            #if i<int(aa)-1:
            f1.write(str(ini[0])+','+str(ini[1])+','+str(ini[2])+'\n')
            #else:
                #f1.write(str(ini[0])+','+str(ini[1])+','+str(ini[2]))
            

        waypoints_file = WAYPOINTS_FILENAME
        waypoints_np   = None
        with open(waypoints_file) as waypoints_file_handle:
            waypoints = list(csv.reader(waypoints_file_handle, 
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            waypoints_np = np.array(waypoints)
        #print("dfjdjfhjdhfjdfhjdfdjdfdufh",waypoints_np)
        print((waypoints_np))
        

        # Because the waypoints are discrete and our controller performs better
        # with a continuous path, here we will send a subset of the waypoints
        # within some lookahead distance from the closest point to the vehicle.
        # Interpolating between each waypoint will provide a finer resolution
        # path and make it more "continuous". A simple linear interpolation
        # is used as a preliminary method to address this issue, though it is
        # better addressed with better interpolation methods (spline 
        # interpolation, for example). 
        # More appropriate interpolation methods will not be used here for the
        # sake of demonstration on what effects discrete paths can have on
        # the controller. It is made much more obvious with linear
        # interpolation, because in a way part of the path will be continuous
        # while the discontinuous parts (which happens at the waypoints) will 
        # show just what sort of effects these points have on the controller.
        # Can you spot these during the simulation? If so, how can you further
        # reduce these effects?
        
        # Linear interpolation computations
        # Compute a list of distances between waypoints
        wp_distance = []   # distance array
        for i in range(1, waypoints_np.shape[0]):
            wp_distance.append(
                    np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                            (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
        wp_distance.append(0)  # last distance is 0 because it is the distance
                               # from the last waypoint to the last waypoint

        # Linearly interpolate between waypoints and store in a list
        wp_interp      = []    # interpolated values 
                               # (rows = waypoints, columns = [x, y, v])
        wp_interp_hash = []    # hash table which indexes waypoints_np
                               # to the index of the waypoint in wp_interp
        interp_counter = 0     # counter for current interpolated point index
        for i in range(waypoints_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            wp_interp.append(list(waypoints_np[i]))
            wp_interp_hash.append(interp_counter)   
            interp_counter+=1
            
            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                         float(INTERP_DISTANCE_RES)) - 1)
            wp_vector = waypoints_np[i+1] - waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)
            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter+=1
        # add last waypoint at the end
        wp_interp.append(list(waypoints_np[-1]))
        wp_interp_hash.append(interp_counter)   
        interp_counter+=1

        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        controller = controller2d.Controller2D(waypoints)

        #############################################
        # Determine simulation average timestep (and total frames)
        #############################################
        # Ensure at least one frame is used to compute average timestep
        num_iterations = ITER_FOR_SIM_TIMESTEP
        if (ITER_FOR_SIM_TIMESTEP < 1):
            num_iterations = 1

        # Gather current data from the CARLA server. This is used to get the
        # simulator starting game time. Note that we also need to
        # send a command back to the CARLA server because synchronous mode
        # is enabled.
        
        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        # Send a control command to proceed to next iteration.
        #print("dddddddddddddddddddddddddddddddddddddddddddddd",sim_start_stamp)
        # This mainly applies for simulations that are in synchronous mode.
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        # Computes the average timestep based on several initial iterations
        sim_duration = 0
        for i in range(num_iterations):
            # Gather current data
            measurement_data, sensor_data = client.read_data()
            # Send a control command to proceed to next iteration
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            # Last stamp
            if i == num_iterations - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 -\
                               sim_start_stamp  
        
        # Outputs average simulation timestep and computes how many frames
        # will elapse before the simulation should end based on various
        # parameters that we set in the beginning.
        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        print("SERVER SIMULATION STEP APPROXIMATION: " + \
              str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                               SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

        #############################################
        # Frame-by-Frame Iteration and Initialization
        #############################################
        # Store pose history starting from the start position
        measurement_data, sensor_data = client.read_data()
        start_x, start_y, start_yaw = get_current_pose(measurement_data)
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history     = [start_x]
        y_history     = [start_y]
        yaw_history   = [start_yaw]
        time_history  = [0]
        speed_history = [0]

        #############################################
        # Vehicle Trajectory Live Plotting Setup
        #############################################
        # Uses the live plotter to generate live feedback during the simulation
        # The two feedback includes the trajectory feedback and
        # the controller feedback (which includes the speed tracking).
        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        lp_1d = lv.LivePlotter(tk_title="Controls Feedback")
        
        ###
        # Add 2D position / trajectory plot
        ###
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

        trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed 
                                           # coordinate system the X
                                           # axis in the graph is flipped
        trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

        # Add waypoint markers
        trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                                 x0=waypoints_np[:,0], y0=waypoints_np[:,1],
                                 linestyle="-", marker="", color='g')
        # Add trajectory markers
        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x]*TOTAL_EPISODE_FRAMES, 
                                 y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])
        # Add lookahead path
        trajectory_fig.add_graph("lookahead_path", 
                                 window_size=INTERP_MAX_POINTS_PLOT,
                                 x0=[start_x]*INTERP_MAX_POINTS_PLOT, 
                                 y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                 color=[0, 0.7, 0.7],
                                 linewidth=4)
        # Add starting position marker
        trajectory_fig.add_graph("start_pos", window_size=1, 
                                 x0=[start_x], y0=[start_y],
                                 marker=11, color=[1, 0.5, 0], 
                                 markertext="Start", marker_text_offset=1)
        # Add end position marker
        trajectory_fig.add_graph("end_pos", window_size=1, 
                                 x0=[waypoints_np[-1, 0]], 
                                 y0=[waypoints_np[-1, 1]],
                                 marker="D", color='r', 
                                 markertext="End", marker_text_offset=1)
        # Add car marker
        trajectory_fig.add_graph("car", window_size=1, 
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)

        ###
        # Add 1D speed profile updater
        ###
        forward_speed_fig =\
                lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        forward_speed_fig.add_graph("forward_speed", 
                                    label="forward_speed", 
                                    window_size=TOTAL_EPISODE_FRAMES)
        forward_speed_fig.add_graph("reference_signal", 
                                    label="reference_Signal", 
                                    window_size=TOTAL_EPISODE_FRAMES)

        # Add throttle signals graph
        throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
        throttle_fig.add_graph("throttle", 
                              label="throttle", 
                              window_size=TOTAL_EPISODE_FRAMES)
        # Add brake signals graph
        brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
        brake_fig.add_graph("brake", 
                              label="brake", 
                              window_size=TOTAL_EPISODE_FRAMES)
        # Add steering signals graph
        steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
        steer_fig.add_graph("steer", 
                              label="steer", 
                              window_size=TOTAL_EPISODE_FRAMES)

        # live plotter is disabled, hide windows
        if not enable_live_plot:
            lp_traj._root.withdraw()
            lp_1d._root.withdraw()        

        # Iterate the frames until the end of the waypoints is reached or
        # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
        # ouptuts the results to the controller output directory.
        reached_the_end = False
        skip_first_frame = True
        closest_index    = 0  # Index of waypoint that is currently closest to
                              # the car (assumed to be the first index)
        closest_distance = 0  # Closest distance of closest waypoint to car
        counter=0
        #print("dssssssssssssssssssssssssssssssssssssssssssssssssssssssss",TOTAL_EPISODE_FRAMES)
        for frame in range(TOTAL_EPISODE_FRAMES):
            # Gather current data from the CARLA server
            measurement_data, sensor_data = client.read_data()
            #print("lllllllllllllllllllllllllll",len(waypoints_np),waypoints_np[-1])
            #update_pts=list(waypoints_np[-1])
            
            
            # Update pose, timestamp
            current_x, current_y, current_yaw = \
                get_current_pose(measurement_data)
            current_speed = measurement_data.player_measurements.forward_speed
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0
            #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",current_timestamp)
            # Wait for some initial time before starting the demo
            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                counter+=1
                #flagbox=darknet_proper_fps.performDetect(res_img,thresh,configPath, weightPath, metaPath, showImage, makeImageOnly, initOnly,pointsmid)
                continue
            else:
                current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START
            #print("sdmskdkfjdkfjdjksf",counter)
            #for i in range(1,5):
            update_pts=[list(sec1),list(sec2),list(sec3),list(sec4)]
            pts_2d_ls=[]
            for i in range(len(update_pts)):
                world_coord = np.asarray(update_pts[i]).reshape(4,-1)
            # print("world_coord.shape",world_coord.shape)
            # world_coord = np.array([[250.0 ,129.0 ,38.0 ,1.0]]).reshape(4,-1)
                world_transform = Transform(
                    measurement_data.player_measurements.transform
                )

                fdfd=str(world_transform)
                sdsd=fdfd[1:-1].split('\n')
                new=[]
                for i in sdsd:
                    dd=i[:-1].lstrip('[ ')
                    ff=re.sub("\s+", ",", dd.strip())
                    gg=ff.split(',')
                    new.append([float(i) for i in gg])
                newworld=np.array(new)
        
                extrinsic = np.matmul(newworld,newcam)
                cam_coord = np.linalg.inv(extrinsic) @ world_coord
                img_coord = intrinsic @ cam_coord[:3]
                img_coord = np.array([img_coord[0]/img_coord[2],
                                img_coord[1]/img_coord[2],
                                img_coord[2]])

                if(img_coord[2] > 0):
                    x_2d = WINDOW_WIDTH- img_coord[0]
                    y_2d = WINDOW_HEIGHT - img_coord[1]
                    pts_2d_ls.append([x_2d,y_2d])

            #x_diff=(pts_2d_ls[0]-get_2d_point[1])
            #y_diff=(pts_2d_ls[1]-get_2d_point[0])
            #print("sdsdjsdsjdksjdskdjk",x_diff,y_diff,pts_2d_ls,get_2d_point)
            #get_2d_point=[pts_2d_ls[1],pts_2d_ls[0]]
            
            image_RGB = to_rgb_array(sensor_data['CameraRGB'])
            im_bgr = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)
            image_depth=to_rgb_array(sensor_data['CameraDepth'])
            counter+=1
            if counter==0:
                flagbox=False
            else:
                #pointsmid[0][1]=pointsmid[0][1]+x_diff
                #pointsmid[0][0]=pointsmid[0][0]+y_diff
                #pointsmid[1][1]=pointsmid[1][1]+x_diff
                #pointsmid[1][0]=pointsmid[1][0]+y_diff
                #pointsmid[2][1]=pointsmid[2][1]+x_diff
                #pointsmid[2][0]=pointsmid[2][0]+y_diff
                #pointsmid[3][1]=pointsmid[3][1]+x_diff
                #pointsmid[3][0]=pointsmid[3][0]+y_diff
                try:
                    pointsmid[0][1]=pts_2d_ls[0][0]
                    pointsmid[0][0]=pts_2d_ls[0][1]
                    pointsmid[1][1]=pts_2d_ls[1][0]
                    pointsmid[1][0]=pts_2d_ls[1][1]
                    pointsmid[2][1]=pts_2d_ls[2][0]
                    pointsmid[2][0]=pts_2d_ls[2][1]
                    pointsmid[3][1]=pts_2d_ls[3][0]
                    pointsmid[3][0]=pts_2d_ls[3][1]
                    disbox=False
                    flagbox=darknet_proper_fps.performDetect(im_bgr,thresh,configPath, weightPath, metaPath, showImage, makeImageOnly, initOnly,pointsmid,disbox)
                except Exception as e:
                    disbox=True
                    flagbox=darknet_proper_fps.performDetect(im_bgr,thresh,configPath, weightPath, metaPath, showImage, makeImageOnly, initOnly,pointsmid,disbox)
                    if flagbox!=False:
                        midofpts=[(pointsmid[1][1]+pointsmid[0][1])/2,(pointsmid[1][0]+pointsmid[0][0])/2]
                        depthflag=image_depth[int(flagbox[1]),int(flagbox[0])]
                        depthpts=image_depth[int(midofpts[1]),int(midofpts[0])]
                        print(depthflag,depthpts)
                        #cv2.circle(im_bgr,(int(flagbox[0]),int(flagbox[1])), 5, (255,0,255), -1)
                        #cv2.circle(im_bgr,(int(midofpts[0]),int(midofpts[1])), 5, (0,0,255), -1)
                        cv2.imwrite('./seg_out/{}_zz.jpg'.format(frame),im_bgr)
                        scalenew=depthflag[0]+depthflag[1]*256+depthflag[2]*256*256
                        scalenew=scalenew/((256*256*256) - 1 )
                        depthflag=scalenew*1000
                        scalenew=depthpts[0]+depthpts[1]*256+depthpts[2]*256*256
                        scalenew=scalenew/((256*256*256) - 1 )
                        depthpts=scalenew*1000
                        diff=abs(depthflag-depthpts)
                        print("entereeeeeeeeeeeeeeeeeeeeeeeeeeeeherree",diff)
                        if diff<10:
                            flagbox=True
                        else:
                            flagbox=False
                    print(e)
                           
            print("fffffffffffffffffff",flagbox)
            x_history.append(current_x)
            y_history.append(current_y)
            yaw_history.append(current_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp) 
            if flagbox==False:

            # Store history
                
            
            ###
            # Controller update (this uses the controller2d.py implementation)
            ###

            # To reduce the amount of waypoints sent to the controller,
            # provide a subset of waypoints that are within some 
            # lookahead distance from the closest point to the car. Provide
            # a set of waypoints behind the car as well.
            
            # Find closest waypoint index to car. First increment the index
            # from the previous index until the new distance calculations
            # are increasing. Apply the same rule decrementing the index.
            # The final index should be the closest point (it is assumed that
            # the car will always break out of instability points where there
            # are two indices with the same minimum distance, as in the
            # center of a circle)
                closest_distance = np.linalg.norm(np.array([
                    waypoints_np[closest_index, 0] - current_x,
                    waypoints_np[closest_index, 1] - current_y]))
                new_distance = closest_distance
                new_index = closest_index
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_index = new_index
                    new_index += 1
                    if new_index >= waypoints_np.shape[0]:  # End of path
                        break
                    new_distance = np.linalg.norm(np.array([
                        waypoints_np[new_index, 0] - current_x,
                        waypoints_np[new_index, 1] - current_y]))
                new_distance = closest_distance
                new_index = closest_index
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_index = new_index
                    new_index -= 1
                    if new_index < 0:  # Beginning of path
                        break
                    new_distance = np.linalg.norm(np.array([
                        waypoints_np[new_index, 0] - current_x,
                        waypoints_np[new_index, 1] - current_y]))

            # Once the closest index is found, return the path that has 1
            # waypoint behind and X waypoints ahead, where X is the index
            # that has a lookahead distance specified by 
            # INTERP_LOOKAHEAD_DISTANCE
                waypoint_subset_first_index = closest_index - 1
                if waypoint_subset_first_index < 0:
                    waypoint_subset_first_index = 0

                waypoint_subset_last_index = closest_index
                total_distance_ahead = 0
                while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
                    total_distance_ahead += wp_distance[waypoint_subset_last_index]
                    waypoint_subset_last_index += 1
                    if waypoint_subset_last_index >= waypoints_np.shape[0]:
                        waypoint_subset_last_index = waypoints_np.shape[0] - 1
                        break

            # Use the first and last waypoint subset indices into the hash
            # table to obtain the first and last indicies for the interpolated
            # list. Update the interpolated waypoints to the controller
            # for the next controller update.
                new_waypoints = \
                    wp_interp[wp_interp_hash[waypoint_subset_first_index]:\
                              wp_interp_hash[waypoint_subset_last_index] + 1]
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
                # Update live plotter with new feedback
                    trajectory_fig.roll("trajectory", current_x, current_y)
                    trajectory_fig.roll("car", current_x, current_y)
                # When plotting lookahead path, only plot a number of points
                # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
                # to decrease load when live plotting
                    new_waypoints_np = np.array(new_waypoints)
                    path_indices = np.floor(np.linspace(0, 
                                                    new_waypoints_np.shape[0]-1,
                                                    INTERP_MAX_POINTS_PLOT))
                    trajectory_fig.update("lookahead_path", 
                        new_waypoints_np[path_indices.astype(int), 0],
                        new_waypoints_np[path_indices.astype(int), 1],
                        new_colour=[0, 0.7, 0.7])
                    forward_speed_fig.roll("forward_speed", 
                                       current_timestamp, 
                                       current_speed)
                    forward_speed_fig.roll("reference_signal", 
                                       current_timestamp, 
                                       controller._desired_speed)

                    throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                    brake_fig.roll("brake", current_timestamp, cmd_brake)
                    steer_fig.roll("steer", current_timestamp, cmd_steer)

                # Refresh the live plot based on the refresh rate 
                # set by the options
                    if enable_live_plot and \
                   live_plot_timer.has_exceeded_lap_period():
                        lp_traj.refresh()
                        lp_1d.refresh()
                        live_plot_timer.lap()

            # Output controller command to CARLA server
                send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake)

            # Find if reached the end of waypoint. If the car is within
            # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
            # the simulation will end.
                dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,###########changed waypoints[-1]
                waypoints[-1][1] - current_y]))
                if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                    reached_the_end = True
                if reached_the_end:
                    break
            else:
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
        store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
        store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
        store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
        store_trajectory_plot(brake_fig.fig, 'brake_output.png')
        store_trajectory_plot(steer_fig.fig, 'steer_output.png')
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

