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
import random
import glob
import cv2
import re
import matplotlib.pyplot as plt

import live_plotter as lv   # Custom live plotting library
import controller2d
import configparser 
import segmentationobj
import darknet_proper_fps

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
thresh= 0.55 #object detection threshold
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

    settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=65,
            NumberOfPedestrians=10,
            WeatherId=1)
    #settings.randomize_seeds()#uncomment to get different random settings
    #initialize depth and rgb camera
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

def convert_transform(trans_forms):
            wt_str=str(trans_forms)
            wt_trunc=wt_str[1:-1].split('\n')
            new=[]
            for i in wt_trunc:
                    wt_str1=i[:-1].lstrip('[ ')
                    wt_str2=re.sub("\s+", ",", wt_str1.strip())
                    gg=wt_str2.split(',')
                    new.append([float(i) for i in gg])
            return new

def _2d_to_3d_roi(_2dpoint,image_depth,extrinsic):
            depth=image_depth[int(_2dpoint[0]),int(_2dpoint[1])]
            scale=depth[0]+depth[1]*256+depth[2]*256*256
            scale=scale/((256*256*256) - 1 )
            depth=scale*1000
            pos2d=np.array([(WINDOW_WIDTH-_2dpoint[1])*depth,(WINDOW_HEIGHT-_2dpoint[0])*depth,depth])
            _3d_point=np.dot(np.linalg.inv(intrinsic),pos2d)
            _3d_point=np.append(_3d_point,1)
            new_3dpoint=np.matmul((extrinsic),_3d_point)
            return new_3dpoint


def get_discrete_waypts(ini_location,sec):

    ini_location[2]=4.5
    goal=list(sec)
    goal[2]=1.08
    aa=ini_location[0]
    dec=abs(ini_location[1]-goal[1])/abs(aa-(goal[0]))
    f1=open(WAYPOINTS_FILENAME,'a+')
    for i in range(int(goal[0]),int(aa)):

        if abs((int(aa)-i))<10:
    
            ini_location=[ini_location[0]-1,ini_location[1]+dec,ini_location[2]-0.03]
        else:
            ini_location=[ini_location[0]-1,ini_location[1]+dec,ini_location[2]]

        f1.write(str(ini_location[0])+','+str(ini_location[1])+','+str(ini_location[2])+'\n')


    waypoints_file = WAYPOINTS_FILENAME
    waypoints_np   = None
    with open(waypoints_file) as waypoints_file_handle:
        waypoints = list(csv.reader(waypoints_file_handle, 
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
    return waypoints


def subset_waypoints(closest_index,current_x,current_y,waypoints_np,wp_interp_hash,wp_distance,wp_interp):

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
    return new_waypoints

def dis_to_continous(waypoints_np):

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
    return waypoints_np,wp_interp_hash,wp_distance,wp_interp


def _3d_to_2d(extrinsic,world_coord,intrinsic):
    
    cam_coord = np.linalg.inv(extrinsic) @ world_coord
    img_coord = intrinsic @ cam_coord[:3]
    img_coord = np.array([img_coord[0]/img_coord[2],
                    img_coord[1]/img_coord[2],
                    img_coord[2]])

    if(img_coord[2] > 0):
        x_2d = WINDOW_WIDTH- img_coord[0]
        y_2d = WINDOW_HEIGHT - img_coord[1]
        if (x_2d>=0 and x_2d<=1280) and (y_2d>=0 and y_2d<=720):
            return [x_2d,y_2d]
        else:
            return []
    else:
        return []


def total_frames(client,sim_start_stamp):
    # Ensure at least one frame is used to compute average timestep
    num_iterations = ITER_FOR_SIM_TIMESTEP
    if (ITER_FOR_SIM_TIMESTEP < 1):
        num_iterations = 1

    # Gather current data from the CARLA server. This is used to get the
    # simulator starting game time. Note that we also need to
    # send a command back to the CARLA server because synchronous mode
    # is enabled.
    
    
    # Send a control command to proceed to next iteration.
    
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

    return int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                        SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

class live_all:
    # FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
    # FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
    # PLOT_LEFT          = 0.1    # in fractions of figure width and height
    # PLOT_BOT           = 0.1    
    # PLOT_WIDTH         = 0.8
    # PLOT_HEIGHT        = 0.8
    def __init__(self,waypoints_np,start_x,start_y,TOTAL_EPISODE_FRAMES,INTERP_MAX_POINTS_PLOT,enable_live_plot):
        self.waypoints_np=waypoints_np
        self.start_y=start_y
        self.start_x=start_x
        self.TOTAL_EPISODE_FRAMES=TOTAL_EPISODE_FRAMES
        self.INTERP_MAX_POINTS_PLOT=INTERP_MAX_POINTS_PLOT
        self.FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
        self.FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
        self.PLOT_LEFT          = 0.1    # in fractions of figure width and height
        self.PLOT_BOT           = 0.1    
        self.PLOT_WIDTH         = 0.8
        self.PLOT_HEIGHT        = 0.8

        self.lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        self.lp_1d = lv.LivePlotter(tk_title="Controls Feedback")
        
        ###
        # Add 2D position / trajectory plot
        ###
        self.trajectory_fig = self.lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(self.FIGSIZE_X_INCHES, self.FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[self.PLOT_LEFT, self.PLOT_BOT, self.PLOT_WIDTH, self.PLOT_HEIGHT])

        self.trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed 
                                        # coordinate system the X
                                        # axis in the graph is flipped
        self.trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

        # Add waypoint markers
        self.trajectory_fig.add_graph("waypoints", window_size=self.waypoints_np.shape[0],
                                x0=self.waypoints_np[:,0], y0=self.waypoints_np[:,1],
                                linestyle="-", marker="", color='g')
        # Add trajectory markers
        self.trajectory_fig.add_graph("trajectory", window_size=self.TOTAL_EPISODE_FRAMES,
                                x0=[self.start_x]*self.TOTAL_EPISODE_FRAMES, 
                                y0=[self.start_y]*self.TOTAL_EPISODE_FRAMES,
                                color=[1, 0.5, 0])
        # Add lookahead path
        self.trajectory_fig.add_graph("lookahead_path", 
                                window_size=self.INTERP_MAX_POINTS_PLOT,
                                x0=[start_x]*self.INTERP_MAX_POINTS_PLOT, 
                                y0=[start_y]*self.INTERP_MAX_POINTS_PLOT,
                                color=[0, 0.7, 0.7],
                                linewidth=4)
        # Add starting position marker
        self.trajectory_fig.add_graph("start_pos", window_size=1, 
                                x0=[self.start_x], y0=[self.start_y],
                                marker=11, color=[1, 0.5, 0], 
                                markertext="Start", marker_text_offset=1)
        # Add end position marker
        self.trajectory_fig.add_graph("end_pos", window_size=1, 
                                x0=[self.waypoints_np[-1, 0]], 
                                y0=[self.waypoints_np[-1, 1]],
                                marker="D", color='r', 
                                markertext="End", marker_text_offset=1)
        # Add car marker
        self.trajectory_fig.add_graph("car", window_size=1, 
                                marker="s", color='b', markertext="Car",
                                marker_text_offset=1)

        ###
        # Add 1D speed profile updater
        ###
        self.forward_speed_fig =\
                self.lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        self.forward_speed_fig.add_graph("forward_speed", 
                                    label="forward_speed", 
                                    window_size=self.TOTAL_EPISODE_FRAMES)
        self.forward_speed_fig.add_graph("reference_signal", 
                                    label="reference_Signal", 
                                    window_size=self.TOTAL_EPISODE_FRAMES)

        # Add throttle signals graph
        self.throttle_fig = self.lp_1d.plot_new_dynamic_figure(title="Throttle")
        self.throttle_fig.add_graph("throttle", 
                            label="throttle", 
                            window_size=self.TOTAL_EPISODE_FRAMES)
        # Add brake signals graph
        self.brake_fig = self.lp_1d.plot_new_dynamic_figure(title="Brake")
        self.brake_fig.add_graph("brake", 
                            label="brake", 
                            window_size=self.TOTAL_EPISODE_FRAMES)
        # Add steering signals graph
        self.steer_fig = self.lp_1d.plot_new_dynamic_figure(title="Steer")
        self.steer_fig.add_graph("steer", 
                            label="steer", 
                            window_size=self.TOTAL_EPISODE_FRAMES)

        # live plotter is disabled, hide windows
        if not enable_live_plot:
            self.lp_traj._root.withdraw()
            self.lp_1d._root.withdraw()   

def get_extrinsic(measurement_data,camera2):

    camera_to_car_transform = camera2.get_unreal_transform()
    world_transform = Transform(
                measurement_data.player_measurements.transform
            )        
    
    newworld=np.array(convert_transform(world_transform))
    newcam=np.array(convert_transform(camera_to_car_transform))
    return np.matmul(newworld,newcam),newcam

def chk_diff(pointsmid,flagbox,image_depth):


    midofpts=[(pointsmid[1][1]+pointsmid[0][1])/2,(pointsmid[1][0]+pointsmid[0][0])/2]
    depthflag=image_depth[int(flagbox[1]),int(flagbox[0])]
    depthpts=image_depth[int(midofpts[1]),int(midofpts[0])]
    
    #cv2.imwrite('./seg_out/{}_zz.jpg'.format(frame),im_bgr)
    scalenew=depthflag[0]+depthflag[1]*256+depthflag[2]*256*256
    scalenew=scalenew/((256*256*256) - 1 )
    depthflag=scalenew*1000
    scalenew=depthpts[0]+depthpts[1]*256+depthpts[2]*256*256
    scalenew=scalenew/((256*256*256) - 1 )
    depthpts=scalenew*1000
    
    return abs(depthflag-depthpts)

def find_pts(update_pts,measurement_data,newcam):

    pts_2d_ls_new=[]
    for i in range(len(update_pts)):
        
        world_transform = Transform(
            measurement_data.player_measurements.transform
        )

        newworld=np.array(convert_transform(world_transform))

        extrinsic = np.matmul(newworld,newcam)

        world_coord = np.asarray(update_pts[i]).reshape(4,-1)
        conversion=_3d_to_2d(extrinsic,world_coord,intrinsic)
        pts_2d_ls_new.append(conversion)
    return pts_2d_ls_new


def chk_obj(im_bgr,pts_2d_ls,image_depth,pointsmid):

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
            diff=chk_diff(pointsmid,flagbox,image_depth)
            print(diff)
            if diff<10:
                flagbox=True
            else:
                flagbox=False
        print("debug",e)
    return flagbox

def plot_live_plot(skip_first_frame,frame,liveobj,live_plot_timer,current_x,current_y,new_waypoints,current_timestamp,current_speed,controller,cmd_throttle, cmd_steer, cmd_brake,enable_live_plot):
# Update live plotter with new feedback
    liveobj.trajectory_fig.roll("trajectory", current_x, current_y)
    liveobj.trajectory_fig.roll("car", current_x, current_y)
# When plotting lookahead path, only plot a number of points
# (INTERP_MAX_POINTS_PLOT amount of points). This is meant
# to decrease load when live plotting
    new_waypoints_np = np.array(new_waypoints)
    path_indices = np.floor(np.linspace(0, 
                                    new_waypoints_np.shape[0]-1,
                                    INTERP_MAX_POINTS_PLOT))
    liveobj.trajectory_fig.update("lookahead_path", 
        new_waypoints_np[path_indices.astype(int), 0],
        new_waypoints_np[path_indices.astype(int), 1],
        new_colour=[0, 0.7, 0.7])
    liveobj.forward_speed_fig.roll("forward_speed", 
                    current_timestamp, 
                    current_speed)
    liveobj.forward_speed_fig.roll("reference_signal", 
                    current_timestamp, 
                    controller._desired_speed)

    liveobj.throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
    liveobj.brake_fig.roll("brake", current_timestamp, cmd_brake)
    liveobj.steer_fig.roll("steer", current_timestamp, cmd_steer)

# Refresh the live plot based on the refresh rate 
# set by the options
    if enable_live_plot and \
live_plot_timer.has_exceeded_lap_period():
        liveobj.lp_traj.refresh()
        liveobj.lp_1d.refresh()
        live_plot_timer.lap()
