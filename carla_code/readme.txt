controller2d.py - has the entire control algo to take the car to a given destination

cutils.py  needed by controller and module_7.py

get_image_and_save.py is for automatically going around,saving the images,3d point,transforms in any .txt file specified in the cmd line.

module_7.py- traverse the car from point A to B using the controller2d script and the waypoints provided in racetract_waypoints.txt


pipeline.py- is to call the segmentation function and convert the returned 2d point to 3d point,and use this as the destination point for module_7.py and gen_ways.py

segmentation.py -segment the given carla image and get the mid point.

gen_ways.py- generate the waypoints between the starting vehicle location and the goal point,generated from pipeline.py

module_7_pipeline.py - combination of module_7.py,pipeline.py and gen_ways.py.

segmentationobj.py - is just like segmentation.py but it returns a few extra parameters
darknet_proper_fps.py- is the object detection code,(yolov4)
module_7_pipeline_obj_proper_track_copy.py - is to segment,and detect obstacle (covering a few cases) and to either stop the vehicle or go to destination.
This script calls segmentationobj.py and darknet_proper_fps.py.


The module_7_pipeline_obj_proper_track_copy.py is cleaned up and uploaded as two files-
main_module_v4.py and module_7_clean_v4_final.py
To execute the basic flow of segmenting the road and either braking(if obstacle is persent) or navigating to the goal point,run:
python3 main_module_v4.py




