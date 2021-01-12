controller2d.py - has the entire control algo to take the car to a given destination

cutils.py  needed by controller and module_7.py

get_image_and_save.py is for automatically going around,saving the images,3d point,transforms in any .txt file specified in the cmd line.

module_7.py- traverse the car from point A to B using the controller2d script and the waypoints provided in racetract_waypoints.txt


pipeline.py- is to call the segmentation function and convert the returned 2d point to 3d point,and use this as the destination point for module_7.py and gen_ways.py

segmentation.py -segment the given carla image and get the mid point.

gen_ways.py- generate the waypoints between the starting vehicle location and the goal point,generated from pipeline.py

module_7_pipeline.py - combination of module_7.py,pipeline.py and gen_ways.py.






