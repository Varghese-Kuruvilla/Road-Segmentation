#Generate waypoints and velocity profile for dubins curves
import dubins
import rospy
import numpy as np 
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import sys 
#Utils
def breakpoint():
  inp = input("Waiting for input...")


class pathplan:

    """ Description
    :raises:

    :rtype:
    """
    def __init__(self):
        self.waypoints = None 
        self.dist = []
        self.vel_profile = []
        self.step_size = 0.3
        self.odom_dist = None
        #Init ros node
        rospy.init_node('path_plan',anonymous=True)
        self.vel_publisher = rospy.Publisher('/nav/cmd_vel',Twist,queue_size=1000)

    def gen_waypoints(self,q0,q1,r):
    
        """ Description
        :type self:
        :param self:
    
        :type q0:
        :param q0:
    
        :type q1:
        :param q1:
    
        :type r:
        :param r:
    
        :raises:
    
        :rtype:
        """
        self.waypoints, _ = dubins.path_sample(q0, q1, r, step_size=self.step_size)
        # print("waypoints",self.waypoints)
        # breakpoint()

    def gen_velocity(self):
    
        """ Description
        :type self:
        :param self:
    
        :raises:
    
        :rtype:
        """
        for i in range(0,len(self.waypoints)-1):
            self.dist.append(np.linalg.norm(np.asarray(self.waypoints[i])\
                                -np.asarray(self.waypoints[i+1])))
            
            ang_vel = (self.waypoints[i+1][2] - self.waypoints[i][2])/(0.3)
            self.vel_profile.append([1.0,0.0,0.0,0.0,0.0,ang_vel])
        
        print("Velocity profile",self.vel_profile)
        print("self.dist",self.dist)

    def callback_odom(self,data):
        #Calculate the distance travelled by the robot
        x_coord = data.pose.pose.position.x
        y_coord = data.pose.pose.position.y
        self.odom_dist = np.linalg.norm(np.asarray(x_coord)\
                                -np.asarray(y_coord))

    def create_pub_msg(self,i,custom_vel=None):
    
        """ Description
        :type self:
        :param self:
    
        :raises:
    
        :rtype:
        """
        vel_msg = Twist()
        if(custom_vel == None):
            vel_msg.linear.x = self.vel_profile[i][0]
            vel_msg.linear.y = self.vel_profile[i][1]
            vel_msg.linear.z = self.vel_profile[i][2]
            vel_msg.angular.x = self.vel_profile[i][3]
            vel_msg.angular.y = self.vel_profile[i][4]
            vel_msg.angular.z = self.vel_profile[i][5]
        else:
            vel_msg.linear.x =  custom_vel[0]
            vel_msg.linear.y =  custom_vel[1]
            vel_msg.linear.z =  custom_vel[2]
            vel_msg.angular.x = custom_vel[3]
            vel_msg.angular.y = custom_vel[4]
            vel_msg.angular.z = custom_vel[5]

        return vel_msg    

    def publish_velocity(self):
    
        """ Description
        :raises:
    
        :rtype:
        """
        i=0
        print("len(self.vel_profile)",len(self.vel_profile))
        print("self.odom_dist",self.odom_dist)
        while not rospy.is_shutdown():
            rospy.Subscriber('/odometry/filtered',Odometry,self.callback_odom) 
            if(self.odom_dist != None):   
                if((i+1)*self.step_size > self.odom_dist):
                    self.vel_publisher.publish(self.create_pub_msg(i))
                else:
                    i += 1
                    print("i",i)
                
                if i == len(self.vel_profile):
                    self.vel_publisher.publish(self.create_pub_msg(i,[0,0,0,0,0,0,0]))
                    break

        sys.exit(0)





if __name__ == '__main__':
    
    pathplan_obj = pathplan()
    pathplan_obj.gen_waypoints(q0=(0,0,0),q1=(7.92,0.77,0.0),r=0.5) #Actual turning radius = 4.5  
    # pathplan_obj.gen_waypoints(q0=(0,0,0),q1=(1.92,0.77,0.0),r=0.5) #Actual turning radius = 4.5  
    pathplan_obj.gen_velocity()
    pathplan_obj.publish_velocity()