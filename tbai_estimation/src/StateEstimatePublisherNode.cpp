#include <ros/ros.h>
#include <tbai_estimation/StateEstimatePublisher.h>

int main(int argc, char** argv )
{
    ros::init(argc, argv, "state_estimation_node");

    ros::NodeHandle nh("");
    ros::NodeHandle nh_private("~");

    tbai::estimation::StateEstimatePublisher state_estimation(nh,nh_private);

    
    ros::spin();
    return 0;
}