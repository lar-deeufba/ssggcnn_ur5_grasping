#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <pcl/visualization/cloud_viewer.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/filters/passthrough.h>
#include <geometry_msgs/Point.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

geometry_msgs::Point p1,p2;
pcl::PointCloud<pcl::PointXYZRGB> cloud_plc;
bool cloud_filled=false;
bool cam_info_taken=false;
sensor_msgs::CameraInfo cam_info;

sensor_msgs::Image point_to_depth(pcl::PointCloud<pcl::PointXYZRGB>::Ptr msg){
	if (cam_info_taken == true){
	//sensor_msgs::CameraInfo info = cam_info;
	float centre_x = cam_info.K[2];
	float centre_y = cam_info.K[5];
	float focal_x = cam_info.K[0];
	float focal_y = cam_info.K[4];

	cv::Mat cv_image = cv::Mat(cam_info.height, cam_info.width, CV_32FC1,cv::Scalar(std::numeric_limits<float>::max()));

	for (int i=0; i<msg->points.size();i++){
		if (msg->points[i].z == msg->points[i].z){
			float z = msg->points[i].z*1000.0;
			float u = (msg->points[i].x*1000.0*focal_x) / z;
			float v = (msg->points[i].y*1000.0*focal_y) / z;
			int pixel_pos_x = (int)(u + centre_x);
			int pixel_pos_y = (int)(v + centre_y);

		if (pixel_pos_x > (cam_info.width-1)){
			pixel_pos_x = cam_info.width -1;
		}
		if (pixel_pos_y > (cam_info.height-1)){
			pixel_pos_y = cam_info.height-1;
		}
			cv_image.at<float>(pixel_pos_y,pixel_pos_x) = z;
		}       
	}

	cv_image.convertTo(cv_image,CV_16UC1);
	sensor_msgs::Image output_image;
	cv_bridge::CvImage img_bridge;
	img_bridge= cv_bridge::CvImage(std_msgs::Header(), "16UC1", cv_image);
	img_bridge.toImageMsg(output_image);
	output_image.header = cam_info.header;
	output_image.header.stamp = cam_info.header.stamp;
	return output_image;
	
	}
}


void set_info(const sensor_msgs::CameraInfo::ConstPtr& msg){
	cam_info.height=msg->height;
	cam_info.width=msg->width;
	cam_info.distortion_model=msg->distortion_model;
	for (int i=0; i < 9; i++){
		cam_info.K[i]=msg->K[i];
		cam_info.R[i]=msg->R[i];
	}
	for (int i=0; i < 12; i++){
		cam_info.P[i]=msg->P[i];
	}
	cam_info.binning_x=msg->binning_x;
	cam_info.binning_y=msg->binning_y;
	cam_info.roi=msg->roi;
	cam_info_taken=true;
}

void pointCall(const geometry_msgs::Point::ConstPtr& msg){
p1.x=msg->x;
p1.y=msg->y;

}

void pointCall2(const geometry_msgs::Point::ConstPtr& msg){
p2.x=msg->x;
p2.y=msg->y;

}

void set_point(const sensor_msgs::PointCloud2Ptr& msg){
    pcl::fromROSMsg(*msg, cloud_plc);
    cloud_filled=true;
}

int main (int argc, char** argv)
{


	ros::init(argc, argv, "talker");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("/point1", 1000, pointCall);
	ros::Subscriber sub2 = n.subscribe("/point2", 1000, pointCall2);
	ros::Subscriber sub3 = n.subscribe("/camera/depth/points", 1000, set_point);
	ros::Subscriber sub4 = n.subscribe("/camera/depth/camera_info", 1000, set_info);
	sensor_msgs::PointCloud2 final_cloud;
	sensor_msgs::Image image;
	ros::Publisher pub = n.advertise<sensor_msgs::PointCloud2> ("obj_cloud",10);
	ros::Publisher pub2 = n.advertise<sensor_msgs::Image> ("obj_img",10);
	bool is_valid = false;

	
	pcl::visualization::CloudViewer viewer ("Obj Cloud");
	//pcl::visualization::CloudViewer viewer2 ("Object Cloud");

	float maxX,minX,minY,maxY;
	/*
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> ("/home/daniel/catkin_ws/src/pcl_sim/cloud.pcd", *cloud) == -1) //* load the file
	{
	PCL_ERROR ("Couldn't read file cloud.pcd \n");
	return (-1);
	}else
		is_valid=true;
	std::cout << "Cheguei aqui" << endl;
	*/
	ros::Rate loop_rate(10);
	while (n.ok()){
		/*
		std::cout << "Loaded "
		    << cloud->width * cloud->height
		    << " data points from test_pcd.pcd with the following fields: "
		    << std::endl;
		*/
		if (p1.x !=0 && p2.x != 0 && cloud_filled ){
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
			//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
			cloud=cloud_plc.makeShared();
			//cout << cloud_plc.height << endl << cloud_plc.width << endl;
			geometry_msgs::Point pd,pd2;
			pd.x= cloud->at(p1.x,p1.y).x;
			pd.y= cloud->at(p1.x,p1.y).y;
			pd2.x= cloud->at(p2.x,p2.y).x;
			pd2.y= cloud->at(p2.x,p2.y).y;
			if (isnan(pd.x) || isnan(pd2.x) || isnan(pd.y) || isnan(pd2.y)){
				ROS_ERROR("NaN Found");
			}else{
				//std::cout << pd << std::endl << pd2 << std::endl;
				if (pd.x > pd2.x){
					maxX=pd.x;
					minX=pd2.x;
				}else{
					maxX=pd2.x;
					minX=pd.x;
				}

				if (pd.y > pd2.y){
					maxY=pd.y;
					minY=pd2.y;
				}else{
					maxY=pd2.y;
					minY=pd.y;
				}
				//cout << "MinY: " << minY << endl << "MaxY: " << maxY << endl;
		    		//cout << "MinX: " << minX << endl << "MaxX: " << maxX << endl;


				//Limita Y
				pcl::PassThrough<pcl::PointXYZRGB> pass;
				pass.setInputCloud (cloud);
				pass.setFilterFieldName ("y");
				pass.setFilterLimits (minY, maxY);
				pass.setKeepOrganized(true);
				//pass.setFilterLimitsNegative (true);
				pass.filter (*cloud);

				//Limita X
				pcl::PassThrough<pcl::PointXYZRGB> pass2;
				pass2.setInputCloud (cloud);
				pass2.setFilterFieldName ("x");
				pass2.setFilterLimits (minX, maxX);
				pass2.setKeepOrganized(true);
				//pass.setFilterLimitsNegative (true);
				pass2.filter (*cloud);

				//Limita Z
				pcl::PassThrough<pcl::PointXYZRGB> pass3;
				pass3.setInputCloud (cloud);
				pass3.setFilterFieldName ("z");
				pass3.setFilterLimits (0, 0.55);
				pass3.setKeepOrganized(true);
				//pass.setFilterLimitsNegative (true);
				pass3.filter (*cloud);
				/*
				pcl::PointXYZRGB basic_point;
				basic_point.x = pd.x;
	      			basic_point.y = pd.y;
	      			basic_point.z = 0.5;
				basic_point.r = 255;
				basic_point.g = 0;
				basic_point.b = 0;
				
				pcl::PointXYZRGB basic_point2;
				basic_point2.x = pd2.x;
	      			basic_point2.y = pd2.y;
	      			basic_point2.z = 0.5;
				basic_point2.r = 255;
				basic_point2.g = 0;
				basic_point2.b = 0;

				pcl::PointXYZRGB basic_point3;
				basic_point3.x = pd.x;
	      			basic_point3.y = pd2.y;
	      			basic_point3.z = 0.5;
				basic_point3.r = 255;
				basic_point3.g = 0;
				basic_point3.b = 0;

				pcl::PointXYZRGB basic_point4;
				basic_point4.x = pd2.x;
	      			basic_point4.y = pd.y;
	      			basic_point4.z = 0.5;
				basic_point4.r = 255;
				basic_point4.g = 0;
				basic_point4.b = 0;
			

				cloud->points.push_back(basic_point);
				cloud->points.push_back(basic_point2);
				cloud->points.push_back(basic_point3);
				cloud->points.push_back(basic_point4);
				*/
				for (std::size_t i = 0; i < cloud->points.size (); ++i){
					if (isnan(cloud->points[i].x)){
						cloud->points[i].r = 0;
						cloud->points[i].g = 0;
						cloud->points[i].b = 0;
					}
				}
				viewer.showCloud (cloud);
				image=point_to_depth(cloud);
				//pcl::toROSMsg (*cloud, image);
				pcl::toROSMsg(*cloud,final_cloud);
				pub.publish(final_cloud);
				pub2.publish(image);
			}

		}
		//std::cout << p1 << std::endl << p2 << std::endl;
		//viewer.showCloud (cloud);
		ros::spinOnce();
	}
	return (0);
}
