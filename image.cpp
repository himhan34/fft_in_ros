#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;

public:
  ImageConverter()
    : it_(nh_)
  {
    // 이미지 토픽을 구독합니다.
    image_sub_ = it_.subscribe("/c2i_intensity_image", 10, &ImageConverter::imageCb, this);
    cv::namedWindow("Image window", cv::WINDOW_AUTOSIZE);
  }

  ~ImageConverter()
  {
    cv::destroyWindow("Image window");
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      // sensor_msgs/Image를 OpenCV 형식으로 변환합니다.
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
      if (msg->encoding == "16UC1") {
        // 이미지 처리 예: 정규화 및 컬러맵 적용
        cv::normalize(cv_ptr->image, cv_ptr->image, 0, 65535, cv::NORM_MINMAX);
        cv_ptr->image.convertTo(cv_ptr->image, CV_8U, 1.0 / 256.0);
        cv::applyColorMap(cv_ptr->image, cv_ptr->image, cv::COLORMAP_JET);
      } else if (msg->encoding == "32FC1") {
        // 이미지 처리 예: 절대값 취하고 0-255로 스케일링
        cv_ptr->image.convertTo(cv_ptr->image, CV_8U, 255.0 / 65535.0);
      }

      // OpenCV 윈도우에 이미지를 표시합니다.
      cv::imshow("Image window", cv_ptr->image);
      cv::waitKey(3);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
