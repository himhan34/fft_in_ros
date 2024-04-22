#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fftw3.h>
#include <nav_msgs/Odometry.h>  // Odometry 메시지를 위한 헤더 추가
#include <tf/tf.h>        

class ImageConverter
{
private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  fftw_complex *in1 = nullptr, *in2 = nullptr, *out1 = nullptr, *out2 = nullptr;
  fftw_plan p1, p2, p3;
  bool first_image_received_ = false;
  int rows_ = 0, cols_ = 0;

  ros::Publisher odom_pub_;
  double current_x_ = 0.0;
  double current_y_ = 0.0;
  double current_theta_ = 0.0;  // 로봇의 현재 방향(라디안)

public:
  ImageConverter()
    : it_(nh_)
  {
    image_sub_ = it_.subscribe("/c2i_intensity_image", 10, &ImageConverter::imageCb, this);
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>("odom2", 10);
  }

  ~ImageConverter()
  {
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p3);
    fftw_free(in1);
    fftw_free(in2);
    fftw_free(out1);
    fftw_free(out2);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
      cv::Mat croppedImage = cropImageByAngle(cv_ptr->image, -180, 180, 360);
      if (!croppedImage.empty()) {
          processImage(croppedImage);
      }
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  cv::Mat cropImageByAngle(const cv::Mat& img, double start_angle, double end_angle, int total_angle_range)
  {
    if (img.empty()) return cv::Mat();

    int original_width = img.cols;
    double pixels_per_degree = original_width / double(total_angle_range);

    int start_pixel = std::round((start_angle + 180) * pixels_per_degree);
    int end_pixel = std::round((end_angle + 180) * pixels_per_degree);

    if (start_pixel < 0 || end_pixel > img.cols) {
      ROS_ERROR("Crop range out of bounds.");
      return cv::Mat();
    }

    int x_start = start_pixel;
    int crop_width = end_pixel - start_pixel;
    int y_start = 0;
    int crop_height = img.rows;

    cv::Rect roi(x_start, y_start, crop_width, crop_height);
    return img(roi);
  }

  void processImage(const cv::Mat& img)
  {
    if (!first_image_received_) {
        // Allocate memory first time and initialize FFT plans
        rows_ = img.rows;
        cols_ = img.cols;
        in1 = fftw_alloc_complex(rows_ * cols_);
        in2 = fftw_alloc_complex(rows_ * cols_);
        out1 = fftw_alloc_complex(rows_ * cols_);
        out2 = fftw_alloc_complex(rows_ * cols_);
        p1 = fftw_plan_dft_2d(rows_, cols_, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
        p2 = fftw_plan_dft_2d(rows_, cols_, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
        p3 = fftw_plan_dft_2d(rows_, cols_, out2, in2, FFTW_BACKWARD, FFTW_ESTIMATE);

        fillInput(img, in1);
        fftw_execute(p1);
        first_image_received_ = true;
    } else {
        fillInput(img, in2);
        fftw_execute(p2);
        phaseCorrelation();
        fftw_execute(p3);
        normalizeAndDisplay(in2);
    }
  }

  void fillInput(const cv::Mat& img, fftw_complex* input)
  {
    if (!input) return;  // Check if the input array is allocated
    int index = 0;
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        input[index][0] = (double)img.at<uint16_t>(i, j);
        input[index][1] = 0.0;
        index++;
      }
    }
  }

  void phaseCorrelation()
  {
    if (!out1 || !out2) return; // Ensure outputs are valid
    for (int i = 0; i < rows_ * cols_; ++i) {
      double mag1 = sqrt(out1[i][0] * out1[i][0] + out1[i][1] * out1[i][1]);
      double mag2 = sqrt(out2[i][0] * out2[i][0] + out2[i][1] * out2[i][1]);
      if (mag1 > 0.0 && mag2 > 0.0) {
        double phase1 = atan2(out1[i][1], out1[i][0]);
        double phase2 = atan2(out2[i][1], out2[i][0]);
        out2[i][0] = cos(phase2 - phase1);
        out2[i][1] = sin(phase2 - phase1);
      } else {
        out2[i][0] = out2[i][1] = 0.0;
      }
    }
  }

    void updateOdometry(const cv::Point& displacement) {
        double dx = displacement.x; // 이 예제에서는 픽셀 단위의 변위를 직접 사용
        double dy = displacement.y;

        current_x_ += dx;
        current_y_ += dy;

        nav_msgs::Odometry odom;
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";

        odom.pose.pose.position.x = current_x_;
        odom.pose.pose.position.y = current_y_;
        odom.pose.pose.orientation = tf::createQuaternionMsgFromYaw(current_theta_);

        odom_pub_.publish(odom);
    }


  void normalizeAndDisplay(fftw_complex* output)
  {
    if (!output) return; // Check output before use
    // 윈도우 생성 및 크기 조정 가능하게 설정
    cv::namedWindow("POC Result", cv::WINDOW_NORMAL);
    cv::resizeWindow("POC Result", 870, 16);  // 윈도우 크기 조정

    // cv::Mat 객체의 생성 및 크기 조정을 확인
    cv::Mat result(rows_, cols_, CV_8UC1);
    cv::Mat tempResult(rows_, cols_, CV_32F);

    // fftw_complex 배열을 사용하여 값을 계산
    for (int i = 0; i < rows_ * cols_; ++i) {
        double val = sqrt(output[i][0] * output[i][0] + output[i][1] * output[i][1]);
        tempResult.at<float>(i / cols_, i % cols_) = val;
    }

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(tempResult, &minVal, &maxVal, &minLoc, &maxLoc);
    tempResult.convertTo(result, CV_8UC1, 255.0 / maxVal);
    cv::circle(result, maxLoc, 5, cv::Scalar(255), 2);

    cv::imshow("POC Result", result);
    cv::waitKey(3);

    ROS_INFO("Peak correlation at (%d, %d) with value %.2f", maxLoc.x, maxLoc.y, maxVal);

    // 오도메트리 업데이트 호출
    updateOdometry(maxLoc);
  }

};



int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
