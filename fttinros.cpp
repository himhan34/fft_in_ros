#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fftw3.h>

class ImageConverter
{
private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  fftw_complex *in1, *in2, *out1, *out2;
  fftw_plan p1, p2, p3;
  bool first_image_received_;
  int rows_, cols_;

public:
  ImageConverter()
    : it_(nh_), first_image_received_(false), rows_(0), cols_(0)
  {
    image_sub_ = it_.subscribe("/c2i_intensity_image", 10, &ImageConverter::imageCb, this);
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
      if (!first_image_received_) {
        rows_ = cv_ptr->image.rows;
        cols_ = cv_ptr->image.cols;
        in1 = fftw_alloc_complex(rows_ * cols_);
        in2 = fftw_alloc_complex(rows_ * cols_);
        out1 = fftw_alloc_complex(rows_ * cols_);
        out2 = fftw_alloc_complex(rows_ * cols_);
        p1 = fftw_plan_dft_2d(rows_, cols_, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
        p2 = fftw_plan_dft_2d(rows_, cols_, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
        p3 = fftw_plan_dft_2d(rows_, cols_, out2, in2, FFTW_BACKWARD, FFTW_ESTIMATE);
        fillInput(cv_ptr->image, in1);
        fftw_execute(p1);
        for (int i = 0; i < rows_ * cols_; ++i) {
          in2[i][0] = out1[i][0];
          in2[i][1] = out1[i][1];
        }
        first_image_received_ = true;
      } else {
        fillInput(cv_ptr->image, in2);
        fftw_execute(p2);
        phaseCorrelation();
        fftw_execute(p3);
        normalizeAndDisplay(in2);
      }
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }

  void fillInput(const cv::Mat& img, fftw_complex* input)
  {
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

    void normalizeAndDisplay(fftw_complex* output)
    {
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
    }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
