#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<libheif/heif.h>    
#include<iostream>
#include<opencv2/imgproc.hpp>
using namespace cv;

void my_callback(int state, void* userdata) {
  if (state == -1) {
    std::cout << "Button clicked!" << std::endl;
  }
}

int main() {
  // Create a window.
  cv::Mat img = cv::imread("./test.webp");
  cv::resize(img, img, cv::Size(400,400));
  namedWindow("My Window");

  // Create a button.
  int button_id = createButton("My Button", my_callback, NULL, QT_PUSH_BUTTON);

  // Show the window.
  imshow("My Window", img);

  // Wait for a key press.
  waitKey(0);

  // Destroy the window.
  destroyWindow("My Window");

  return 0;
}
