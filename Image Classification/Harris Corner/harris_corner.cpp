/*
 * Danen Van De Ven
 * 100820351
 * COMP 4102 - Assignment 2
 * March 1, 2016
 */

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include <iostream>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 5
#define K 0.04

int main(int argc, char* argv[]) {
    char* filename;

    int block_size = BLOCK_SIZE,
        scale = 1,
        delta = 0,
        ddepth = CV_32F;

    float k = K,
          threshold = 50;

    Mat src, gray_src,
        Ix, Iy,
        Ixx, Iyy, Ixy,
        det, trace, response, Ixy_2;

    // Check for the input image.
    filename = (argc == 2) ? argv[1] : (char*) "checker.jpg";

    src = imread(filename);

    cvtColor(src, gray_src, CV_RGB2GRAY);

    Sobel(gray_src, Ix, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(gray_src, Iy, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    multiply(Ix, Ix, Ixx);
    multiply(Iy, Iy, Iyy);
    multiply(Ix, Iy, Ixy);

    // Structure tensor of Ixx, Iyy, Ixy.
    // https://en.wikipedia.org/wiki/Structure_tensor
    for (int i = 0; i < Ixx.rows - block_size; i++) {
        for (int j = 0; j < Ixx.cols - block_size; j++) {
            for (int offset_y = i; offset_y < block_size; offset_y++) {
                for (int offset_x = j; offset_x < block_size; offset_x++) {
                    if (!(offset_y == i && offset_x == j)) {
                        Ixx.at<float>(i, j) += Ixx.at<float>(offset_y, offset_x);
                        Iyy.at<float>(i, j) += Iyy.at<float>(offset_y, offset_x);
                        Ixy.at<float>(i, j) += Ixy.at<float>(offset_y, offset_x);
                    }
                }
            }
        }
    }

    GaussianBlur(Ixx, Ixx, Size(3, 3), 3, 0, BORDER_DEFAULT);
    GaussianBlur(Iyy, Iyy, Size(3, 3), 3, 0, BORDER_DEFAULT);
    GaussianBlur(Ixy, Ixy, Size(3, 3), 3, 0, BORDER_DEFAULT);

    // Harris response.
    multiply(Ixx, Iyy, det);
    multiply(Ixy, Ixy, Ixy_2);

    det = det - Ixy_2;
    trace = Ixx + Iyy;

    multiply(trace, trace, trace, k);

    response = det - trace ;

    normalize(response, response, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    // Draw circles around features when over normalized threshold.
    for (int i = 0; i < response.rows; i++) {
        for (int j = 0; j < response.cols; j++) {
            if ((int)response.at<float>(i, j) > threshold) {
                circle(src, Point(j, i), 1, Scalar(222, 0, 0), 1, 8, 0);
            }
        }
    }

    // Create window and display modified src image.
    const char* wdSource = "Corners"; // Name of the app window.
    namedWindow(wdSource, WINDOW_AUTOSIZE);
    imshow(wdSource, src);

    waitKey(0);

    return(0);
}
