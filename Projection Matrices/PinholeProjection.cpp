/*
 * Danen Van De Ven
 * 100820351
 *
 * COMP 4102 - Assignment 1
 * Question 10
 *
 * February 4, 2015
 */

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    double f = 500.0; // Focal length

    std::vector<cv::Point2f> projectedPoints;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point3f> T;

    objectPoints.push_back(cv::Point3f(150.0, 200.0, 350.0)); // Pixel coordinates of the world point X_w
    T.push_back(cv::Point3f(-70.0, -95.0, -120.0)); // Translation vector.

    // Create the known rotation matrix.
    cv::Mat R(3,3,cv::DataType<double>::type);
    R.at<double>(0,0) = 1;
    R.at<double>(1,0) = 0;
    R.at<double>(2,0) = 0;

    R.at<double>(0,1) = 0;
    R.at<double>(1,1) = 1;
    R.at<double>(2,1) = 0;

    R.at<double>(0,2) = 0;
    R.at<double>(1,2) = 0;
    R.at<double>(2,2) = 1;

    // Create the known intrinsic parameter matrix.
    cv::Mat K(3,3,cv::DataType<double>::type);
    K.at<double>(0,0) = f;
    K.at<double>(1,0) = 0;
    K.at<double>(2,0) = 0;

    K.at<double>(0,1) = 0;
    K.at<double>(1,1) = f;
    K.at<double>(2,1) = 0;

    K.at<double>(0,2) = 320; // Principal point o_x.
    K.at<double>(1,2) = 240; // Principal point o_y.
    K.at<double>(2,2) = 1;

    std::cout << "K: " << K << std::endl;
    std::cout << "R: " << R << std::endl;
    std::cout << "T: " << T << std::endl;

    // Create zero distortion.
    cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
    distCoeffs.at<double>(0) = 0;
    distCoeffs.at<double>(1) = 0;
    distCoeffs.at<double>(2) = 0;
    distCoeffs.at<double>(3) = 0;

    cv::Mat RRodrigues(3,1,cv::DataType<double>::type); //Rodrigues rotation matrix.
    cv::Rodrigues(R,RRodrigues);

    cv::projectPoints(objectPoints, RRodrigues, T, K, distCoeffs, projectedPoints);

    for(unsigned int i = 0; i < projectedPoints.size(); ++i) {
        std::cout << "Image point: " << objectPoints[i] << " Projected to " << projectedPoints[i] << std::endl;
    }

    return 0;
}
