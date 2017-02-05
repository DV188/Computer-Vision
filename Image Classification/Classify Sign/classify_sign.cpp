/*
 * Danen Van De Ven
 * 100820351
 * COMP 4102 - Assignment 2
 * March 1, 2016
 */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

#define NO_MATCH 0
#define STOP_SIGN 1
#define SPEED_LIMIT_40_SIGN 2
#define SPEED_LIMIT_80_SIGN 3

static double angle(Point pt1, Point pt2, Point pt0);

int main(int argc, char* argv[]) {
    bool is_speed_sign = true;

    int canny_thresh = 154;
    int sign_recog_result = NO_MATCH;

    Mat src,
        src_gray, canny_edges,
        drawing,
        speed_80, speed_40,
        perspective_matrix, perspective_output, perspective_output_gray;

    Point2f perspective_inputQuad[4], perspective_outputQuad[4];

    Scalar color_blue = Scalar(255, 0, 0);
    Scalar color_green = Scalar(0, 255, 0);
    Scalar color_red = Scalar(0, 0, 255);

    vector<vector<Point> > contours;

    // Reference speed sign images.
    speed_40 = imread("speed_40.bmp", 0);
    speed_80 = imread("speed_80.bmp", 0);

    // string sign_name = "speedsign3";
    // string sign_name = "stop4";
    // string sign_name = "speedsign4";
    string sign_name = "speedsign12";

    string final_sign_input_name = sign_name + ".jpg";
    string final_sign_output_name = sign_name + "_result" + ".jpg";

    // Load source image.
    src = imread(final_sign_input_name, IMREAD_COLOR);

    // Convert image to gray and blur it.
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    blur(src_gray, src_gray, Size(3, 3));

    // Perform Canny edge detector on grayscale/blur image.
    Canny(src_gray, canny_edges, canny_thresh, canny_thresh*.5, 3);

    // Find the contours of the edges found during the Canny step.
    findContours(canny_edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

    vector<vector<Point> > contours_poly(contours.size());

    // Approximates the corner points of contours found by findContours.
    for (int i = 0; i < contours.size(); i++) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], contours[i].size()*0.02, true);
    }

    drawing = Mat::zeros(canny_edges.size(), CV_8UC3);

    for (int i = 0; i < contours.size(); i++) {
        drawContours(drawing, contours, i, color_blue, 1, 8, vector<Vec4i>(), 0, Point());
    }

    for (int i = 0; i < contours_poly.size(); i++) {
        for (int j = 0; j < contours_poly[i].size(); j++) {
            drawContours(drawing, Mat(contours_poly[i]), j, color_green, 1, 8, vector<Vec4i>(), 0, Point());
        }

        if (contours_poly[i].size() == 4) {
            double maxCosine = 0;

            for (int j = 2; j < 5; j++) {
                // Find the maximum cosine of the angle between joint edges.
                double cosine = fabs(angle(contours_poly[i][j%4], contours_poly[i][j - 2], contours_poly[i][j - 1]));
                maxCosine = MAX(maxCosine, cosine);
            }

            if (maxCosine < 0.3) {
                cout << "Speed sign likely in image." << endl;
                drawContours(drawing, contours, i, color_red, 2, 8, vector<Vec4i>(), 0, Point());
                drawContours(src, contours, i, color_red, 2, 8, vector<Vec4i>(), 0, Point());

                perspective_inputQuad[0] = contours_poly[i][3];
                perspective_inputQuad[1] = contours_poly[i][0];
                perspective_inputQuad[2] = contours_poly[i][1];
                perspective_inputQuad[3] = contours_poly[i][2];
            }
        }

        if (contours_poly[i].size() == 8 && fabs(contourArea(Mat(contours_poly[i]))) > 1000) {
            double maxCosine = 0;

            for (int j = 2; j < 9; j++) {
                // Find the maximum cosine of the angle between joint edges.
                double cosine = fabs(angle(contours_poly[i][j%8], contours_poly[i][j - 2], contours_poly[i][j - 1]));
                maxCosine = MAX(maxCosine, cosine);
            }

            if (maxCosine > 0.7 && maxCosine < 0.8) {
                cout << "Stop sign likely in image." << endl;
                is_speed_sign = false;
                drawContours(drawing, contours, i, color_red, 2, 8, vector<Vec4i>(), 0, Point());
                drawContours(src, contours, i, color_blue, 2, 8, vector<Vec4i>(), 0, Point());
            }
        }
    }

    perspective_outputQuad[0] = Point2f(speed_40.cols - 1, 0);
    perspective_outputQuad[1] = Point2f(0, 0);
    perspective_outputQuad[2] = Point2f(0, speed_40.rows - 1);
    perspective_outputQuad[3] = Point2f(speed_40.cols - 1, speed_40.rows - 1);

    perspective_matrix = Mat::zeros(src.rows, src.cols, src.type());
    perspective_matrix = getPerspectiveTransform(perspective_inputQuad, perspective_outputQuad);

    warpPerspective(src, perspective_output, perspective_matrix, speed_40.size());

    cvtColor(perspective_output, perspective_output_gray, COLOR_BGR2GRAY);

    if (is_speed_sign) {
        if (norm(perspective_output_gray) < norm(speed_40))
            sign_recog_result = SPEED_LIMIT_40_SIGN;
        else if (norm(perspective_output_gray) < norm(speed_80))
            sign_recog_result = SPEED_LIMIT_80_SIGN;
        else
            sign_recog_result = NO_MATCH;
    } else
        sign_recog_result = STOP_SIGN;

    // Draw text.
    string text;
    if (sign_recog_result == SPEED_LIMIT_40_SIGN) text = "Speed 40";
    else if (sign_recog_result == SPEED_LIMIT_80_SIGN) text = "Speed 80";
    else if (sign_recog_result == STOP_SIGN) text = "Stop";
    else if (sign_recog_result == NO_MATCH) text = "Fail";

    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 2;
    int thickness = 3;
    cv::Point textOrg(10, 130);
    cv::putText(src, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    // Create Window.
    const char* source_window = "Result";
    namedWindow(source_window, WINDOW_AUTOSIZE);
    imwrite(final_sign_output_name, src);

    const char* contour_window = "Contour Window";
    namedWindow(contour_window, WINDOW_AUTOSIZE);
    moveWindow("Contour Window", 300, 0);

    imshow(source_window, src);
    imshow(contour_window, drawing);

    if (is_speed_sign) {
        const char* perspective_window = "Perspective Window";
        namedWindow(perspective_window, WINDOW_AUTOSIZE);
        moveWindow("Perspective Window", 600, 0);

        imshow(perspective_window, perspective_output);
    }

    waitKey(0);

    return(0);
}

// Helper function:
// Finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2.
static double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
