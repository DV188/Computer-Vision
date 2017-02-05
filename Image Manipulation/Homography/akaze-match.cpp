/*
 * Danen Van De Ven
 * 100820351
 *
 * COMP 4102 - Assignment 3, Question 2
 * March 31, 2016
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f; // Nearest neighbor matching ratio

int main(void) {
    cout << "Opening." << endl;

    BFMatcher matcher(NORM_HAMMING);

    Mat homography,
        desc1, desc2,
        img1 = imread("keble_a_half.bmp", IMREAD_GRAYSCALE),
        img2 = imread("keble_b_long.bmp", IMREAD_GRAYSCALE),
        img3 = Mat(img2.rows, img2.cols, CV_8UC1);

    vector<KeyPoint> kpts1, kpts2,
        matched1, matched2;

	vector<Point2f> inliers1, inliers2;

    vector< vector<DMatch> > nn_matches;

	cout << "Opened." << endl;

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

	cout << "Computed akaze." << endl;

    matcher.knnMatch(desc1, desc2, nn_matches, 2);

	cout << "Done match." << endl;
 
    for (size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if (dist1 < nn_match_ratio*dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }

	cout << "Matches " << matched1.size() << " " << matched2.size() << endl;

    // Convert KeyPoint to Point2f to be used in findHomography.
    KeyPoint::convert(matched1, inliers1);
    KeyPoint::convert(matched2, inliers2);

    homography = findHomography(inliers1, inliers2, RANSAC); // Calculate homography using RANSAC descriptor.

    warpPerspective(img1, img3, homography, img3.size()); // Warp img1 wrt. homography points, output to img3.

    // Display input and output
    imshow("Warped", img3);

    bitwise_or(img2, img3, img3); // Combine images to form larger combined image.
    imshow("Merged", img3);

	waitKey(0);

    return 0;
}
