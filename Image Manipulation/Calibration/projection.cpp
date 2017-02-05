/*
 * Danen Van De Ven
 * 100820351
 *
 * COMP 4102 - Assignment 3, Question 1
 * March 31, 2016
 */

#include <stdio.h>
#include "opencv/cv.h"

#define NUM_POINTS 10
#define RANGE 100.00

#define MAX_CAMERAS 100 
#define MAX_POINTS 3000

float projection[3][4] = {
    0.902701, 0.051530, 0.427171, 10.0,
    0.182987, 0.852568, -0.489535, 15.0,
    -0.389418, 0.520070, 0.760184, 20.0
};

float intrinsic[3][3] = {
    -1000.000000, 0.000000, 0.000000, 
    0.000000, -2000.000000, 0.000000, 
    0.000000, 0.000000, 1.000000
};

float all_object_points[10][3] = {
    0.1251, 56.3585, 19.3304, 
    80.8741, 58.5009, 47.9873,
    35.0291, 89.5962, 82.2840,
    74.6605, 17.4108, 85.8943,
    71.0501, 51.3535, 30.3995,
    1.4985, 9.1403, 36.4452,
    14.7313, 16.5899, 98.8525,
    44.5692, 11.9083, 0.4669,
    0.8911, 37.7880, 53.1663,
    57.1184, 60.1764, 60.7166
};


// Prints the matrix to std out with tabbed columns.
// Used instead of std::cout << matrix << std::endl;
void print_matrix(cv::Mat *matrix) {
    for (int row = 0; row < matrix->rows; row++) {
        for (int col = 0; col < matrix->cols; col++) {
            printf("%f\t", matrix->at<float>(row, col));
        }

        printf("\n");
    }
}

// you write this routine
void compute_projection_matrix(CvMat *image_points, CvMat *object_points, CvMat *projection_matrix) {
    cv::Mat D_mat, R_mat, Q_mat,
        image_points_mat(image_points),
        object_points_mat(object_points),
        projection_matrix_mat(projection_matrix);

    D_mat = cv::Mat::zeros(NUM_POINTS*2, 11, CV_32F);
    R_mat = cv::Mat::zeros(NUM_POINTS*2, 1, CV_32F);
    Q_mat = cv::Mat::zeros(12, 1, CV_32F);

    int i = 0;

    for (int row = 0; row < NUM_POINTS; row++) {
        D_mat.at<float>(row, 0) = object_points_mat.at<float>(row, 0);
        D_mat.at<float>(row, 1) = object_points_mat.at<float>(row, 1);
        D_mat.at<float>(row, 2) = object_points_mat.at<float>(row, 2);
        D_mat.at<float>(row, 3) = 1;
        D_mat.at<float>(row, 4) = 0;
        D_mat.at<float>(row, 5) = 0;
        D_mat.at<float>(row, 6) = 0;
        D_mat.at<float>(row, 7) = 0;
        D_mat.at<float>(row, 8) = -image_points_mat.at<float>(row, 0)*object_points_mat.at<float>(row, 0);
        D_mat.at<float>(row, 9) = -image_points_mat.at<float>(row, 0)*object_points_mat.at<float>(row, 1);
        D_mat.at<float>(row, 10) = -image_points_mat.at<float>(row, 0)*object_points_mat.at<float>(row, 2);

        D_mat.at<float>(row + NUM_POINTS, 0) = 0;
        D_mat.at<float>(row + NUM_POINTS, 1) = 0;
        D_mat.at<float>(row + NUM_POINTS, 2) = 0;
        D_mat.at<float>(row + NUM_POINTS, 3) = 0;
        D_mat.at<float>(row + NUM_POINTS, 4) = object_points_mat.at<float>(row, 0);
        D_mat.at<float>(row + NUM_POINTS, 5) = object_points_mat.at<float>(row, 1);
        D_mat.at<float>(row + NUM_POINTS, 6) = object_points_mat.at<float>(row, 2);
        D_mat.at<float>(row + NUM_POINTS, 7) = 1;
        D_mat.at<float>(row + NUM_POINTS, 8) = -image_points_mat.at<float>(row, 1)*object_points_mat.at<float>(row, 0);
        D_mat.at<float>(row + NUM_POINTS, 9) = -image_points_mat.at<float>(row, 1)*object_points_mat.at<float>(row, 1);
        D_mat.at<float>(row + NUM_POINTS, 10) = -image_points_mat.at<float>(row, 1)*object_points_mat.at<float>(row, 2);
    }

    // print_matrix(&D_mat); // Print D matrix to std out.

    // Create R matrix.
    for (int row = 0; row < image_points_mat.rows; row++) {
        R_mat.at<float>(row, 0) = image_points_mat.at<float>(row, 0);
    }

    for (int row = 0; row < image_points_mat.rows; row++) {
        R_mat.at<float>(row + NUM_POINTS, 0) = image_points_mat.at<float>(row, 1);
    }

    // printf("\n");
    // print_matrix(&R_mat); // Print R matrix.

    // Q = (A^T*A)^(-1)*A^T*R
    Q_mat = (D_mat.t()*D_mat).inv(cv::DECOMP_LU)*D_mat.t()*R_mat;

    // printf("\n");
    // print_matrix(&Q_mat); // Print Q matrix to std out.

    // Reorganize 12x1 matrix Q to 4x3 matrix to be returned.
    i = 0; // Reusing counter.

    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 4; col++) {
            if (row == 2 && col == 3)
                cvmSet(projection_matrix, row, col, 1.0);
            else
                cvmSet(projection_matrix, row, col, Q_mat.at<float>(i++, 0));
        }
    }
}

// you write this routine
void decompose_projection_matrix(CvMat* projection_matrix, CvMat* rotation_matrix, CvMat* translation, CvMat* camera_matrix) {
    cv::Mat q_1, q_2, q_3, q_4,
        projection_matrix_mat(projection_matrix),
        rotation_matrix_mat(rotation_matrix),
        translation_mat(translation),
        camera_matrix_mat(camera_matrix),
        temp;

    float gamma = sqrt(pow(projection_matrix_mat.at<float>(2, 0), 2) +
            pow(projection_matrix_mat.at<float>(2, 1), 2) +
            pow(projection_matrix_mat.at<float>(2, 2) ,2));

    translation_mat.at<float>(2, 0) = projection_matrix_mat.at<float>(2, 3)/gamma;

    q_1 = cv::Mat::zeros(3, 1, CV_32F);
    q_2 = cv::Mat::zeros(3, 1, CV_32F);
    q_3 = cv::Mat::zeros(3, 1, CV_32F);
    q_4 = cv::Mat::zeros(3, 1, CV_32F);

    // q_1 = [m_11, m_12, m_13]^T
    q_1.at<float>(0, 0) = projection_matrix_mat.at<float>(0, 0)/gamma;
    q_1.at<float>(1, 0) = projection_matrix_mat.at<float>(0, 1)/gamma;
    q_1.at<float>(2, 0) = projection_matrix_mat.at<float>(0, 2)/gamma;

    // q_2 = [m_21, m_22, m_23]^T
    q_2.at<float>(0, 0) = projection_matrix_mat.at<float>(1, 0)/gamma;
    q_2.at<float>(1, 0) = projection_matrix_mat.at<float>(1, 1)/gamma;
    q_2.at<float>(2, 0) = projection_matrix_mat.at<float>(1, 2)/gamma;

    // q_3 = [m_31, m_32, m_33]^T
    q_3.at<float>(0, 0) = projection_matrix_mat.at<float>(2, 0)/gamma;
    q_3.at<float>(1, 0) = projection_matrix_mat.at<float>(2, 1)/gamma;
    q_3.at<float>(2, 0) = projection_matrix_mat.at<float>(2, 2)/gamma;

    // q_4 = [m_14, m_24, m_34]^
    q_4.at<float>(0, 0) = projection_matrix_mat.at<float>(0, 3)/gamma;
    q_4.at<float>(1, 0) = projection_matrix_mat.at<float>(1, 3)/gamma;
    q_4.at<float>(2, 0) = projection_matrix_mat.at<float>(2, 3)/gamma;

    for (int col = 0; col < 3; col++) {
        rotation_matrix_mat.at<float>(2, col) = projection_matrix_mat.at<float>(2, col)/gamma;
    }

    temp = q_1.t()*q_3;
    camera_matrix_mat.at<float>(0, 2) = temp.at<float>(0, 0);

    temp = q_2.t()*q_3;
    camera_matrix_mat.at<float>(1, 2) = temp.at<float>(0, 0);

    temp = q_1.t()*q_1;
    camera_matrix_mat.at<float>(0, 0) = sqrt(temp.at<float>(0, 0) - pow(camera_matrix_mat.at<float>(0, 2), 2));

    temp = q_2.t()*q_2;
    camera_matrix_mat.at<float>(1, 1) = sqrt(temp.at<float>(0, 0) - pow(camera_matrix_mat.at<float>(1, 2), 2));

    camera_matrix_mat.at<float>(2, 2) = 1;

    for (int col = 0; col < 3; col++) {
        rotation_matrix_mat.at<float>(0, col) = (camera_matrix_mat.at<float>(0, 2)*projection_matrix_mat.at<float>(3, col) -
                projection_matrix_mat.at<float>(0, col)/gamma)/camera_matrix_mat.at<float>(0, 0);

        rotation_matrix_mat.at<float>(1, col) = (camera_matrix_mat.at<float>(1, 2)*projection_matrix_mat.at<float>(3, col) -
                projection_matrix_mat.at<float>(1, col)/gamma)/camera_matrix_mat.at<float>(1, 1);
    }

    translation_mat.at<float>(0, 0) = (camera_matrix_mat.at<float>(0, 2)*translation_mat.at<float>(2, 0) -
            projection_matrix_mat.at<float>(0, 3)/gamma)/camera_matrix_mat.at<float>(0, 0);

    translation_mat.at<float>(1, 0) = (camera_matrix_mat.at<float>(1, 2)*translation_mat.at<float>(2, 0) -
            projection_matrix_mat.at<float>(1, 3)/gamma)/camera_matrix_mat.at<float>(1, 1);

    *camera_matrix = camera_matrix_mat;
    *rotation_matrix= rotation_matrix_mat;
    *translation = translation_mat;
}

int main() {
    CvMat *camera_matrix, *computed_camera_matrix,
          *rotation_matrix, *computed_rotation_matrix,
          *translation, *computed_translation,
          *image_points, *transp_image_points,
          *rot_vector,
          *object_points, *transp_object_points,
          *computed_projection_matrix,
          *final_projection;

    CvMat temp_projection, temp_intrinsic;

    FILE *fp;

    cvInitMatHeader(&temp_projection, 3, 4, CV_32FC1, projection);
    cvInitMatHeader(&temp_intrinsic, 3, 3, CV_32FC1, intrinsic);

    final_projection = cvCreateMat(3, 4, CV_32F);

    object_points = cvCreateMat(NUM_POINTS, 4, CV_32F);
    transp_object_points = cvCreateMat(4, NUM_POINTS, CV_32F);

    image_points = cvCreateMat(NUM_POINTS, 3, CV_32F);
    transp_image_points = cvCreateMat(3, NUM_POINTS, CV_32F);

    rot_vector = cvCreateMat(3, 1, CV_32F);
    camera_matrix = cvCreateMat(3, 3, CV_32F);
    rotation_matrix = cvCreateMat(3, 3, CV_32F);
    translation = cvCreateMat(3, 1, CV_32F);

    computed_camera_matrix = cvCreateMat(3, 3, CV_32F);
    computed_rotation_matrix = cvCreateMat(3, 3, CV_32F);
    computed_translation = cvCreateMat(3, 1, CV_32F);
    computed_projection_matrix = cvCreateMat(3, 4, CV_32F);

    fp = fopen("assign3-out","w");

    fprintf(fp, "Rotation matrix\n");

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cvmSet(camera_matrix, i, j, intrinsic[i][j]);
            cvmSet(rotation_matrix, i, j, projection[i][j]);
        }

        fprintf(fp, "%f %f %f\n", 
                cvmGet(rotation_matrix,i,0),
                cvmGet(rotation_matrix,i,1),
                cvmGet(rotation_matrix,i,2));
    }

    for (int i = 0; i < 3; i++)
        cvmSet(translation, i, 0, projection[i][3]);

    fprintf(fp, "\nTranslation vector\n");
    fprintf(fp, "%f %f %f\n", 
            cvmGet(translation,0,0),
            cvmGet(translation,1,0),
            cvmGet(translation,2,0));

    fprintf(fp, "\nCamera Calibration\n");

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%f %f %f\n", 
                cvmGet(camera_matrix,i,0),
                cvmGet(camera_matrix,i,1),
                cvmGet(camera_matrix,i,2));
    }

    fprintf(fp,"\n");

    for (int i = 0; i < NUM_POINTS; i++) {
        cvmSet(object_points, i, 0, all_object_points[i][0]);
        cvmSet(object_points, i, 1, all_object_points[i][1]);
        cvmSet(object_points, i, 2, all_object_points[i][2]);
        cvmSet(object_points, i, 3, 1.0);
        fprintf(fp, "Object point %d X %f Y %f Z %f\n", i,
                all_object_points[i][0],
                all_object_points[i][1],
                all_object_points[i][2]);
    }
    fprintf(fp, "\n");

    cvTranspose(object_points, transp_object_points);

    cvMatMul(&temp_intrinsic, &temp_projection, final_projection);
    cvMatMul(final_projection, transp_object_points, transp_image_points);
    //cvTranspose(transp_image_points, image_points);

    for (int i = 0; i < NUM_POINTS; i++) {
        cvmSet(image_points, i, 0, cvmGet(transp_image_points, 0, i)/cvmGet(transp_image_points, 2, i));
        cvmSet(image_points, i, 1, cvmGet(transp_image_points, 1, i)/cvmGet(transp_image_points, 2, i));

        fprintf(fp, "Image point %d x %f y %f\n", i,
                cvmGet(image_points, i, 0),
                cvmGet(image_points, i, 1));
    }

    compute_projection_matrix(image_points, object_points, computed_projection_matrix);
    decompose_projection_matrix(computed_projection_matrix, computed_rotation_matrix, computed_translation, computed_camera_matrix);

    fprintf(fp, "\nComputed Projection matrix\n");

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%f %f %f %f\n", 
                cvmGet(computed_projection_matrix, i, 0),
                cvmGet(computed_projection_matrix, i, 1),
                cvmGet(computed_projection_matrix, i, 2),
                cvmGet(computed_projection_matrix, i, 3));
    }

    fprintf(fp, "\nComputed Rotation matrix\n");

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%f %f %f\n", 
                cvmGet(computed_rotation_matrix,i,0),
                cvmGet(computed_rotation_matrix,i,1),
                cvmGet(computed_rotation_matrix,i,2));
    }

    fprintf(fp, "\nComputed Translation vector\n");
    fprintf(fp, "%f %f %f\n", 
            cvmGet(computed_translation,0,0),
            cvmGet(computed_translation,1,0),
            cvmGet(computed_translation,2,0));
    fprintf(fp, "\nComputed Camera Calibration\n");

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%f %f %f\n", 
                cvmGet(computed_camera_matrix,i,0),
                cvmGet(computed_camera_matrix,i,1),
                cvmGet(computed_camera_matrix,i,2));
    }

    fclose(fp);
    return 0;
}
