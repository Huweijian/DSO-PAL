#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
namespace dso_line{
    

struct Line{
    Eigen::Vector3f *x0;
    Eigen::Vector3f *u;
};

bool line3d_to_2d(
        const Eigen::Vector3f &line_x0, const Eigen::Vector3f &line_u, 
        Eigen::Vector2f &line2d_x0, Eigen::Vector2f &line2d_u, 
        const Eigen::Matrix3f &R, const Eigen::Vector3f &t, 
        const Eigen::Matrix3f &K);
void draw_line2d(
        cv::Mat &img, 
        const Eigen::Vector2f &x0, const Eigen::Vector2f &u, unsigned char color = 255);


Eigen::Matrix4f plucker_l(const Line &l);

Eigen::Matrix4f plucker_swap(Eigen::Matrix4f &L);

}