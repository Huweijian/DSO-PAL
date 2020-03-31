#include "line_base.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace dso_line{

bool line3d_to_2d(
        const Eigen::Vector3f &line_x0, const Eigen::Vector3f &line_u, 
        Eigen::Vector2f &line2d_x0, Eigen::Vector2f &line2d_u, 
        const Eigen::Matrix3f &R, const Eigen::Vector3f &t, 
        const Matrix3f &K){
    
    Vector3f P[2];
    P[0] = line_x0;
    P[1] = line_x0 + line_u;
    for(int i=0; i<2; i++){
        P[i] = R * P[i] + t; 
        P[i] /= P[i][2]; 
        P[i] = K * P[i];
    }
    line2d_x0 = P[0].block<2, 1>(0, 0);
    P[1] = P[1] - P[0];
    line2d_u = P[1].block<2, 1>(0, 0);
    line2d_u.normalize();
}

void draw_line2d(cv::Mat &img, const Eigen::Vector2f &x0, const Eigen::Vector2f &u, unsigned char color) {
    using namespace Eigen;
    int l_dir[2] = {0, 1};  // 直线的主方向和次方向
    int sz[2] = {img.cols, img.rows};

    if (abs(u[1]) > abs(u[0])) {
        std::swap(l_dir[0], l_dir[1]);
    }

    Vector2f uu = u / u[l_dir[0]];               //主方向归一化
    Vector2f p0 = x0 + (0 - x0[l_dir[0]]) * uu;  //直线起点归零
    Vector2f p1 = x0 + (sz[l_dir[0]] - x0[l_dir[0]]) * uu;
    line(img, cv::Point(p0[0], p0[1]), cv::Point(p1[0], p1[1]), color);
    return;
}

Eigen::Matrix4f plucker_l(const Line &l){
    auto &pt = *l.x0;
    auto &u = *l.u;
    Vector4f p0{pt[0], pt[1], pt[2], 1};
    Vector4f p1{pt[0]+u[0], pt[1]+u[1], pt[2]+u[2], 1};
    return p0*p1.transpose() - p1*p0.transpose();
}

Eigen::Matrix4f plucker_swap(Eigen::Matrix4f &L){
    Eigen::Matrix4f Ls = L;
    Ls(0, 1) = L(2, 3);
    Ls(0, 2) = L(3, 1);
    Ls(0, 3) = L(1, 2);
    Ls(1, 2) = L(0, 3);
    Ls(3, 1) = L(0, 2);
    Ls(2, 3) = L(0, 1);

    Ls(1, 0) = -Ls(0, 1);
    Ls(2, 0) = -Ls(0, 2);
    Ls(2, 1) = -Ls(1, 2);
    Ls(3, 0) = -Ls(0, 3);
    Ls(1, 3) = -Ls(3, 1);
    Ls(3, 2) = -Ls(2, 3);
    return Ls;
}

}