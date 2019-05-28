#pragma once
#include "opencv2/opencv.hpp" 
#include <Eigen/Core>
#include "pal_model.h"
#include <string>
#include <sophus/sim3.hpp>

// #define PAL // 不再使用，改用USE_PAL 变量以减少编译次数
extern int USE_PAL;
extern bool ENH_PAL;
const int pal_max_level = 6;
extern pal::PALCamera* pal_model_g;

float pal_get_weight(Eigen::Vector2f pt, int lvl = 0);

bool pal_check_in_range_g(float u, float v, float padding, int level = 0);

bool pal_check_valid_sensing(float u, float v);

bool pal_init(std::string calibFile);

inline void pal_project(float u_ori, float v_ori, float idepth, const Eigen::Matrix3f &R, const Eigen::Vector3f t, 
    float &u, float &v, float &Ku, float &Kv)
    {
    Eigen::Vector3f pt = R * pal_model_g->cam2world(u_ori, v_ori) + t*idepth;
    u = pt[0] / pt[2];
    v = pt[1] / pt[2];

    Eigen::Vector2f Kpt = pal_model_g->world2cam(pt);
    Ku = Kpt[0];
    Kv = Kpt[1];
}

int getPoseFromMarker(const cv::Mat &img,const Eigen::Matrix3f &K, Eigen::Vector3f &t, Eigen::Matrix3f &R);

bool calcWorldCoord(const Eigen::Matrix3f &Rdso, const Eigen::Vector3f &tdso, const Eigen::Matrix3f &Rmk, const Eigen::Vector3f &tmk, Sophus::Sim3f &Sim3_dso_mk);


const int COORDINATE_ALIGNMENT_BUF_NUM = 100;
class CoordinateAlign{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool calcWorldCoord(const Eigen::Matrix3f &Rdso, const Eigen::Vector3f &tdso, const Eigen::Matrix3f &Rmk, const Eigen::Vector3f &tmk, Sophus::Sim3f &Sim3_dso_mk);
    void resetBuf();

private:
    int cnt = 0;
    Eigen::Matrix<float, 3, COORDINATE_ALIGNMENT_BUF_NUM> tra_dso_buf; // dso_buf 
    Eigen::Matrix<float, 3*COORDINATE_ALIGNMENT_BUF_NUM, 1> B; // mk_buf
    Eigen::Vector4f Rv_dso2mk_mean;
};