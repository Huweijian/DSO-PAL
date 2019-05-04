#pragma once
#include "opencv2/opencv.hpp" 
#include <Eigen/Core>
#include "pal_model.h"
#include <string>

#include "util/Undistort.h"
// #define PAL // 不再使用，改用USE_PAL 变量以减少编译次数
extern int USE_PAL;
extern bool ENH_PAL;
const int pal_max_level = 6;
extern pal::PALCamera* pal_model_g;


// inline Eigen::Vector3f cam2world(float x, float y, ocam_model& myocam_model){
//     double pWorld[3];
//     double pCam[2] = {x, y};
//     cam2world(pWorld, pCam, &myocam_model);
//     return Eigen::Vector3f(pWorld[0], pWorld[1], pWorld[2]);
// };

// inline Eigen::Vector2f world2cam(float x, float y, float z, ocam_model& my_model){
//     double pCam[2];
//     double pWorld[3] = {x, y, z};
//     world2cam(pCam, pWorld, &my_model);
//     return Eigen::Vector2f(pCam[0], pCam[1]);
// };

float pal_get_weight(Eigen::Vector2f pt, int lvl = 0);

bool pal_check_in_range_g(float u, float v, float padding, int level = 0);

bool pal_check_valid_sensing(float u, float v);

bool pal_init(std::string calibFile);

bool pal_undistort_mask(dso::Undistort *u);

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