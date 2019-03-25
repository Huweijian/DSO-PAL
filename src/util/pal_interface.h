#pragma once
#include "opencv2/opencv.hpp" 
#include <Eigen/Core>
#include "pal_model.h"
#include <string>

// #define PAL
const int pal_max_level = 6;
extern cv::Mat pal_mask_g[pal_max_level];
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

bool pal_check_in_range_g(float u, float v, float left_top, float right, float bottom, int level = 0);

bool pal_check_in_range_g(float u, float v, float padding, int level = 0);

bool pal_check_in_range_g(int idx, int lvl = 0);

bool init_pal(std::string calibFile);