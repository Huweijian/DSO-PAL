#include "pal_interface.h"

using namespace pal;
using namespace std;
using namespace cv;
cv::Mat pal_mask_g[pal_max_level];
cv::Mat pal_valid_sensing_mask_g;
cv::Mat pal_weight;
PALCamera* pal_model_g = nullptr;
bool USE_PAL = false;
bool ENH_PAL = false;

bool pal_check_in_range_g(float u, float v, float padding, int level){
    float right = pal_mask_g[level].cols;
    float bottom = pal_mask_g[level].rows;

    if(isnanf(u + v))
        return false;
    if(u < 0 || v < 0 || u > right-1 || v > bottom-1)
        return false;

    if(pal_mask_g[level].at<uchar>((int)v, (int)u) > padding )
        return true;
    else 
        return false;
}

bool pal_check_valid_sensing(float u, float v){
    if(pal_valid_sensing_mask_g.at<uchar>((int)v, (int)u) == 0){
        return false;
    }
    return true;
}

void pal_addMaskbuffer(cv::Mat &mask, int size) {
	Mat maskt = mask.clone();
	Mat maskt_sml;
	for (int i = 0; i < size; i++) {
		erode(maskt, maskt_sml, getStructuringElement(MORPH_RECT, Size(3, 3)));
		Mat diff = (maskt - maskt_sml);
		mask = mask - diff / 255 * (255 - (i + 1));
		maskt = maskt_sml.clone();
	}
}

float pal_get_weight(Eigen::Vector2f pt, int lvl){
    int multi = (int)1 << lvl;
    int x = (pt[0]+0.5f) * multi - 0.5; // 亚像素精度
    int y = (pt[1]+0.5f) * multi - 0.5;   
    return pal_weight.at<float>(y, x);
}


bool pal_init(string calibFile){
    // model
    pal_model_g = new pal::PALCamera(calibFile);
    auto &pal = pal_model_g;

    USE_PAL = true;
    // ENH_PAL = true;
    // max radius check
    if(pal->mask_radius[0] > pal->mask_radius[1] || pal->sensing_radius[0] > pal->sensing_radius[1]){
        return false;
    }
    auto pt_z0 = pal->world2cam(Vector3f(100, 0, 0));
    float maxR_z0 = (pt_z0-Vector2f(pal->xc_, pal->yc_)).norm(); 
    if(maxR_z0 < pal->mask_radius[1]){
        printf(" ! [WARNING] pal mask outer radius is %.2d larger than maximum(%.2f)! force set to maximum\n", pal->mask_radius[1], maxR_z0);
        pal->mask_radius[1] = maxR_z0;
    }

    // valid mask and buffing
	pal_mask_g[0] = cv::Mat::zeros(pal->height_, pal->width_, CV_8UC1);
	cv::circle(pal_mask_g[0], cv::Point(pal->xc_, pal->yc_), pal->mask_radius[1], 255, -1);
	cv::circle(pal_mask_g[0], cv::Point(pal->xc_, pal->yc_), pal->mask_radius[0], 0, -1);
	for(int i=1; i<pal_max_level; i++){
		cv::resize(pal_mask_g[i-1], pal_mask_g[i], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
	}
    for(int i=0; i<pal_max_level; i++){
        pal_addMaskbuffer(pal_mask_g[i], (5-i));
        // imshow("mask" + to_string(i), pal_mask_g[i]);
    }
    // waitKey();

    // valid sensing mask
	pal_valid_sensing_mask_g = cv::Mat::zeros(pal->height_, pal->width_, CV_8UC1);
	cv::circle(pal_valid_sensing_mask_g, cv::Point(pal->xc_, pal->yc_), pal->sensing_radius[1], 255, -1);
	cv::circle(pal_valid_sensing_mask_g, cv::Point(pal->xc_, pal->yc_), pal->sensing_radius[0], 0, -1);

    // pal weight
    pal_weight = Mat::ones(pal->height_, pal->width_, CV_32FC1);
    float setting_pal_min_weight = 0.9;
    float setting_pal_max_weight = 1;
    for(int u = 0; u<pal->width_; u++){
        for(int v = 0; v<pal->height_; v++){
            float w = 0;
            if(pal_check_valid_sensing(u, v)){
                float r = (Vector2f(u, v) - Vector2f(pal->xc_, pal->yc_)).norm();
                float r0 = pal->sensing_radius[0]-5, r1 = pal->sensing_radius[1]+5;
                float maxw = (r1-r0)*(r1-r0); 
                w = -(setting_pal_max_weight - setting_pal_min_weight) * ((r-r0)*(r-r0)/maxw) + setting_pal_max_weight;
                // printf(" - (%d, %d) w=%.2f \n", u, v, w);
            }
            pal_weight.at<float>(u, v) = w;
        }
    }
    // imshow("weight", pal_weight);
    // waitKey();

    return true;
}     