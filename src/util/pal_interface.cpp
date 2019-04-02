#include "pal_interface.h"

using namespace pal;
using namespace std;
using namespace cv;
cv::Mat pal_mask_g[pal_max_level];
PALCamera* pal_model_g = nullptr;

bool pal_check_in_range_g(float u, float v, float left_top, float right, float bottom, int level){
    if(isnanf(u + v))
        return false;
    if(u < left_top || v < left_top || u > right || v > bottom)
        return false;
    if(pal_mask_g[level].at<uchar>((int)v, (int)u) == 0 )
        return false;
    return true;
}

bool pal_check_in_range_g(float u, float v, float padding, int level){
    float right = pal_mask_g[level].cols-padding;
    float bottom = pal_mask_g[level].rows-padding;
    return pal_check_in_range_g(u, v, padding, right, bottom, level);
}

// bool pal_check_in_range_g(int idx, int lvl){
//     int cols = pal_mask_g[lvl].cols; 
//     int rows = pal_mask_g[lvl].rows;
//     return pal_check_in_range_g(idx%cols, idx/cols, 0, cols, rows, lvl);
// }

bool init_pal(string calibFile){
    pal_model_g = new pal::PALCamera(calibFile);
	pal_mask_g[0] = cv::Mat::zeros(pal_model_g->height_, pal_model_g->width_, CV_8UC1);
	cv::circle(pal_mask_g[0], cv::Point(pal_model_g->xc_, pal_model_g->yc_), pal_model_g->mask_radius[1], 255, -1);
	cv::circle(pal_mask_g[0], cv::Point(pal_model_g->xc_, pal_model_g->yc_), pal_model_g->mask_radius[0], 0, -1);

	for(int i=1; i<pal_max_level; i++){
		cv::resize(pal_mask_g[i-1], pal_mask_g[i], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        // imshow("mask" + to_string(i), pal_mask_g[i]);
	}
    // imshow("mask0", pal_mask_g[0]);
    // waitKey();

    if(pal_model_g->mask_radius[0] < pal_model_g->mask_radius[1])
        return true;
    else
        return false;
}    