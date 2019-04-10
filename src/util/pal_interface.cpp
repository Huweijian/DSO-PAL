#include "pal_interface.h"

using namespace pal;
using namespace std;
using namespace cv;
cv::Mat pal_mask_g[pal_max_level];
cv::Mat pal_valid_sensing_mask_g;
PALCamera* pal_model_g = nullptr;
bool USE_PAL = false;

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


bool pal_init(string calibFile){
    pal_model_g = new pal::PALCamera(calibFile);

	pal_mask_g[0] = cv::Mat::zeros(pal_model_g->height_, pal_model_g->width_, CV_8UC1);
	cv::circle(pal_mask_g[0], cv::Point(pal_model_g->xc_, pal_model_g->yc_), pal_model_g->mask_radius[1], 255, -1);
	cv::circle(pal_mask_g[0], cv::Point(pal_model_g->xc_, pal_model_g->yc_), pal_model_g->mask_radius[0], 0, -1);
    pal_addMaskbuffer(pal_mask_g[0], 5);

	pal_valid_sensing_mask_g[0] = cv::Mat::zeros(pal_model_g->height_, pal_model_g->width_, CV_8UC1);
	cv::circle(pal_valid_sensing_mask_g[0], cv::Point(pal_model_g->xc_, pal_model_g->yc_), pal_model_g->sensing_radius[1], 255, -1);
	cv::circle(pal_valid_sensing_mask_g[0], cv::Point(pal_model_g->xc_, pal_model_g->yc_), pal_model_g->sensing_radius[0], 0, -1);

	for(int i=1; i<pal_max_level; i++){
		cv::resize(pal_mask_g[i-1], pal_mask_g[i], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        pal_addMaskbuffer(pal_mask_g[i], 5);
        // imshow("mask" + to_string(i), pal_mask_g[i]);
	}
    // imshow("mask0", pal_mask_g[0]);
    // waitKey();

    USE_PAL = true;
    if(pal_model_g->mask_radius[0] < pal_model_g->mask_radius[1] || pal_model_g->sensing_radius[0] < pal_model_g->sensing_radius[1])
        return true;
    else
        return false;
}    