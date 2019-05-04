#include "pal_interface.h"
#include "IOWrapper/ImageDisplay.h"
using namespace pal;
using namespace std;
using namespace cv;
cv::Mat pal_mask_g[pal_max_level];
cv::Mat pal_valid_sensing_mask_g;
cv::Mat pal_weight;
PALCamera* pal_model_g = nullptr;

int USE_PAL = 0;
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

    USE_PAL = pal_model_g->undistort_mode;

    if(USE_PAL == 1){
        ENH_PAL = true;
    }

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

    // init mask and buffing
    int hh = pal->height_, ww = pal->width_;
    if(USE_PAL == 1){
        float r1 = pal->mask_radius[1], r0 = pal->mask_radius[0];
        float cx = pal->xc_, cy = pal->yc_;

        for(int i=0; i<pal_max_level; i++){
            pal_mask_g[i] = cv::Mat::zeros(hh, ww, CV_8UC1);
            cv::circle(pal_mask_g[i], cv::Point(cx, cy), r1, 255, -1);
            cv::circle(pal_mask_g[i], cv::Point(cx, cy), r0, 0, -1);
            hh/=2; ww/=2; r1/=2.0; r0/=2.0; cx/=2.0; cy/=2.0;
        }
    }
    else if(USE_PAL == 2){
        float ocx = pal_model_g->width_ * pal_model_g->pin_cx;
        float ocy = pal_model_g->height_ * pal_model_g->pin_cy;
        float ofx = pal_model_g->width_ * pal_model_g->pin_fx; 
        float ofy = pal_model_g->height_ * pal_model_g->pin_fy;

        float xi = pal_model_g->xc_ + pal_model_g->mask_radius[0];
        float yi = pal_model_g->yc_;
        float x_cam = (xi - ocx) / ofx;
        float y_cam = (yi - ocy) / ofy;
        auto pt_ori = pal_model_g->world2cam(Eigen::Vector3f(x_cam, y_cam, 1));
        float r_inner = pt_ori[0] - pal_model_g->width_ * pal_model_g->pin_cx; 

        float cx = pal->width_*pal->pin_cx, cy = pal->height_*pal->pin_cy;

        for(int i=0; i<pal_max_level; i++){
            pal_mask_g[i] = cv::Mat::ones(hh, ww, CV_8UC1)*255;
            circle(pal_mask_g[i], cv::Point(cx, cy), r_inner, 0, -1);
            rectangle(pal_mask_g[i], Rect(Point(0, 0), Point(ww-1, hh-1)), 0); 
            if(i == 0){
                // TODO: 这个解决方案不太好,先这样把
                pal_mask_g[i].rowRange(hh/32*32, hh-1) = 0;
            }
            hh/=2; ww/=2; cx/=2.0; cy/=2.0; r_inner/=2;
        }

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
    float setting_pal_min_weight = 0.99;
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

    if(ENH_PAL){
        printf(" ! [ENH_PAL] is on !!!!!\n");
    }

    return true;
}     

bool pal_undistort_mask(dso::Undistort *u){
    using namespace dso;
    for(int i=0; i<pal_max_level; i++){
        MinimalImageB img(pal_mask_g[i].cols, pal_mask_g[i].rows);
        memcpy(img.data, pal_mask_g[i].data, img.w * img.h);
        u->undistort<unsigned char>(&img, 1.0f, 0.0);
        pal_mask_g[i] = IOWrap::getOCVImg_tem(img.data, img.w, img.h);
        imshow("mask" + to_string(i), pal_mask_g[i]);
    }
    waitKey();
}