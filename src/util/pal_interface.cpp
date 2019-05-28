#include "pal_interface.h"
#include "IOWrapper/ImageDisplay.h"
#include "aruco/aruco.h"
#include "opencv2/core/eigen.hpp"

using namespace pal;
using namespace std;
using namespace cv;
cv::Mat pal_mask_g[pal_max_level];
cv::Mat pal_valid_sensing_mask_g;
cv::Mat pal_weight;
PALCamera* pal_model_g = nullptr;
static aruco::MarkerDetector mdetector;

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
    float maxR_z0 = (pt_z0-Vector2f(pal->cx, pal->cy)).norm(); 
    if(maxR_z0 < pal->mask_radius[1]){
        printf(" ! [WARNING] pal mask outer radius is %.2d larger than maximum(%.2f)! force set to maximum\n", pal->mask_radius[1], maxR_z0);
        pal->mask_radius[1] = maxR_z0;
    }

    // init mask and buffing
    int hh = pal->height_, ww = pal->width_;
    if(USE_PAL == 1){
        float r1 = pal->mask_radius[1], r0 = pal->mask_radius[0];
        float cx = pal->cx, cy = pal->cy;

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

        float xi = pal_model_g->cx + pal_model_g->mask_radius[0];
        float yi = pal_model_g->cy;
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
                // dso的图像尺寸是640*480, 都是32的倍数,所以在点选择的时候,它可以把图像恰好划分为多个32*32的块,对于pal图像,分辨率为720,不能整除,
                // 下面的代码把不能整除的部分赋值为0.
                // 这个解决方案不太好,先这样把
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
	cv::circle(pal_valid_sensing_mask_g, cv::Point(pal->cx, pal->cy), pal->sensing_radius[1], 255, -1);
	cv::circle(pal_valid_sensing_mask_g, cv::Point(pal->cx, pal->cy), pal->sensing_radius[0], 0, -1);
    // imshow("sensing area", pal_valid_sensing_mask_g); waitKey();

    // pal weight
    pal_weight = Mat::ones(pal->height_, pal->width_, CV_32FC1);
    float setting_pal_min_weight = 0.99;
    float setting_pal_max_weight = 1;
    for(int u = 0; u<pal->width_; u++){
        for(int v = 0; v<pal->height_; v++){
            float w = 0;
            if(pal_check_valid_sensing(u, v)){
                float r = (Vector2f(u, v) - Vector2f(pal->cx, pal->cy)).norm();
                float r0 = pal->sensing_radius[0]-5, r1 = pal->sensing_radius[1]+5;
                float maxw = (r1-r0)*(r1-r0); 
                w = -(setting_pal_max_weight - setting_pal_min_weight) * ((r-r0)*(r-r0)/maxw) + setting_pal_max_weight;
                // printf(" - (%d, %d) w=%.2f \n", u, v, w);
            }
            pal_weight.at<float>(v, u) = w;
        }
    }
    // imshow("weight", pal_weight);
    // waitKey();

    if(ENH_PAL){
        printf(" ! [ENH_PAL] is on !!!!!\n");
    }

    return true;
}

// return marker id(-1 means no marker is detected);
// warning this is R vector
int getPoseFromMarker(const cv::Mat &img,const Eigen::Matrix3f &K, Eigen::Vector3f &t, Eigen::Matrix3f &R){
    using namespace aruco;
    const float MARKER_SIZE = 0.139; // m

    int mkid = -1;
    Mat Kcv(3, 3, CV_32FC1);
    eigen2cv(K, Kcv);

	vector<Marker> Markers;
    mdetector.detect(img, Markers, Kcv, Mat(), 0.139, false); // 13.9 cm

	if(Markers.size() == 1){
        auto &mk = Markers[0];
        mkid = mk.id;

		for(auto &pt:mk)
			circle(img, Point(pt.x, pt.y), 3, 255);

        // 自己求解H R t
        // std::vector<cv::Point2f> ptsP;	// marker的墙壁坐标系
        // std::vector<cv::Point2f> ptsI; // marker的矫正后归一化图像坐标系
		// ptsI.push_back(cv::Point2f(mk[0].x, mk[0].y));
		// ptsI.push_back(cv::Point2f(mk[1].x, mk[1].y));
		// ptsI.push_back(cv::Point2f(mk[2].x, mk[2].y));
		// ptsI.push_back(cv::Point2f(mk[3].x, mk[3].y));
		// ptsP.push_back(cv::Point2f(-MARKER_SIZE/2,MARKER_SIZE/2));
		// ptsP.push_back(cv::Point2f(MARKER_SIZE/2, MARKER_SIZE/2));
		// ptsP.push_back(cv::Point2f(MARKER_SIZE/2, -MARKER_SIZE/2));
		// ptsP.push_back(cv::Point2f(-MARKER_SIZE/2, -MARKER_SIZE/2));
        // cv::Mat Hcv = cv::findHomography(ptsP, ptsI);
        // vector<Mat> Rcv; 
        // vector<Mat> tcv; 
        // vector<Mat> norm;
        // int resolution = cv::decomposeHomographyMat(Hcv, Kcv, Rcv, tcv, norm);

        cv::Mat Rcv;
        Rodrigues(mk.Rvec, Rcv);
        cv2eigen(Rcv, R);
        cv2eigen(mk.Tvec, t);
        
        // 这里的R和t是marker相对于cam的
        Matrix4f T; 
        T << R, t, 0, 0, 0, 1;
        R = T.block(0, 0, 3, 3);
        t = T.block(0, 3, 3, 1);

        // cout << T << endl;
        // cout << R << endl;
        // cout << t << endl;

        // imshow("aha", img);
        // waitKey();
	}
    return mkid;
}

void CoordinateAlign::resetBuf(){
    cnt = 0;
    tra_dso_buf.setZero(); // dso_buf 
    B.setZero(); // mk_buf
    Rv_dso2mk_mean.setZero();
    printf(" [Coord Align] Reset!\n");
    waitKey(1000);
}

// dso: pose cam to dso frame 0
// mk: pose marker to cam
bool CoordinateAlign::calcWorldCoord(const Matrix3f &Rdso, const Vector3f &tdso, const Matrix3f &Rmk, const Vector3f &tmk, Sophus::Sim3f &Sim3_dso_mk){
    using namespace Sophus;

    if(cnt > COORDINATE_ALIGNMENT_BUF_NUM)
        return false;

    // localize t variables
    tra_dso_buf.col(cnt) = tdso;
    Matrix4f Tmk;
    Tmk << Rmk, tmk, 0, 0, 0, 1;
    Matrix4f Tmkinv = Tmk.inverse();
    B.block(cnt*3, 0, 3, 1) = Tmkinv.block<3, 1>(0, 3);

    // accumulate rotation 
    cnt ++;
    Matrix3f R_dso2mk = Rdso * Rmk;
    AngleAxisf Rv_dso2mk_tem(R_dso2mk);
    Vector4f Rv_dso2mk;
    Rv_dso2mk << Rv_dso2mk_tem.axis(), Rv_dso2mk_tem.angle();
    float w = ((float)(cnt-1)) / cnt;
    Rv_dso2mk_mean = Rv_dso2mk_mean * w + Rv_dso2mk * (1-w);

    // // output Rv to debug 
    // static ofstream testLog("/home/hwj23/Desktop/pal_rot.txt");
    // char outmsg[100] = "";
    // sprintf(outmsg, "%.4f %.4f %.4f %.4f", Rv_dso2mk(0), Rv_dso2mk(1) ,Rv_dso2mk(2) ,Rv_dso2mk(3));
    // testLog << outmsg << endl;

    // output rotation vec--------
    cout << " [$$$] global map initilize cnt = " << cnt << "  Rv = ";
    // cout << R_dso2mk << endl;
    cout << Rv_dso2mk_mean.transpose() << endl;
    // ------------------

    // calc scale and t through DLT
    if(cnt == COORDINATE_ALIGNMENT_BUF_NUM){
        // rotate trajectory
        Vector3f ax = Rv_dso2mk_mean.head<3>();
        ax.normalize();
        float angle = Rv_dso2mk_mean(3);
        Matrix3f R_dso2mk_traj =AngleAxisf(angle, ax).toRotationMatrix().inverse();
        Sim3_dso_mk.setRotationMatrix(R_dso2mk_traj);
        Matrix<float, 3, COORDINATE_ALIGNMENT_BUF_NUM> tra_dso2 = R_dso2mk_traj * tra_dso_buf;

        // cout << "rotation matrix:" << endl;
        // cout << R_dso2mk << endl << endl;

        // cout << "t dso:" << endl;
        // cout << tra_dso2.block(0, (COORDINATE_ALIGNMENT_BUF_NUM-6)*3, 3, 5) << endl << endl;

        // cout << "t mk" << endl;
        // cout << B.block((COORDINATE_ALIGNMENT_BUF_NUM-6)*3, 0, 15, 1) << endl;

        // DLT to calc s and t 
        Matrix<float, COORDINATE_ALIGNMENT_BUF_NUM*3, 4> A;
        A.setZero();
        for(int i=0; i<COORDINATE_ALIGNMENT_BUF_NUM; i++){
            A.block<3, 1>(i*3, 0) = tra_dso2.col(i);
            A(i*3, 1) = 1;
            A(i*3+1, 2) = 1;
            A(i*3+2, 3) = 1;
        }
        Vector4f dlt = (A.transpose()*A).ldlt().solve(A.transpose()*B);
        // cout << "DLT result = " << dlt.transpose() << endl;
        
        Matrix<float, COORDINATE_ALIGNMENT_BUF_NUM*3, 1> err;
        err = A*dlt - B;
        cout << "err = " << err.squaredNorm()/COORDINATE_ALIGNMENT_BUF_NUM << endl;
        
        Sim3_dso_mk.setScale(dlt(0));
        Sim3_dso_mk.translation() = dlt.tail<3>();
        return true;
    }
    else
    {
        return false;
    }

}


