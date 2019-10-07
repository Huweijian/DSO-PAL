#include <iostream>
#include <string>
#include <Eigen/Core>
#include <util/pal_model.h>
#include <util/pal_interface.h>
#include <OptimizationBackend/MatrixAccumulators.h>
#include <vector>
#include <opencv2/core/eigen.hpp>
#include <util/Undistort.h>

#define T transpose()
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace dso;
using namespace Sophus;

class A{
    public:
    int a;
    virtual int func(int aa){
        a = aa*10;
        printf("A::func() called\n");
        return a;
    }
    void printa(){
        cout <<"A::printa() a=" <<  a << endl;
    }
};

class B:public A{
    public:
    int func(int bb){
        a = bb;
        printf("B::func() called\n");
        return a;
    }
};

void imshow32f(const std::string &winname, const cv::Mat &img, float alpha = 1.0){
    Mat img_show;
    img.convertTo(img_show, CV_8U, alpha);
    imshow(winname, img_show);
}

void imshow32mp(const std::string &winname, const cv::Mat &img, float alpha = 1.0){

}


void testmpslam(){
	UndistortPAL *ump = nullptr;
    ump = new UndistortPAL(3);
    ump->loadPhotometricCalibration("", "", "");
    Mat raw_img2 = imread("02200.png", 0);
    Mat raw_img;
    raw_img2.convertTo(raw_img, CV_32FC1);

    Mat remapX = Mat(ump->h, ump->w, CV_32FC1, ump->remapX);
    Mat remapY = Mat(ump->h, ump->w, CV_32FC1, ump->remapY);

    Mat mpimg(ump->h, ump->w, CV_32FC1);

    remap(raw_img, mpimg, remapX, remapY, CV_INTER_LINEAR, BORDER_CONSTANT, 0);
    vector<Mat> mpimgs;
    for(int i=0; i<4; i++){
        Mat img = mpimg.colRange(i*pal_model_g->mp_width, (i+1)*pal_model_g->mp_width - 1);
        mpimgs.push_back(img);
        imshow32f("img", img);
        waitKey();
    }
    
    imshow32f("img_raw", raw_img);
    waitKey();
    imshow32f("img", mpimg);
    waitKey();
}

int main(void){
    pal_init("/home/hwj23/Dataset/PAL/calib_results_real.txt"); 


    // // flip
    // Mat img = imread("/home/hwj23/Dataset/PAL/real/s6/images/00000.png");
    // flip(img, img, 0);
    // imshow("img", img);
    // waitKey();
    // return -1;


    // 测试mp去畸变
    testmpslam();

    // ImageAndExposure *mp = ump->undistort<unsigned char>(mpimg, 1.0, 1.0);
    // Mat mpcv = IOWrap::getOCVImg_tem(mp->image, mp->w, mp->h);

    // // test SE3 * Vec
    // Sim3f trans;
    // Vector6f se3;
    // // TODO: se3的位移不是真的位移,还要乘以一个J
    // se3 << 0, 0, 0, M_PI/2, 0, 0;
    // // trans = SE3f::exp(se3);
    // trans.setRotationMatrix(AngleAxisf(M_PI/2, Vector3f::UnitX()).matrix());
    // trans.translation() = Vector3f(1, 1, 1);
    // trans.setScale(0.5);
    // Sim3f transinv = trans.inverse();
    // Vector3f p(1, 2, 3);
    // Vector3f p2 = transinv * p;
    // cout<< p2 << endl;
    // return -1;
    
    // calc dso coord to mk coord
    // Matrix3f R_dso2mk = AngleAxisf(0, Vector3f::UnitY()).matrix();
    // Vector3f t_dso2mk(1, 2, 3);
    // Sim3f result;
    // for(int i=0; i<10; i++){
    //     Matrix3f Rdso, Rmk;
    //     Vector3f tdso, tmk;

    //     Rdso = AngleAxisf(0, Vector3f::UnitY()).matrix(); 
    //     tdso << i*0.1, 1, 2;

    //     Rmk = Rdso * R_dso2mk;
    //     tmk = (0.5 * tdso + t_dso2mk) ;
    //     cout << i << "  dso:";
    //     cout << tdso.T << "   |  mk:" << tmk.T << endl;

    //     int ret = calcWorldCoord(Rdso, tdso, Rmk, tmk, result);
    // }
    
    // return 0;

    // verify pal center
    // auto img = imread("/home/hwj23/Dataset/PAL/real/s6/images/00000.png");
    // // auto img = imread("/home/hwj23/Dataset/PAL/real/s6/images/00590.png");
    // for(int r=3; r<400; r+=30){
    //     circle(img, Point(pal_model_g->yc, pal_model_g->xc), r, {0, 0, 255}, 2);
    // }
    // imshow("raw img", img);
    // waitKey();
    // return -1;

    // resize vignette image 
    // auto img = cv::imread("/home/hwj23/Dataset/TUM/sequence_19/sml_vignette.png", cv::IMREAD_UNCHANGED);
    // int h = 512, w = 640;
    // img = img.colRange(w/2-h/2-1, w/2+h/2-2);
    // cv::resize(img, img, cv::Size(720, 720));
    // cv::imwrite("/home/hwj23/Dataset/TUM/sequence_19/720sml_vignette.png", img);
    // cv::Mat img2 = cv::imread("/home/hwj23/Dataset/TUM/sequence_19/720sml_vignette.png", cv::IMREAD_UNCHANGED);
    // if(img2.type() == CV_16UC1)
    //     printf("OK\n");
    // else{
    //     printf("SB\n");
    //     cout << img.type() << endl;
    // }
    // return -1;


    // // create vignette image for pal 
    // cv::Mat img = cv::Mat::ones(720, 720, CV_16UC1)*60000;
    // imwrite("/home/hwj23/Dataset/PAL/vignette.png", img);
    // return -1;

    // // test pal model
    // float u, v;
    // for(int inc = -200; inc <=200; inc+= 20){
    //     u = pal_model_g->cx + inc;
    //     v = pal_model_g->cy ;
    //     Vector2f uv1(u, v);
    //     cout << uv1.T << " -> ";
    //     Vector3f pt = pal_model_g->cam2world(u, v);
    //     cout << pt.T << " -> ";
    //     auto uv2 = pal_model_g->world2cam(pt);
    //     cout << uv2.T << endl << endl;
    // }
    // return -1;

    // test J -----------------------------
    // Vector3f pt(1, 2, 3);
    // Eigen::Matrix<float, 2, 6> dx2dSE;
    // Eigen::Matrix<float, 2, 3> duv2dxyz;
    // pal_model_g->jacobian_xyz2uv(pt, dx2dSE, duv2dxyz);
    // cout << pt.T << endl;
    // auto p = pal_model_g->world2cam(pt);
    // cout << p.T << endl;
    // cout << duv2dxyz << endl << endl;
    // float delta = 0.001;
    // pt << 1+delta, 2, 3;
    // cout << pt.T << endl;
    // auto p2 = pal_model_g->world2cam(pt);
    // cout << p2.T << endl;
    // Vector2f d = (p2-p)/delta;
    // Vector2f diff = d - duv2dxyz.col(0);
    // cout << d.T << endl;
    // cout << diff.T << endl;
    // return -1

    // // dr/dd ---------------------------
    // Vector3f dxyzdd(0, 0, 1);
    // Vector2f duvdd = duv2dxyz * dxyzdd; 
    // cout << pt.T << "\n\n";
    // cout << duv2dxyz << "\n\n";
    // cout << duvdd.transpose() << endl;
}
