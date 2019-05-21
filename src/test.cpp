#include <iostream>
#include <string>
#include <Eigen/Core>
#include <util/pal_model.h>
#include <util/pal_interface.h>
#include <OptimizationBackend/MatrixAccumulators.h>
#include <vector>
#include <opencv2/core/eigen.hpp>

#define T transpose()
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace dso;

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


int main(void){
    pal_init("/home/hwj23/Dataset/PAL/calib_results_real.txt"); 

    Matrix4f m; 
    m << 
    0.0972,    0.8549,   -0.5095,    0.2746,
    0.9922,   -0.0430,    0.1172,    0.0331,
    0.0783,   -0.5169,   -0.8524,    0.7169,
         0,         0,         0,    1.0000;
    Matrix4f minv = m.inverse();
    cout << minv << endl;
    return 0;

    // test cv::rodrigues
    Vector3f Rv(0, 3.14159/2, 0);
    Mat Rvcv, Rcv(3, 3, CV_32FC1);
    cv::eigen2cv(Rv, Rvcv);
    Rodrigues(Rvcv, Rcv);
    Matrix3f R;
    cv2eigen(Rcv, R);
    cout<< R << endl;
    

    return 0;

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
