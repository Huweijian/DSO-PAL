#include <util/pal_model.h>
#include <util/pal_interface.h>
#include <util/Undistort.h>
#include <OptimizationBackend/MatrixAccumulators.h>
#include <line/lsd.h>
#include <line/line_estimate.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <ceres/ceres.h>
#include <ceres/solver.h>
#include <ceres/rotation.h>
#include <ceres/cost_function.h>

#include <vector>
#include <iostream>
#include <string>


#include "tic_toc.h"
// #include <opencv2/ximageproc.hpp>

#define TT transpose()
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace dso;
using namespace Sophus;

class A
{
public:
    int a;
    virtual int func(int aa)
    {
        a = aa * 10;
        printf("A::func() called\n");
        return a;
    }
    void printa()
    {
        cout << "A::printa() a=" << a << endl;
    }
};

class B : public A
{
public:
    int func(int bb)
    {
        a = bb;
        printf("B::func() called\n");
        return a;
    }
};

void imshow32f(const std::string &winname, const cv::Mat &img, float alpha = 1.0)
{
    Mat img_show;
    img.convertTo(img_show, CV_8U, alpha);
    imshow(winname, img_show);
}

void imshow32mp(const std::string &winname, const cv::Mat &img, float alpha = 1.0)
{
}


const int param_line_pos_thres = 3;

struct Line
{
    enum Position{
        COMMON,
        LEFT,
        RIGHT,
    };
    float x1, y1, x2, y2;
    float length;
    float angle;
    Position posflag;
    bool isMerged;
};

inline double distSq_euc(const double x1, const double y1, const double x2, const double y2) {
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

inline double dist_euc(const double x1, const double y1, const double x2, const double y2) {
    return sqrt(distSq_euc(x1, y1, x2, y2));
}

// 尝试把编号为il的直线变成长直线
void expandLine(vector<Line> &lines, int il, Line &long_edge_lines){
    auto l = lines[il]; 
    assert(l.posflag != Line::COMMON);
    float angle_th = 3;
    float merge_dist_th = 2;

    // 根据斜率,筛选可能融合的直线
    vector<Line> ready_merge; 
    ready_merge.reserve(lines.size());
    for(int i=0; i<lines.size(); i++){
        if(i == il) continue;

        float angle_d = abs(l.angle - lines[i].angle);
        if(angle_d < angle_th || (180 - angle_d) < angle_th){
            ready_merge.emplace_back(lines[i]);
        }
    }

    // 端点排序
    if(l.posflag == Line::LEFT){
        std::sort(ready_merge.begin(), ready_merge.end(), [](const Line &lhs, const Line &rhs){return lhs.x1 < rhs.x1;});
    }
    else{
        std::sort(ready_merge.begin(), ready_merge.end(), [](const Line &lhs, const Line &rhs){return lhs.x2 > rhs.x2;});
    }

    // 依次尝试融合
    for(int i=0; i<ready_merge.size(); i++){
        if(l.posflag == Line::LEFT){
            double dist = dist_euc(l.x2, l.y2, ready_merge[i].x1, ready_merge[i].y2);
            if(dist < merge_dist_th){
                // TODO 先别着急写直线融合,先看看LSD对直线的refine需不需要修
            }
        }
    }

}

void testmpslam()
{
    UndistortPAL *ump = nullptr;
    ump = new UndistortPAL(3);
    ump->loadPhotometricCalibration("", "", "");
    Mat raw_img = imread("02200.png", 0);
    raw_img.convertTo(raw_img, CV_32FC1);

    Mat remapX = Mat(ump->h, ump->w, CV_32FC1, ump->remapX);
    Mat remapY = Mat(ump->h, ump->w, CV_32FC1, ump->remapY);
    Mat mpimg(ump->h, ump->w, CV_32FC1);
    remap(raw_img, mpimg, remapX, remapY, CV_INTER_LINEAR, BORDER_CONSTANT, 0);

    // convert map
    // Mat dstmap1, dstmap2;
    // convertMaps(remapX, remapY, dstmap1, dstmap2, CV_32FC1);
    // cout << dstmap1.size() << " " << dstmap2.size() <<endl;

    // auto lsd = new LineSegmentDetectorMy(cv::LSD_REFINE_ADV, 1, 0.6, 1, 20);
    auto lsd = new LineSegmentDetectorMy(cv::LSD_REFINE_NONE, 1, 0.6, 1.0, 22.5, 0, 0.5);
    // auto lsd = createLineSegmentDetector(0, 1, 0.6, 1, 45);

    vector<Mat> mpimg_showv;
    vector<vector<cv::Vec4f>> linesv4(4);
    vector<vector<Line>> lines(4);
    for (int i = 0; i < 4; i++)
    {
        Mat img = mpimg.colRange(i * pal_model_g->mp_width, (i + 1) * pal_model_g->mp_width - 1);
        img = img.rowRange(0, img.rows - 20);
        img.convertTo(img, CV_8UC1);
        imwrite(to_string(i)+".png", img);

        lsd->detect(img, linesv4[i]);
        int width = img.cols, height = img.rows;

        for (auto &l : linesv4[i]) {
            float x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
            if (x1 > x2) {
                std::swap(x1, x2);
                std::swap(y1, y2);
            }

            float len = std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
            float angle = std::atan2(y2 - y1, x2 - x1) / M_PI * 180.0f;
            Line::Position pos = Line::Position::COMMON;
            if (x1 < param_line_pos_thres) {
                pos = Line::Position::LEFT;
            } else if (x2 > width - 1 - param_line_pos_thres) {
                pos = Line::Position::RIGHT;
            }

            Line ll{x1, y1, x2, y2, len, angle, pos, false};
            lines[i].emplace_back(ll);

        }

        Mat img_show;
        cv::cvtColor(img, img_show, COLOR_GRAY2RGB);
        if(!linesv4[i].empty())
            lsd->drawSegments(img_show, linesv4[i]);

        // imshow("img", img_show);
        // moveWindow("img", 100, 100);
        // imshow("angle_rgb", lsd->angle_rgb_);
        // moveWindow("angle_rgb", 100*2+img.cols, 100);
        // imshow("angle_d2", lsd->angle_degd2_);
        // moveWindow("angle_d2", 100*2+img.cols, 100*2 + img.rows);
        // imshow("valid_grad", lsd->valid_grad_);
        // moveWindow("valid_grad", 100*3+img.cols*2, 100);

        waitKey(1);
        mpimg_showv.push_back(img_show);
    }

    // TODO: 融合当前pinhole中靠边的直线
    // 1. 角度排序 
    // 2. 判断边沿直线和相邻角度有无合并可能性
    // 直接上2,先不考虑效率
    vector<vector<Line>> long_edge_lines(4);
    for (int i = 0; i < 4; i++) {
        auto cur_lines = lines[i];
        for(int il=0, sz = cur_lines.size(); il<sz; il++){
            auto &l = cur_lines[il];
            if(l.isMerged == false && l.posflag != Line::Position::COMMON){
                Line long_line;
                expandLine(cur_lines, il, long_line); 
                long_edge_lines[i].push_back(long_line);
            }
        }
    }

    Mat mpimg_show;
    hconcat(mpimg_showv, mpimg_show);
    imshow("mp", mpimg_show);
    waitKey();
}

struct ACOSReprojectError{
public:
    ACOSReprojectError(double gx, double gy)
        :gx_(gx){
            gy_ = gy;
        };

    template<typename T> 
    bool operator()(const T* const pose, T* residuals) const {
        T theta = pose[0]*3.1415/180.0;
        T x2 = ceres::cos(theta);
        T y2 = ceres::sin(theta);
        
        T v1dotv2 = T(gx_)*x2 + T(gy_)*y2;
        if((v1dotv2) >= T(1.0) || v1dotv2 <= T(-1.0)){
            residuals[0] = T(0);
        }
        else
        {
            residuals[0] = ceres::acos(v1dotv2);
        }

        // cout <<"res: "<< residuals[0] <<"\tv1.v2: " << v1dotv2 << endl;
        return true;
    }

private:
    const double gx_ = 0;
    double gy_;
};

cv::Mat cost_img1;
cv::Mat cost_img2;
class ImageCost : public ceres::SizedCostFunction<1, 2> {
    public:
    ImageCost(cv::Mat* img, double v1):img(img), v1(v1){};
    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const {
        double x = parameters[0][0];
        double y = parameters[0][1];
        int ix = (int)x;
        int iy = (int)y;
        float dx = x - ix;
        float dy = y - iy;
        float dxdy = dx * dy;
        if (x < 1 || x > cost_img2.cols - 2 || y < 1 || y > cost_img2.rows - 2) {
            return false;
        }
        double v2 = dxdy * cost_img2.at<ushort>(iy + 1, ix + 1) + (dy - dxdy) * cost_img2.at<ushort>(iy + 1, ix) + (dx - dxdy) * cost_img2.at<ushort>(iy, ix + 1) + (1 - dx - dy + dxdy) * cost_img2.at<ushort>(iy, ix);
        residuals[0] = v2 - v1;

        if (jacobians != nullptr) {
            jacobians[0][1] = cost_img2.at<ushort>(iy+1, ix) -cost_img2.at<ushort>(iy-1, ix) ;
            jacobians[0][0] = cost_img2.at<ushort>(iy, ix+1) -cost_img2.at<ushort>(iy, ix-1) ;
        }

        printf(" p2=(%.2f, %.2f) \t%.2f-%.2f = \t%.2f(cost)\n", x, y, v2, v1, residuals[0]);
        return true;
    }
    cv::Mat* img;
    double v1;
};

struct AllCost {
    AllCost(cv::Mat *img, cv::Point2f &pt, double v1)
        : image_cost_(new ImageCost(img, v1)), pt(pt) {};

    template <typename T>
    bool operator()(const T *trans,
                    T *residual) const {
        
        T p2[2];
        p2[0] = T(pt.x) + trans[0];
        p2[1] = T(pt.y) + trans[1];

        // Note that we call image_cost_, just like it was
        // any other templated functor.
        cout << "[T = " << trans[0]  << " " << trans[1]<< "  ] ";
        return image_cost_(p2, residual);
    }

    private:
    double v1;
    cv::Point2f pt;
    ceres::CostFunctionToFunctor<1, 2> image_cost_;
};

int main(int argc, char** argv)
{
    // pal_init("/home/hwj23/Dataset/PAL/calib_results_real.txt");

    // test Ceres-Solver mix diff
    {
        using namespace ceres;
        google::InitGoogleLogging("ceres_tttttttest");
        cost_img1 = Mat::zeros(201, 201, CV_16UC1);
        cost_img2 = Mat::zeros(201, 201, CV_16UC1);
        for(int i=0; i<201; i++){
            for(int j=0; j<201; j++){
                float ctr[2] = {100, 100};
                int val = (i-ctr[0])*(i-ctr[0]) + (j-ctr[1])*(j-ctr[1]);
                cost_img1.at<ushort>(i, j) = val;

                ctr[0] = 103.222; ctr[1] = 93;
                val = (i-ctr[0])*(i-ctr[0]) + (j-ctr[1])*(j-ctr[1]);
                cost_img2.at<ushort>(i, j) = val;
            }
        }
        imshow("img1", cost_img1);
        imshow("img2", cost_img2);
        printf("start\n");
        waitKey(1);
            
        double a_init[2] = {0.0, 0.0};
        double a_val[2] = {a_init[0], a_init[1]};

        Problem problem;
        for(int i=0; i<10; i++){
            int x = 30+ rand() % 140;
            int y = 30+ rand() % 140;
            printf("raw data: (%d, %d)\n", x, y);
            double v1 = cost_img1.at<ushort>(y, x);
            cv::Point2f p2(x, y);
            CostFunction *cost_function =
                new AutoDiffCostFunction<AllCost, 1, 2>(
                    new AllCost(&cost_img2, p2, v1));

            problem.AddResidualBlock(cost_function, NULL, a_val);
        }

        // a_val[0] = 3.222; a_val[1] =  3.222;
        // double cost;
        // problem.Evaluate(Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
        // cout << cost << endl; return 0;


        Solver::Options options;
        options.linear_solver_type = DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        Solver::Summary summary;

        ceres::Solve(options, &problem, &summary);

        std::cout << summary.BriefReport() << "\n";
        printf("x: (%.2f, %.2f) -> (%.2f, %.2f)\n", a_init[0], a_init[1], a_val[0], a_val[1]);
        return 0;
    }

    // test Ceres-Solver
    // {
    //     using namespace ceres;
    //     google::InitGoogleLogging("ceres_tttttttest");
            
    //     double x_init = 20.0;
    //     double x_val = x_init;

    //     Problem problem;

    //     ifstream file("/home/hwj23/Project/dso-master/matlab/line/2d_grad_simu.txt");
    //     float gx, gy;
    //     while(file >> gx >> gy){
    //         float g_norm = std::sqrt(gx*gx + gy*gy);
    //         gx /= g_norm; gy/= g_norm;
    //         CostFunction* cost_function =
    //             new AutoDiffCostFunction<ACOSReprojectError, 1, 1>(
    //                 new ACOSReprojectError(gx, gy));
            

    //         problem.AddResidualBlock(cost_function, NULL, &x_val);
    //         double resi = 0;
    //         double* param[] = {&x_val};

    //         for(int i=0; i<80; i+=5){
    //             x_val = i;
    //             double resi = 0;
    //             cost_function->Evaluate(param, &resi, nullptr);
    //             printf("param=%.2f, resi=%.4f\n", x_val, resi);
    //         }

    //         cost_function->Evaluate(param, &resi, nullptr);
    //         printf("resi = %.2f [ %.2f ], param = %.2f, data_theta = %.2f \n", resi, -x_val/180*3.14159 + atan2(gy, gx), x_val/180*3.14, atan2(gy, gx));
    //     }

    //     Solver::Options options;
    //     options.linear_solver_type = DENSE_QR;
    //     options.minimizer_progress_to_stdout = true;
    //     Solver::Summary summary;

    //     ceres::Solve(options, &problem, &summary);

    //     std::cout << summary.BriefReport() << "\n";
    //     std::cout << "x : " << x_init << " -> " << x_val << "\n";
    // }
    
    // // test RANSAC line estimation
    // ifstream fin("/home/hwj23/Project/dso-master/matlab/line/data1.txt");
    // vector<Eigen::Vector3f> pts;
    // float x, y, z;
    // while(fin >> x >> y >> z){
    //     pts.push_back(Vector3f(x, y, z)); 
    // }
    // Eigen::MatrixXf pts_mat(pts.size(), 3);
    // for(int i=0; i<pts.size(); i++){
    //     pts_mat.block<1, 3>(i, 0) = pts[i];
    // }
    // Eigen::Vector3f x0, u;
    // float inlier = dso_line::line_estimate_g(pts_mat, x0, u);
    // cout << "x0: " << x0.T << endl;
    // cout << "u:  " << u.T << endl;
    // printf("inlier: %.2f%%\n", inlier*100);

    // test Eigen
    // Matrix3f R ;
    // R << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    // Eigen::Quaternion<float> RQ(R);
    // printf("%.2f %.2f %.2f %.2f \n", RQ.x(), RQ.y(), RQ.z(), RQ.w()); 
    // cout << RQ.vec().transpose() <<"   " <<  RQ.w << endl;

    // Vector3f a, b;
    // a << 1, 2, 3;
    // auto arr = a.data();
    // Matrix3f m;
    // m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    // cout << m(0, 0);


    // // flip image
    // Mat img = imread("/home/hwj23/Dataset/PAL/real/s6/images/00000.png");
    // flip(img, img, 0);
    // imshow("img", img);
    // waitKey();
    // return -1;

    // 测试mp去畸变
    // testmpslam();

    // ImageAndExposure *mp = ump->undistort<unsigned char>(mpimg, 1.0, 1.0);
    // Mat mpcv = IOWrap::getOCVImg_tem(mp->image, mp->w, mp->h);

    // // test SE3 * Vec
    // Sim3f trans;
    // Vector6f se3;
    // // se3的位移不是真的位移,还要乘以一个J
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
