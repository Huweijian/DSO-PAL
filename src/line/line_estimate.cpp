// ceres必须放前面,因为其中定义了vector<Eigen> 的偏特化
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "line_estimate.h"
#include "FullSystem/CoarseTracker.h"
#include "FullSystem/HessianBlocks.h"

#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace Eigen;

namespace dso_line{

bool line3d_to_2d(
        const Eigen::Vector3f &line_x0, const Eigen::Vector3f &line_u, 
        Eigen::Vector2f &line2d_x0, Eigen::Vector2f &line2d_u, 
        const Eigen::Matrix3f &R, const Eigen::Vector3f &t, 
        const Matrix3f &K){
    
    Vector3f P[2];
    P[0] = line_x0;
    P[1] = line_x0 + line_u;
    for(int i=0; i<2; i++){
        P[i] = R * P[i] + t; 
        P[i] /= P[i][2]; 
        P[i] = K * P[i];
    }
    line2d_x0 = P[0].block<2, 1>(0, 0);
    P[1] = P[1] - P[0];
    line2d_u = P[1].block<2, 1>(0, 0);
    line2d_u.normalize();
    }

void draw_line2d(cv::Mat &img, const Eigen::Vector2f &x0, const Eigen::Vector2f &u, unsigned char color = 255) {
    using namespace Eigen;
    int l_dir[2] = {0, 1};  // 直线的主方向和次方向
    int sz[2] = {img.cols, img.rows};

    if (abs(u[1]) > abs(u[0])) {
        std::swap(l_dir[0], l_dir[1]);
    }

    Vector2f uu = u / u[l_dir[0]];               //主方向归一化
    Vector2f p0 = x0 + (0 - x0[l_dir[0]]) * uu;  //直线起点归零
    Vector2f p1 = x0 + (sz[l_dir[0]] - x0[l_dir[0]]) * uu;
    line(img, cv::Point(p0[0], p0[1]), cv::Point(p1[0], p1[1]), color);
    return;
}

void pose_to_Rt(double pose[7], Eigen::Matrix3f &R, Eigen::Vector3f &t){
    R = Eigen::Quaternionf(pose[0], pose[1], pose[2], pose[3]);    
    t = Eigen::Vector3f(pose[4], pose[5], pose[6]);
    return;
}

float line_estimate_g(const Eigen::MatrixXf &points, Eigen::Vector3f &x0_ret, Eigen::Vector3f &u_ret){

    float dist_thres = 0.01;

    // RANSAC 求解   
    int npts = points.rows();
    Vector3f best_x0, best_u;
    int max_inlier = -1;

    for(int i=0; i<1000; i++){
        int r1 = rand() % npts;
        int r2 = rand() % npts;
        while(r2 == r1){
            r2 = rand() % npts;
        }

        // line model (x0+u)
        Vector3f x0 = points.row(r1); 
        Vector3f u = Vector3f(points.row(r2)) - Vector3f(points.row(r1));
        u.normalize();

        int inliner = 0;
        for(int j=0; j<npts; j++){
            Vector3f p = points.row(j);
            Vector3f distv = (x0 - p) + u.dot(p - x0) * u;
            float dist = distv.norm();
            if(dist < dist_thres){
                inliner ++ ;
            }
        }
        if(inliner > max_inlier){
            printf("*");
            max_inlier = inliner;
            best_x0 = x0;
            best_u = u;
        }
        // debug info
        // printf("(%2d) inline:%d/%d ", i, inliner, npts);
        // cout << x0.transpose() << "\t" << u.transpose() << endl;
    }

    x0_ret = best_x0;
    u_ret = best_u;
    return float(max_inlier)/npts;

    // Ceres NLSQ 稍微有点麻烦,比较难表达出残差项 
    // Problem problem;
    // CostFunction* cost_function = new AutoDiffCostFunction<LineResidual, 1, 1>(new LineResidual);
    // double x = 5;
    // problem.AddResidualBlock(cost_function, NULL, &x);
    // Solver::Options options;
    // options.linear_solver_type = DENSE_QR;
    // options.minimizer_progress_to_stdout = true;
    // Solver::Summary summary;
    // Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
}

struct LineReprojectError{
public:

    LineReprojectError(float gx, float gy, float dist, float l_p0[3], float l_p1[3], float camera[4])
        :gx_(gx), gy_(gy), dist_(dist){
            for(int i=0; i<3; i++){
                l_p0_[i] = l_p0[i];
                l_p1_[i] = l_p1[i];
            }
            for(int i=0; i<4; i++){
                camera_[i] = camera[i];
            }
        };

    template<typename T> 
    bool operator()(const T* const pose, T* residuals) const {
        T p0[3], p1[3];
        T l_p0T[3] = {T(l_p0_[0]), T(l_p0_[1]), T(l_p0_[2]) };
        T l_p1T[3] = {T(l_p1_[0]), T(l_p1_[1]), T(l_p1_[2]) };

        ceres::UnitQuaternionRotatePoint(pose, l_p0T, p0);
        ceres::UnitQuaternionRotatePoint(pose, l_p1T, p1);
        for(int i=0; i<3; i++){
            p0[i]+= pose[4+i];  
            p1[i]+= pose[4+i];
        }

        // cout <<"Pose.R = (" << pose[0] << ", " << pose[1] << ", "<< pose[2] << ", "<< pose[3] << "); Pose.T = "<< pose[4] <<" " << pose[5] << " " << pose[6] << endl;
        // cout <<"Line P1= (" << l_p0T[0]  << " " << l_p0T[1] << " " << l_p0T[2] << ") -> ( " << p0[0] << " " << p0[1] << " " << p0[2] << endl;
        // cout <<"Line P2= (" << l_p1T[0]  << " " << l_p1T[1] << " " << l_p1T[2] << ") -> ( " << p1[0] << " " << p1[1] << " " << p1[2] << endl;
        // cout <<" - RotPt = " << p0[0] << ", " << p1[0] << endl;
        T lx = p1[0]/p1[2] - p0[0]/p0[2];
        T ly = p1[1]/p1[2] - p0[1]/p0[2];
        // cout <<" - NormL = " << lx << ", " << ly << endl;

        // 归一化似乎用处不看
        T l_norm = ceres::sqrt(lx*lx + ly*ly);
        lx /= l_norm; ly /= l_norm; 
        
        // test 计算直线角度和像素梯度之差 ---------------------
        // T theta = pose[0]*3.1415/180.0;
        // T x2 = ceres::cos(theta);
        // T y2 = ceres::sin(theta);
        // T dot_res = gx_ * x2 + gy_ * y2;
        // -------------------------------------------------

        T dot_res = lx*gx_+ly*gy_;
        // cout <<" - Res = (" << lx << ", " << ly << ")  *  (" << gx_ << ", " << gy_ << ") = " << dot_res << endl;

        if((dot_res) >= T(1.0) || dot_res <= T(-1.0)){
            residuals[0] = T(0);
        }
        else{
            residuals[0] = (dot_res) / dist_ ;
        }
        return true;
    }

    static ceres::CostFunction *Create(
            float gx, float gy, float dist, 
            const Eigen::Vector3f &l_x0, const Eigen::Vector3f &l_u,
            const Eigen::Matrix3f &K,
            LineReprojectError *err_out = nullptr){

        // double l_p0[3], l_p1[3], 
        float camera[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
        Eigen::Vector3f l_p0 = l_x0;
        Eigen::Vector3f l_p1 = l_p0 + l_u;
        float g_norm = std::sqrt(gx*gx + gy*gy) + 0.00001;
        gx /= g_norm; gy/= g_norm;

        LineReprojectError *err = new LineReprojectError(gx, gy, dist, l_p0.data(), l_p1.data(), camera);
        if(err_out != nullptr){
            err_out = err;
        }

        return (new ceres::AutoDiffCostFunction<LineReprojectError, 1, 7>(err));
    }

   private:
    const double gx_;
    const double gy_;
    const double dist_;
    double l_p0_[3];
    double l_p1_[3];
    double camera_[4];  // fx, fy, cx, cy
};
}

namespace dso{

	void CoarseTracker::testLine(const SE3 &refToNew){
        using namespace ceres;
        using namespace dso_line; 

        // 本地化变量
        int lvl = 0;
        int nl = pc_n[lvl];
        int wl = w[lvl];
        int hl = h[lvl];
        int sizel[2] = {wl, hl};
        int line_num = 1; //lastRef->line_u.size();
        Mat33f R = refToNew.rotationMatrix().cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        Mat33f K; K << fx[lvl], 0, cx[lvl], 0, fy[lvl], cy[lvl], 0, 0, 1;
        Vec3f* dIp_new = newFrame->dIp[lvl];
        cv::Mat img_new = IOWrap::getOCVImg(newFrame->dI, wG[0], hG[0]);

        // 优化初值
        Eigen::Quaternion<float> q(R);
        double pose_val_init[7] = {q.w(), q.x(), q.y(), q.z(), t[0], t[1], t[2]};
        double pose_val[7] = {q.w(), q.x(), q.y(), q.z(), t[0], t[1], t[2]};

        // --debug -----------
        // Mat33f R_d; Vec3f t_d;
        // {
        //     float pose_dy = 0.01;
        //     pose_val[5] = pose_val_init[5] + pose_dy;
        //     pose_to_Rt(pose_val, R_d, t_d);
        // }
        // ---------------

        // auto &pv_ = pose_val;
        // printf(" * Pose = (%.2f, %.2f, %.2f, %.2f),  (%.2f, %.2f, %.2f) \n", pv_[0], pv_[1],pv_[2],pv_[3],pv_[4],pv_[5],pv_[6]);

        // 添加残差项
        Problem problem;
        for(int i = 0; i<line_num; i++){
            Vec2f line2d_x0, line2d_u;
            line3d_to_2d(lastRef->line_x0[i], lastRef->line_u[i], line2d_x0, line2d_u, R, t, K);
            // 尝试把直线附近的点加入到residualblock中
            int l_dir[2] = {0, 1};     // 直线的主方向和次方向
            if(abs(line2d_u[1]) > abs(line2d_u[0])){
                swap(l_dir[0], l_dir[1]);
            }

            line2d_u = line2d_u / line2d_u[l_dir[0]];  //主方向归一化
            line2d_x0 = line2d_x0 + (0 - line2d_x0[l_dir[0]]) * line2d_u;  //直线起点归零

            cout << "line 3D: (" << lastRef->line_x0[i].transpose() << ")\t(" << lastRef->line_u[i].transpose() << ")" << endl;
            cout << "line 2D: (" << line2d_x0.transpose() << ")\t(" << line2d_u.transpose() << ")" << endl;
            printf("\t main dir : [ %s ]\n", l_dir[0] == 0 ? "--" : "|");

            for (int px_idx1 = 3; px_idx1 < sizel[l_dir[0]] - 3; px_idx1+=3) {
                Vec2f px = line2d_x0 + px_idx1 * line2d_u;
                float px_idx2 = px[l_dir[1]];
                if(px_idx2 < 3 && px_idx2 >= sizel[l_dir[1]] - 3){
                    continue;
                }
                float hit_pt[2] = {(float)px_idx1, px_idx2};
                float hit_ptx = hit_pt[l_dir[0]];
                float hit_pty = hit_pt[l_dir[1]];

                // 添加hit点
                const float HIT_PT_DIST = 1;
                Vec3f hit_img = getInterpolatedElement33(dIp_new, hit_ptx, hit_pty, wl);
                CostFunction* cost_function = LineReprojectError::Create(
                        hit_img[1], hit_img[2], HIT_PT_DIST, 
                        lastRef->line_x0[i], lastRef->line_u[i], K);

                // // debug single residual block ---------------------
                // 这里似乎没用,因为不能真正可视化出位姿改变导致的hitpt的改变
                // if(px_idx1 >= 225)
                // {
                //     using namespace cv;
                //     // 在这里调节pose,看看单个残差的效果
                //     for (double dy = -0.5; dy <= 0.5; dy += 0.02) {
                //         double cost = 0.0;
                //         pose_val[4] = pose_val_init[4] + dy;
                //         double *param[] = {pose_val};
                //         float gra[2] = {hit_img[1], hit_img[2]};
                //         bool ret = cost_function->Evaluate(param, &cost, NULL);
                //         printf(" - cost(%.4f) dx(%.2f) dist(%.2f) grad(%.2f, %.2f) idx=%d\n ",
                //                ret ? 90 - std::acos(std::abs(cost)) / M_PI * 180 : -9999.9999, dy, HIT_PT_DIST, hit_img[1], hit_img[2], px_idx1);
                //         cv::Mat img_nd = img_new.clone();
                //         cv::drawMarker(img_nd, cv::Point(hit_ptx, hit_pty), 255, MARKER_SQUARE);
                //         cv::line(img_nd, Point(hit_ptx, hit_pty), Point(hit_ptx + gra[0], hit_pty + gra[1]), 255);
                //         imshow("debug_residuals", img_nd);
                //         waitKey(0);
                //     }
                //     pose_val[4] = pose_val_init[4];
                // }

                // ----------------------------------------


                problem.AddResidualBlock(cost_function, NULL, pose_val);

                // printf("hit pt: (%.2f, %.2f)\n", hit_ptx, hit_pty);
                // printf("%.2f  %.2f \n", hit_img[1], hit_img[2]);

                // 添加周围点
                const int EX_NUM = -1;
                for(int j=-EX_NUM; j<=EX_NUM; j++){
                    if( j != 0){
                        int hit_pti[2] = {(int)hit_ptx, (int)hit_pty};
                        hit_pti[l_dir[1]] += j;
                        // printf("\tpt: (%d, %d)\n", hit_pti[0], hit_pti[1]);
                        int dist = abs(j);
                        Vec3f &px_img = dIp_new[hit_pti[1]*wl + hit_pti[0]];
                        CostFunction* cost_fun = LineReprojectError::Create(
                            px_img[1], px_img[2], dist, 
                            lastRef->line_x0[i], lastRef->line_u[i], K);
                        problem.AddResidualBlock(cost_fun, NULL, pose_val);
                    }
                }

            }
        }

        // // debug overall cost -------------------
        // {
            double cost = 0.0;
            bool ret = problem.Evaluate(Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
            printf(" - cost=%.8f pose=(%.2f, %.2f, %.2f, %.2f | %.2f, %.2f, %.2f)\n", 
                ret ? cost : -999, pose_val[0], pose_val[1], pose_val[2], pose_val[3], pose_val[4], pose_val[5], pose_val[6]);
        //     Mat33f RR; Vec3f tt; Vec2f ll[2];
        //     pose_to_Rt(pose_val, RR, tt);
        //     line3d_to_2d(lastRef->line_x0[0], lastRef->line_u[0], ll[0], ll[1], RR, tt, K); 
        //     cv::Mat img_line = img_new.clone();
        //     draw_line2d(img_line, ll[0], ll[1]);
        //     imshow("debug_img", img_line);
        //     cv::waitKey();
        //     return ;
        // }

        // ------------------

        // 求解优化问题
        Solver::Options options;
        options.linear_solver_type = DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 20;
        Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 输出结果
        std::cout << summary.FullReport() << "\n";
        {
            using namespace cv;
            printf(" * [%d] residuals\n", problem.NumResidualBlocks());
            auto &pvi_ = pose_val_init;
            printf("   Pose0 = (%.3f, %.3f, %.3f, %.3f),  (%.3f, %.3f, %.3f) \n", pvi_[0], pvi_[1],pvi_[2],pvi_[3],pvi_[4],pvi_[5],pvi_[6]);
            auto &pv_ = pose_val;
            printf("   Pose1 = (%.3f, %.3f, %.3f, %.3f),  (%.3f, %.3f, %.3f) \n", pv_[0], pv_[1],pv_[2],pv_[3],pv_[4],pv_[5],pv_[6]);
            Mat33f R2 = Quaternionf(pv_[0], pv_[1], pv_[2], pv_[3]).toRotationMatrix(); 
            Vec3f t2(pv_[4], pv_[5], pv_[6]);
            
            Mat img_ref = IOWrap::getOCVImg(lastRef->dI, wG[0], hG[0]);
            Mat img_pt = IOWrap::getOCVImg(newFrame->dI, wG[0], hG[0]);
            Mat img_line = img_pt.clone();
            Vec2f line2d[2];
            for(int i=0; i<line_num; i++){
                line3d_to_2d(lastRef->line_x0[i], lastRef->line_u[i], line2d[0], line2d[1], Mat33f::Identity(), Vec3f::Zero(), K);
                draw_line2d(img_ref, line2d[0], line2d[1]);
                line3d_to_2d(lastRef->line_x0[i], lastRef->line_u[i], line2d[0], line2d[1], R, t, K);
                draw_line2d(img_pt, line2d[0], line2d[1]);
                line3d_to_2d(lastRef->line_x0[i], lastRef->line_u[i], line2d[0], line2d[1], R2, t2, K);
                draw_line2d(img_line, line2d[0], line2d[1]);
            }
            cv::imshow("ref",img_ref);
            cv::imshow("pt_track",img_pt);
            cv::imshow("pt+line_track",img_line);

            cv::waitKey();
        }
}
}