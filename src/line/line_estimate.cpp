// ceres必须放前面,因为其中定义了vector<Eigen> 的偏特化
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "line_estimate.h"
#include "line_base.h"
#include "FullSystem/CoarseTracker.h"
#include "FullSystem/HessianBlocks.h"

#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace Eigen;

namespace dso_line{
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
            // printf("*");
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

class ImageCost : public ceres::SizedCostFunction<1, 4> {
    public:
    ImageCost(
            // const Eigen::Vector3f* img, 
            const cv::Mat* img, 
            const Eigen::Vector2i &wh)
    :img(img), wh(wh){
        rows = wh(1);
        cols = wh(0);
    };

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const {
        using namespace cv;
        double x = parameters[0][0];
        double y = parameters[0][1];
        double lx = parameters[0][2];
        double ly = parameters[0][3];

        if (x < 1 || x > cols - 2 || y < 1 || y > rows - 2) {
            // printf("eval() failed! p=(%.2f, %.2f)\n", x, y);
            return false;
        }
        // printf("eval() succes! p=(%.2f, %.2f)\n", x, y);

        int ix = (int)x;
        int iy = (int)y;
        float dx = x - ix;
        float dy = y - iy;
        float dxdy = dx * dy;
        
        // const Eigen::Vector3f* bp = img +ix+iy*cols;
        // Eigen::Vector3f hitpt = dxdy * *(const Eigen::Vector3f*)(bp+1+cols)
	    //     + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+cols)
	    //     + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)
		// 	+ (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);
        // double gx = hitpt(1), gy = hitpt(2); 

        // printf(" - pt=(%.2f, %.2f)  [%.2f %.2f %.2f %.2f] \n", x, y, dxdy, (dy-dxdy), (dx-dxdy), (1-dx-dy+dxdy));
        // printf(" - (%.2f, %.2f, %.2f)\n", (*(bp+1+cols))(0), (*(bp+1+cols))(1),(*(bp+1+cols))(2));
        // printf(" - (%.2f, %.2f, %.2f)\n", (*(bp+cols))(0), (*(bp+cols))(1),(*(bp+cols))(2));
        // printf(" - (%.2f, %.2f, %.2f)\n", (*(bp+1))(0), (*(bp+1))(1),(*(bp+1))(2));
        // printf(" - (%.2f, %.2f, %.2f)\n", (*(bp))(0), (*(bp))(1),(*(bp))(2));

        cv::Vec2f hitpt =  dxdy * (*img).at<cv::Vec2f>(iy+1, ix+1)
	        + (dy-dxdy) * (*img).at<cv::Vec2f>(iy+1, ix)
	        + (dx-dxdy) * (*img).at<cv::Vec2f>(iy, ix+1) 
			+ (1-dx-dy+dxdy) * (*img).at<cv::Vec2f>(iy, ix);
        double gx = hitpt[0], gy = hitpt[1];

        double grad_len_2 = (gx*gx + gy*gy);

        residuals[0] = (gx*lx + gy*ly) / (grad_len_2 + 0.000001);

        // grad
        if (jacobians != nullptr) {
            // double dgx_dx = 0.5 * ((*(bp+1))[1]     - (*(bp-1))[1])     );
            // double dgx_dy = 0.5 * ((*(bp+cols))[1]  - (*(bp-cols))[1])  );
            // double dgy_dx = 0.5 * ((*(bp+1))[2]     - (*(bp-1))[2])     );
            // double dgy_dy = 0.5 * ((*(bp+cols))[2]  - (*(bp-cols))[2])  );

            double dgx_dx = 0.5 * (img->at<Vec2f>(iy, ix+1)[0] - img->at<Vec2f>(iy, ix-1)[0]);
            double dgx_dy = 0.5 * (img->at<Vec2f>(iy+1, ix)[0] - img->at<Vec2f>(iy-1, ix)[0]);
            double dgy_dx = 0.5 * (img->at<Vec2f>(iy, ix+1)[1] - img->at<Vec2f>(iy, ix-1)[1]);
            double dgy_dy = 0.5 * (img->at<Vec2f>(iy+1, ix)[1] - img->at<Vec2f>(iy-1, ix)[1]);

            double gx2 = gx*gx;
            double gy2 = gy*gy;


            double down = gy2*gy2 + gx2*gx2 + 2*gx2*gy2;
            double dgx = (lx*gy2 - 2*ly*gx*gy - lx*gx2) / down;
            double dgy = -(ly*gy2 + 2*lx*gx*gy - ly*gx2) / down;

            jacobians[0][0] = dgx*dgx_dx + dgy*dgy_dx;
            jacobians[0][1] = dgx*dgx_dy + dgy*dgy_dy;
            jacobians[0][2] = gx / grad_len_2;
            jacobians[0][3] = gy / grad_len_2;
        }


        // printf(" grad=( %.2f, %.2f) \t line=(%.2f, %.2f) \t cost=%.2f\n", hitpt(1), hitpt(2), lx, ly, residuals[0]);
        return true;
    }
    int rows = 0, cols = 0;
    // const Eigen::Vector3f* img;
    const cv::Mat* img;
    const Eigen::Vector2i &wh;
};

struct LineReprojectError {
   public:
    LineReprojectError(
            const Eigen::Vector3f* line_pt[3], 
            float camera_param[4], 
            // const Eigen::Vector3f* img, 
            const cv::Mat* img, 
            const Eigen::Vector2i &wh)
    : image_cost_(new ImageCost(img, wh))
    {
        for (int i = 0; i < 3; i++) {
            L_p0[i] = (*line_pt[0])[i];
            L_p1[i] = (*line_pt[1])[i];
            L_p2[i] = (*line_pt[2])[i];
        }
        for (int i = 0; i < 4; i++) {
            camera[i] = camera_param[i];
        }
    };

    // 待估计变量:R t
    // 残差:res = ADIFF( dir(l) - grad(p2))
    // l = proj(L, R, t)
    // p2 = proj(P, R, t)
    template <typename T>
    bool operator()(const T *const pose, T *residuals) const {
        // 3D直线上的3D点(左右两个端点+直线上的点),变量本地化
        T L_pt[3][3]; 
        for(int i=0; i<3; i++){
            L_pt[0][i] = T(L_p0[i]);
            L_pt[1][i] = T(L_p1[i]); 
            L_pt[2][i] = T(L_p2[i]); 
        }

        // 旋转位移
        T l_p[3][3];
        for(int i=0; i<3; i++){
            ceres::AngleAxisRotatePoint(pose, L_pt[i], l_p[i]);
            for (int j = 0; j < 3; j++) {
                if( j != 0)
                    l_p[i][j] = l_p[i][j] + pose[3+j];
            }
        }

        // 计算2D直线方向
        T lx = l_p[1][0] / l_p[1][2] - l_p[0][0] / l_p[0][2];
        T ly = l_p[1][1] / l_p[1][2] - l_p[0][1] / l_p[0][2];
        T l_norm = ceres::sqrt(lx * lx + ly * ly);// 归一化似乎用处不看
        lx /= l_norm;
        ly /= l_norm;

        // 另一个点投影到图像上
        T l_pt[2];
        l_pt[0] = l_p[2][0] / l_p[2][2] * camera[0] + camera[2];
        l_pt[1] = l_p[2][1] / l_p[2][2] * camera[1] + camera[3];

        T params[4] = {l_pt[0], l_pt[1], lx, ly};
        // cout << "line_PT3 = " << L_pt[2][0] << " " << L_pt[2][1] << " " << L_pt[2][2] << endl;
        // cout <<"Pose.R = (" << pose[0] << ", " << pose[1] << ", "<< pose[2] << ", "<< pose[3] << "); Pose.T = "<< pose[4] <<" " << pose[5] << " " << pose[6] << endl;
        // cout << "Transformed line_PT3 = " << l_p[2][0] << " " << l_p[2][1] << " " << l_p[2][2] << endl;
        // printf("camera = [%.2f %.2f %.2f %.2f]\n", camera[0], camera[1],camera[2],camera[3]);
        // cout << "line_pt2 = " << l_pt[0] << " " << l_pt[1] << "\t";

        return image_cost_(params, residuals);
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3f &l_x0, const Eigen::Vector3f &l_u,
        const Eigen::Vector3f &l_pt,
        const Eigen::Matrix3f &K,
        // const Eigen::Vector3f *img,
        const cv::Mat *img,
        const Eigen::Vector2i &wh) {
        float camera[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
        Eigen::Vector3f l_x1 = l_x0 + l_u;
        const Eigen::Vector3f * line_pt[3] = {&l_x0, &l_x1, &l_pt};

        LineReprojectError *err = new LineReprojectError(line_pt, camera, img, wh);

        return (new ceres::AutoDiffCostFunction<LineReprojectError, 1, 6>(err));
    }

   private:
    double L_p0[3];
    double L_p1[3];
    double L_p2[3];
    double camera[4];  // fx, fy, cx, cy
    ceres::CostFunctionToFunctor<1, 4> image_cost_;
};
}

namespace dso{

	void CoarseTracker::refinePose(SE3 &refToNew){
        using namespace ceres;
        using namespace dso_line; 
        using namespace cv;

        // 本地化变量
        int lvl = 0;
        int nl = pc_n[lvl];
        int wl = w[lvl];
        int hl = h[lvl];
        int sizel[2] = {wl, hl};
        int line_num = lastRef->line_u.size();
        Mat33f R = refToNew.rotationMatrix().cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        Mat33f K; K << fx[lvl], 0, cx[lvl], 0, fy[lvl], cy[lvl], 0, 0, 1;
        Vec3f* dIp_new = newFrame->dIp[lvl];
        vector<Vector3f> sample_pts;
        cv::Mat img_new = IOWrap::getOCVImg(newFrame->dI, wl, hl);

        // [debug] use Mat to represent grads map 
        cv::Mat gradx = IOWrap::getOCVImg_float(dIp_new, wl, hl, 1);
        cv::Mat grady = IOWrap::getOCVImg_float(dIp_new, wl, hl, 2);
        cv::Mat grads;
        cv::merge(vector<cv::Mat>{gradx, grady}, grads);

        // TODO!!! 在这里模糊梯度图,增大感受野,试试效果.
        Mat gradx2 = gradx.clone();
        resize(gradx2, gradx2, Size(), 0.5, 0.5);
        boxFilter(gradx2, gradx2, -1, cv::Size(3, 3));
        cout << gradx2 << endl;

        cv::Mat gradx_show, grady_show;    
        gradx2.convertTo(gradx_show, CV_8UC1, 1.0, 60.0);
        grady.convertTo(grady_show, CV_8UC1, 1.0, 60.0);

        imshow("grad_x", gradx_show);
        imshow("grad_y", grady_show);
        // ---------------------------------

        // 优化初值
        double pose_val_init[6] = {0, 0, 0, 0, 0, 0};
        double pose_val[6] = {0, 0, 0, 0, 0, 0};

        // 添加残差项
        Problem problem;
        for(int i = 0; i<line_num; i++){
            // [1]. 3D直线的方程,从世界坐标系转换到当前坐标系
            Vector3f P[2];
            P[0] = R * lastRef->line_x0[i] + t;
            P[1] = R * (lastRef->line_x0[i] + lastRef->line_u[i]) + t;
            Vector3f U = P[1] - P[0];


            // [2]. 找到当前视场下的3D直线段
            // 相机FoV的3D平面
            Vector3f pt[4];
            pt[0] = Vector3f(-cx[lvl] /fx[lvl], -cy[lvl]/fy[lvl], 1);
            pt[1] = Vector3f(-cx[lvl] /fx[lvl], +cy[lvl]/fy[lvl], 1);
            pt[2] = Vector3f(+cx[lvl] /fx[lvl], +cy[lvl]/fy[lvl], 1);
            pt[3] = Vector3f(+cx[lvl] /fx[lvl], -cy[lvl]/fy[lvl], 1);
            Vector4f plane[4];
            for(int i=0; i<4; i++){
                Vector3f &x1=pt[i], &x2=pt[(i+1)%4], x3(0, 0, 0);
                plane[i].block<3, 1>(0, 0) = (x1-x3).cross(x2-x3);
                plane[i](3) = (-x3.transpose()) * (x1.cross(x2));
            }

            // 3D直线投影到2D
            Vector3f Pt[4]; int Pt_cnt = 0;
            Vec2f line2d_x0, line2d_u;
            line3d_to_2d(lastRef->line_x0[i], lastRef->line_u[i], line2d_x0, line2d_u, R, t, K);

            // 找到2D直线在FoV边界点对应的3D点
            Vector2f pt_pixel[4];
            pt_pixel[0] = Vector2f(0, 0);
            pt_pixel[1] = Vector2f(0, hl);
            pt_pixel[2] = Vector2f(wl, hl);
            pt_pixel[3] = Vector2f(wl, 0);
            Matrix4f L = plucker_l(Line{&P[0], &U});
            for(int i=0; i<4; i++){
                Vector2f p1 = pt_pixel[i]-line2d_x0, p2 = pt_pixel[(i+1)%4]-line2d_x0;
                auto &u = line2d_u;
                if((u[0]*p1[1] - u[1]*p1[0]) * (u[0]*p2[1] - u[1]*p2[0]) <= 0){
                    Vector4f Pt4 = L * plane[i]; 
                    Pt[Pt_cnt++] = Pt4.segment<3>(0) / Pt4(3);
                }
            }
            assert(Pt_cnt == 2);

            // [3]. 在FoV内的3D直线段上均匀选择N个点,投影到2D,加入优化框架
            Vector3f Pt_dir = Pt[1] - Pt[0];
            const int NUM_SAMPLE_PT = 50;
            for (int p = 1; p <= NUM_SAMPLE_PT; p++) {
                Vector3f Pt_cur = Pt[0] + Pt_dir / (NUM_SAMPLE_PT+1)*p;
                CostFunction *cost_function = LineReprojectError::Create(
                    lastRef->line_x0[i], lastRef->line_u[i], Pt_cur,
                    K, &grads, Vector2i(wl, hl));
                Vector3f pt_cur = K * Pt_cur;
                pt_cur /= pt_cur(2);
                sample_pts.push_back(pt_cur);

                // cout << "line 3D: (" << lastRef->line_x0[i].transpose() << ")\t(" << lastRef->line_u[i].transpose() << ")" << endl;
                // cout << "line 2D: (" << line2d_x0.transpose() << ")\t(" << line2d_u.transpose() << ")" << endl;
                // printf("\t main dir : [ %s ]\n", l_dir[0] == 0 ? "--" : "|");

                // // debug single residual block ---------------------
                // if (p == 10) {
                //     using namespace cv;
                //     // 在这里调节pose,看看单个残差的效果
                //     double cost = 0.0;
                //     double *param[] = {pose_val};
                //     bool ret = cost_function->Evaluate(param, &cost, NULL);
                //     printf(" - cost(%.4f)\n ",ret ? 90 - std::acos(std::abs(cost)) / M_PI * 180 : -9999.9999);

                //     cv::Mat img_nd = img_new.clone();
                //     Vec3f pt_cur = K * Pt_cur;
                //     pt_cur /= pt_cur(2);
                //     cv::drawMarker(img_nd, cv::Point(pt_cur(0), pt_cur(1)), 255, MARKER_SQUARE);
                //     float gra[2] = {newFrame->dIp[lvl][(int)(pt_cur(0)+pt_cur(1)*wl)](1), newFrame->dIp[lvl][(int)(pt_cur(0)+pt_cur(1)*wl)](2)};
                //     cv::line(img_nd, Point(pt_cur(0), pt_cur(1)), Point(pt_cur(0) + gra[0], pt_cur(1) + gra[1]), 255);
                //     imshow("debug_single_residuals", img_nd);
                //     waitKey(0);
                // }
                // ----------------------------------------

                problem.AddResidualBlock(cost_function, new HuberLoss(0.2), pose_val);
            }
        }

        // [debug] only print overall cost, not optimize -------------------
        // {
        //     using namespace cv;
        //     double cost = 0.0;
        //     vector<double> residual;
        //     vector<double> grad;
        //     ceres::CRSMatrix jaco;
        //     bool ret = problem.Evaluate(Problem::EvaluateOptions(), &cost, &residual, &grad, &jaco);
        //     // printf(" - cost=%.8f\n", ret ? cost : -999);
        //     // printf(" - Resi(%lu): ", residual.size());for(auto i : residual) printf("%.2f ", i); printf("\n");
        //     // printf(" - Grad(%lu): ", grad.size());for(auto i : grad) printf("%.2f ", i); printf("\n");
        //     printf(" %.8f \n", ret ? cost : -999);

        //     Mat33f RR = R;
        //     Vec3f tt = t;
        //     Vec2f ll[2];
        //     // pose_to_Rt(pose_val, RR, tt);
        //     line3d_to_2d(lastRef->line_x0[0], lastRef->line_u[0], ll[0], ll[1], RR, tt, K);
        //     Mat img_line = img_new.clone();

        //     // draw_line2d(img_line, ll[0], ll[1]);
        //     for(auto &ptf : sample_pts){
        //         Vector3i pt = ptf.cast<int>();
        //         drawMarker(img_line, Point(pt(0), pt(1)), 255, MARKER_SQUARE, 5);

        //         int ix = pt(0);
        //         int iy = pt(1);
        //         float dx = ptf(0) - ix;
        //         float dy = ptf(1) - iy;
        //         float dxdy = dx * dy;
        //         const Eigen::Vector3f* bp = dIp_new +ix+iy*wl;
        //         Eigen::Vector3f hitpt = dxdy * *(const Eigen::Vector3f*)(bp+1+wl)
        //             + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+wl)
        //             + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)
        //             + (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);
                
        //         // double grad[] = {dIp_new[pt(1)*wl+pt(0)](1), dIp_new[pt(1)*wl+pt(0)](2)};
        //         double grad[] = {hitpt(1), hitpt(2)};
        //         double grad_len = Vector2f(grad[0], grad[1]).norm();

        //         line(img_line, Point(pt(0), pt(1)), Point(pt(0)+grad[0], pt(1)+grad[1]), 255);
        //         // printf("line_pt(%.2f, %.2f) grad(%.2f, %.2f)\n", ptf(0), ptf(1), grad[0], grad[1]);
        //     }
        //     imshow("debug_img", img_line);
        //     moveWindow("debug_img", 100, 100);
        //     cv::waitKey(0);
        //     return;
        // }
        // ------------------

        // 求解优化问题
        Solver::Options options;
        options.minimizer_type = TRUST_REGION;
        options.linear_solver_type = DENSE_QR;
        // options.minimizer_progress_to_stdout = true;
        // options.use_nonmonotonic_steps = true;
        options.max_num_iterations = 100;
        Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        Mat33f R2 = Mat33f::Identity(); 

        Vector3f rotv(pose_val[0], pose_val[1], pose_val[2]);
        if(rotv.norm() > 1e-8)
            R2 = AngleAxisf(rotv.norm(), rotv.normalized()).toRotationMatrix();
        Vec3f t2 = R2 * t + Vec3f(pose_val[3], pose_val[4], pose_val[5]);
        R2 = R2 * R;
        
        // 返回优化结果
        // refToNew.translation() = t2.cast<double>();
        // refToNew.setRotationMatrix(R2.cast<double>());

        // 输出结果
        std::cout << summary.BriefReport() << "\n";
        {
            using namespace cv;

            // printf(" * [%d] residuals\n", problem.NumResidualBlocks());
            printf("\tcost(%s): %.4f -> %.4f \tmsg:%s\n",
                    summary.final_cost<3?"GOOD":" BAD", 
                    summary.initial_cost, summary.final_cost, summary.message.data());

            auto &pv_ = pose_val;
            // printf("\t - Pose1 = (%.4f, %.4f, %.4f),  (%.4f, %.4f, %.4f) \n", pv_[0], pv_[1], pv_[2], pv_[3], pv_[4], pv_[5]);

            Mat img_ref = IOWrap::getOCVImg(lastRef->dI, wG[0], hG[0]);
            Mat img_pt = IOWrap::getOCVImg(newFrame->dI, wG[0], hG[0]);
            Mat img_line = img_pt.clone();
            Vec2f line2d[2];
            for (int i = 0; i < line_num; i++) {
                line3d_to_2d(lastRef->line_x0[i], lastRef->line_u[i], line2d[0], line2d[1], Mat33f::Identity(), Vec3f::Zero(), K);
                draw_line2d(img_ref, line2d[0], line2d[1]);
                line3d_to_2d(lastRef->line_x0[i], lastRef->line_u[i], line2d[0], line2d[1], R, t, K);
                draw_line2d(img_pt, line2d[0], line2d[1]);
                line3d_to_2d(lastRef->line_x0[i], lastRef->line_u[i], line2d[0], line2d[1], R2, t2, K);
                draw_line2d(img_line, line2d[0], line2d[1]);
            }
            // cv::imshow("ref", img_ref);
            cv::imshow("pt_track", img_pt);
            moveWindow("pt_track", 100, 100);
            cv::imshow("pt+line_track", img_line);
            moveWindow("pt+line_track", 100 + 480 + 100, 100);

            cv::waitKey();
        }
    }
}