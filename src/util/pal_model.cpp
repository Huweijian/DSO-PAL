/*
 * pal_camera.cpp
 */
#include "pal_model.h"
#include <iostream>
using namespace std;
using namespace Eigen;
namespace pal
{

PALCamera::PALCamera(std::string filename)
{
    double pol[MAX_POL_LENGTH];
    double invpol[MAX_POL_LENGTH];
    double xc_tem, yc_tem, c, d, e;
    int length_pol, length_invpol;
 
    FILE *f;
    char buf[CMV_MAX_BUF];
    int i;

    //Open file
    if (!(f = fopen(filename.c_str(), "r")))
    {
        printf("PAL calib file %s cannot be opened\n", filename.c_str());
        assert(0);
        return;
    }

    //Read polynomial coefficients
    int wr = 0; // wanring remover
    char *wrc = nullptr;
    wrc= fgets(buf, CMV_MAX_BUF, f);
    wr = fscanf(f, "\n");
    wr = fscanf(f, "%d", &length_pol);
    for (i = 0; i < length_pol; i++)
    {
        wr = fscanf(f, " %lf", &pol[i]);
    }

    //Read inverse polynomial coefficients
    wr = fscanf(f, "\n");
    wrc= fgets(buf, CMV_MAX_BUF, f);
    wr = fscanf(f, "\n");
    wr = fscanf(f, "%d", &length_invpol);
    for (i = 0; i < length_invpol; i++)
    {
        wr = fscanf(f, " %lf", &invpol[i]);
    }

    //Read center coordinates
    wr = fscanf(f, "\n");
    wrc= fgets(buf, CMV_MAX_BUF, f);
    wr = fscanf(f, "\n");
    wr = fscanf(f, "%lf %lf\n", &xc_tem, &yc_tem);

    //Read affine coefficients
    wrc= fgets(buf, CMV_MAX_BUF, f);
    wr = fscanf(f, "\n");
    wr = fscanf(f, "%lf %lf %lf\n", &c, &d, &e);

    //Read image size
    wrc= fgets(buf, CMV_MAX_BUF, f);
    wr = fscanf(f, "\n");
    wr = fscanf(f, "%lf %lf %lf\n", &this->in_height, &this->in_width, &this->height_);

    // mask size
    wrc= fgets(buf, CMV_MAX_BUF,f);
    wr = fscanf(f,"\n");
    wr = fscanf(f,"%d %d %d %d\n", &(this->mask_radius[1]),  &(this->mask_radius[0]), &(this->sensing_radius[1]),  &(this->sensing_radius[0]));

    // undistort fake pin parameters
    wrc= fgets(buf, CMV_MAX_BUF,f);
    wr = fscanf(f,"\n");
    wr = fscanf(f,"%lf %lf %lf %lf\n", &this->pin_fx,  &this->pin_fy, &this->pin_cx, &this->pin_cy);

    // undistort fake pin parameters
    wrc= fgets(buf, CMV_MAX_BUF,f);
    wr = fscanf(f,"\n");
    wr = fscanf(f,"%f %f %d %d\n", &this->mp_fov[0],  &this->mp_fov[1], &this->mp_num, &this->mp_width);

    wrc= fgets(buf, CMV_MAX_BUF,f);
    wr = fscanf(f,"\n");
    wr = fscanf(f,"%s\n", buf);

    fclose(f);

    for (auto i = 0; i < length_pol; i++)
    {
        this->pol_[i] = pol[i];
    }
    for (auto i = 0; i < length_invpol; i++)
    {
        this->invpol_[i] = invpol[i];
    }

    this->resize = this->height_ / this->in_height;
    this->width_ = this->in_width * this->resize;
    this->length_pol_ = length_pol;
    this->length_invpol_ = length_invpol;
    this->xc_ = xc_tem;
    this->yc_ = yc_tem;
    this->cx = yc_tem*resize;//PAL模型的xy定义和我们常用的方向是相反的
    this->cy = xc_tem*resize;
    this->c_ = c;
    this->d_ = d;
    this->e_ = e;
    mask_radius[0]*=resize;
    mask_radius[1]*=resize;
    sensing_radius[0] *= resize;
    sensing_radius[1] *= resize;

    std::string bufs(buf);
    // undistort_mode = 0: not undistort
    if(bufs == "unity"){
        undistort_mode = 1;
    }
    else if(bufs == "pin"){
        undistort_mode = 2;
    }
    else if(bufs == "multipin"){
        undistort_mode = 3;
    }

    assert(this->mp_fov[0] < 90);
}

PALCamera::~PALCamera()
{
}

/// Project from pixels to world coordiantes. Returns a bearing vector on z=1.
Vector3f PALCamera::cam2world(double x, double y, int lvl) const
{
    int multi = (int)1 << lvl;
    x = (x+0.5) * multi - 0.5; // 亚像素精度
    y = (y+0.5) * multi - 0.5;  
    x /= resize;
    y /= resize;
    swap(y, x);

    Vector3f xyz_f;
    double invdet = 1 / (c_ - d_ * e_); // 1/det(A), where A = [c,d;e,1] as in the Matlab file

    double xp = invdet * ((x - xc_) - d_ * (y - yc_));
    double yp = invdet * (-e_ * (x - xc_) + c_ * (y - yc_));

    double r = sqrt(xp * xp + yp * yp); //distance [pixels] of  the point from the image center
    double zp = pol_[0];
    double r_i = 1;
    int i;

    for (i = 1; i < length_pol_; i++)
    {
        r_i *= r;
        zp += r_i * pol_[i];
    }

    //normalize to unit norm
    // double invnorm = 1 / sqrt(xp * xp + yp * yp + zp * zp);

    //normalize to z=1;
    double invnorm = 1 / abs(zp);

    xyz_f[0] = invnorm * xp;
    xyz_f[1] = invnorm * yp;
    xyz_f[2] = invnorm * zp;

    // 修改pal坐标系和针孔相机模型一致
    xyz_f[2] = -xyz_f[2];
    swap(xyz_f[0], xyz_f[1]);
    return xyz_f;
}

Vector3f PALCamera::cam2world(const Vector2f &px, int lvl) const
{
    return cam2world(px[0], px[1], lvl);
}

Vector2f PALCamera::world2cam(const Vector3f &xyz_c, int lvl) 
{
    Vector3f p_world = xyz_c;
    // 修改pal坐标系和针孔相机模型一致
    swap(p_world(0), p_world(1));
    p_world(2) = -p_world(2);

    Vector2f px;
    double norm = sqrt(p_world[0] * p_world[0] + p_world[1] * p_world[1]);
    double theta = atan(p_world[2] / norm);
    double t, t_i;
    double rho, x, y;
    double invnorm;
    int i;

    if (norm != 0)
    {
        invnorm = 1 / norm;
        t = theta;
        rho = invpol_[0];
        t_i = 1;

        for (i = 1; i < length_invpol_; i++)
        {
            t_i *= t;
            rho += t_i * invpol_[i];
        }

        x = p_world[0] * invnorm * rho;
        y = p_world[1] * invnorm * rho;

        px[0] = x * c_ + y * d_ + xc_;
        px[1] = x * e_ + y + yc_;
    }
    else
    {
        px[0] = xc_;
        px[1] = yc_;
    }
    
    swap(px(0), px(1));
    px(0) *= resize;
    px(1) *= resize;
    float multi = int(1)<<lvl;
    px = (px.array() + 0.5f) / multi - 0.5;
    return px;
}

double 
PALCamera::errorMultiplier2() const
{
    return 0;
}

double
PALCamera::errorMultiplier() const
{
    return 0;
}

double
PALCamera::getDerivativeOnTheta(const double theta) const
{
    double t, t_i;
    double drho;
    t = theta;
    drho = invpol_[1];
    t_i = 1;

    for (auto i = 2; i < length_invpol_; i++)
    {
        t_i *= t;
        drho += i * t_i * invpol_[i];
    }

    return drho;
}

double PALCamera::getInvPolynomialOnTheta(const double theta) const
{
    double t, t_i;
    double ret;
    t = theta;
    t_i = 1;
    ret = invpol_[0];
    for (auto i = 1; i < length_invpol_; i++)
    {
        t_i *= t;
        ret += invpol_[i] * t_i;
    }
    return ret;
}

void PALCamera::jacobian_xyz2uv(const Vector3f &xyz, Matrix<float, 2, 6> &J){
    Matrix<float, 2, 3> puv_pxyz;
    jacobian_xyz2uv(xyz, J, puv_pxyz);
}

void PALCamera::jacobian_xyz2uv(
    const Vector3f &xyz,
    Matrix<float, 2, 6> &J,
    Matrix<float, 2, 3> &puv_pxyz
    )
{
    const double x = xyz[0];
    const double y = xyz[1];
    const double z = -xyz[2];

    const double n = sqrt(x * x + y * y);
    const double n_inv = 1.0 / n;
    const double n_inv_2 = n_inv * n_inv;

    const double pt_2 = x * x + y * y + z * z;
    const double pt_2_inv = 1.0 / pt_2;

    const double theta = atan(z * n_inv);
    const double rho = getInvPolynomialOnTheta(theta);

    // 导数部分
    const double pn_px = x * n_inv;
    const double pn_py = y * n_inv;

    const double ptheta_pn = -z * pt_2_inv;
    const double ptheta_pz = n * pt_2_inv;

    const double prho_ptheta = getDerivativeOnTheta(theta);

    const double put_prho = x * n_inv;
    const double put_px = rho * n_inv;
    const double put_pn = -x * rho * n_inv_2;

    const double pvt_prho = y * n_inv;
    const double pvt_py = rho * n_inv;
    const double pvt_pn = -y * rho * n_inv_2;

    const double drho_dx = prho_ptheta * ptheta_pn * pn_px;
    const double drho_dy = prho_ptheta * ptheta_pn * pn_py;

    // 链式法则

    const double dut_dx_hall = put_prho * prho_ptheta * ptheta_pn * pn_px + put_pn * pn_px + put_px;
    const double dut_dy_hall = put_prho * prho_ptheta * ptheta_pn * pn_py + put_pn * pn_py;
    const double dut_dz_hall = put_prho * prho_ptheta * ptheta_pz;

    const double dvt_dx_hall = pvt_prho * prho_ptheta * ptheta_pn * pn_px + pvt_pn * pn_px;
    const double dvt_dy_hall = pvt_prho * prho_ptheta * ptheta_pn * pn_py + pvt_pn * pn_py + pvt_py;
    const double dvt_dz_hall = pvt_prho * prho_ptheta * ptheta_pz;

    const double ignore_me = dut_dx_hall + dut_dy_hall + dut_dz_hall + 
    dvt_dx_hall + dvt_dy_hall + dvt_dz_hall;

    // u = x*rho/n  
    // --> du/dx = drho_dx * (x/n) + rho * dx_n_dx 
    // --> du/dy = x * (drho/dy * 1/n + rho * d(1/n)/dy)
    const double dx_n_dx = (1*n - x*pn_px) * n_inv_2;
    const double dut_dx = drho_dx * x * n_inv + rho * dx_n_dx; 
    const double dut_dy = x * (drho_dy * n - rho * pn_py)* n_inv_2;  
    const double dut_dz = -1*(x * n_inv * prho_ptheta * ptheta_pz); // 乘以-1 因为pal坐标轴z轴是反的

    const double dy_n_dy = (1*n - y*pn_py) * n_inv_2;
    const double dvt_dy = drho_dy * y * n_inv + rho * dy_n_dy; 
    const double dvt_dx = y * (drho_dx * n - rho * pn_px)* n_inv_2;  
    const double dvt_dz = -1*(y * n_inv * prho_ptheta * ptheta_pz); // 乘以-1 因为pal坐标轴z轴是反的

    // 和hallc的结果对比，一样的
    // cout << dut_dx - dut_dx_hall << endl;
    // cout << dut_dy - dut_dy_hall << endl;
    // cout << dvt_dy - dvt_dy_hall << endl;
    // cout << dvt_dx - dvt_dx_hall << endl;
    // cout << dut_dz - dut_dz_hall << endl;
    // cout << dvt_dz - dvt_dz_hall << endl;

    // 加上校正
    // const double du_dx = c_ * dut_dx + d_ * dvt_dx;
    // const double du_dy = c_ * dut_dy + d_ * dvt_dy;
    // const double du_dz = c_ * dut_dz + d_ * dvt_dz;
    // const double dv_dx = e_ * dut_dx + dvt_dx;
    // const double dv_dy = e_ * dut_dy + dvt_dy;
    // const double dv_dz = e_ * dut_dz + dvt_dz;

    Eigen::Matrix<float, 2, 2> A;
    A << c_, d_, e_, 1;

    Eigen::Matrix<float, 2, 3> dutvt_dxyz;
    // 这里乘以了缩放系数(du 乘以resize 而不是 resize,因为这里uv是反的)
    dutvt_dxyz << dut_dx*resize, dut_dy*resize, dut_dz*resize,
        dvt_dx*resize, dvt_dy*resize, dvt_dz*resize;
    puv_pxyz = dutvt_dxyz;

    Eigen::Matrix<float, 3, 6> dxyz_dT;
    dxyz_dT <<  1, 0, 0, 0, z, -y,
                0, 1, 0, -z, 0, x,
                0, 0, 1, y, -x, 0;

    J = A * dutvt_dxyz * dxyz_dT;

    // J(0, 0) = -z_inv;               // -1/z
    // J(0, 1) = 0.0;                  // 0
    // J(0, 2) = x * z_inv_2;          // x/z^2
    // J(0, 3) = y * J(0, 2);          // x*y/z^2
    // J(0, 4) = -(1.0 + x * J(0, 2)); // -(1.0 + x^2/z^2)
    // J(0, 5) = y * z_inv;            // y/z

    // J(1, 0) = 0.0;               // 0
    // J(1, 1) = -z_inv;            // -1/z
    // J(1, 2) = y * z_inv_2;       // y/z^2
    // J(1, 3) = 1.0 + y * J(1, 2); // 1.0 + y^2/z^2
    // J(1, 4) = -J(0, 3);          // -x*y/z^2
    // J(1, 5) = -x * z_inv;        // x/z
}

} // namespace vk