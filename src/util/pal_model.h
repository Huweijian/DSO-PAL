/*
 * pal_camera.h
 * 
 */

#ifndef PAL_CAMERA_H_
#define PAL_CAMERA_H_

#include <cstdio>
#include <Eigen/Core>

namespace pal
{
using namespace std;
using namespace Eigen;

class PALCamera
{
#define CMV_MAX_BUF 1024
#define MAX_POL_LENGTH 64

public:
  double height_;
  double width_;
  double pol_[MAX_POL_LENGTH];    // the polynomial coefficients: pol[0] + x"pol[1] + x^2*pol[2] + ... + x^(N-1)*pol[N-1]
  int length_pol_;                // length of polynomial
  double invpol_[MAX_POL_LENGTH]; // the coefficients of the inverse polynomial
  int length_invpol_;             // length of inverse polynomial
  double xc_;                     // row coordinate of the center
  double yc_;                     // column coordinate of the center
  double c_;                      // affine parameter
  double d_;                      // affine parameter
  double e_;                      // affine parameter
  int mask_radius[2] = {0, 0};    // mask的内径和外经
  int sensing_radius[2] = {0, 0}; // 传感的内径和外经
  double pin_fx, pin_fy, pin_cx, pin_cy;  // undistort fake pin params
  float mp_fov[2]; int mp_num, mp_width;  
  int undistort_mode = 0;

public:
  PALCamera(std::string filename);
  ~PALCamera();

  /// Project from pixels to world coordiantes. Returns a bearing vector of unit length.
  virtual Vector3f cam2world(double x, double y, int lvl = 0) const;

  /// Project from pixels to world coordiantes. Returns a bearing vector of unit length.
  virtual Vector3f cam2world(const Vector2f &px, int lvl =0) const;

  virtual Vector2f world2cam(const Vector3f &xyz_c, int lvl = 0);

  /// projects unit plane coordinates to camera coordinates
  // virtual Vector2f
  // world2cam(const Vector2f &uv) const;

  virtual double errorMultiplier2() const;

  virtual double errorMultiplier() const;

  /// Frame jacobian for projection of 3D point in (f)rame coordinate to
  /// unit plane coordinates uv (focal length = 1).
  void jacobian_xyz2uv(const Vector3f &xyz, Matrix<float, 2, 6> &J, Matrix<float, 2, 3> &puv_pxyz);
  void jacobian_xyz2uv(const Vector3f &xyz, Matrix<float, 2, 6> &J);

private:
  double getDerivativeOnTheta(const double theta) const;

  double getInvPolynomialOnTheta(const double theta) const;
};

} // namespace pal

#endif