#include <Eigen/Eigen>

namespace dso_line{
    
// (x0, u): 3D line model
// return: inliear percentange
float line_estimate_g(const Eigen::MatrixXf &points, Eigen::Vector3f &x0_ret, Eigen::Vector3f &u_ret);

}