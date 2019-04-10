#include <iostream>
#include <string>
#include <Eigen/Core>
#include <util/pal_model.h>
#include <util/pal_interface.h>
#include <OptimizationBackend/MatrixAccumulators.h>
#include <vector>

#define T transpose()

using namespace std;
using namespace Eigen;
using namespace dso;

int main(void){
    pal_init("/home/hwj23/Dataset/PAL/calib_results_fish.txt"); 

	Eigen::Matrix<float, 2, 6> dx2dSE;
	Eigen::Matrix<float, 2, 3> duv2dxyz;

    Vec3f p(0.04, -1.03, 0.98);
	pal_model_g->jacobian_xyz2uv(p, dx2dSE, duv2dxyz);
    cout << "p = " << p.transpose() << endl;
    cout << duv2dxyz << endl;

    return 0;
}
