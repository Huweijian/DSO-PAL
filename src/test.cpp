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

    float a = NAN;
    if(10 < a)
        cout << "OK" << endl;
    else
    {
        cout <<"not ok" << endl;
    }
    
    return 0;

    init_pal("/home/hwj23/Dataset/PAL/calib_results_fish.txt"); 

	Eigen::Matrix<float, 2, 6> dx2dSE;
	Eigen::Matrix<float, 2, 3> duv2dxyz;

    auto p = pal_model_g->cam2world(100, 200);
	pal_model_g->jacobian_xyz2uv(p, dx2dSE, duv2dxyz);
    cout << "p1 = " << p.transpose() << endl;
    cout << duv2dxyz << endl;

    p = p * 10;
	pal_model_g->jacobian_xyz2uv(p, dx2dSE, duv2dxyz);
    cout << "p2 = " << p.transpose() << endl;
    cout << duv2dxyz << endl;

    return 0;
}
