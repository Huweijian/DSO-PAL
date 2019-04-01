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

    vector<int> pointHessians; 
    for(int i=1; i<=100; i++){
        if(i % 10 == 0)
            pointHessians.push_back(0);
        else
            pointHessians.push_back(i);
    }

    for(unsigned int i=0;i<pointHessians.size();i++){
        int ph = pointHessians[i];

        if(ph == 0)
        {
            pointHessians[i] = pointHessians.back();
            pointHessians.pop_back();
            i--;
        }
    }

    for(auto i:pointHessians)
        cout << i << " ";

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
