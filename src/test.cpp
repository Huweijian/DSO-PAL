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

    int u = pal_model_g->xc_;
    int v = pal_model_g->yc_ + 135;

    Vector3f pt = pal_model_g->cam2world(u, v);
    cout << atan2(pt[1], pt[2])/3.14*180;    
    return -1;

    Vector3f t(100, 0, 0);
    float dxdd = (t[0]-t[2]*u); // \rho_2 / \rho1 * (tx - u'_2 * tz)
    float dydd = (t[1]-t[2]*v); // \rho_2 / \rho1 * (ty - v'_2 * tz)
    Vec2f dr2duv2(10, 0);


    Eigen::Matrix<float, 2, 6> dx2dSE;
    Eigen::Matrix<float, 2, 3> duv2dxyz;
    pal_model_g->jacobian_xyz2uv(pt, dx2dSE, duv2dxyz);
    Vec3f dxyzdd = Vec3f(dxdd, dydd, 0);

    // dr/dd
    Vector2f duvdd = duv2dxyz * dxyzdd; 
    cout << pt.T << "\n\n";
    cout << duv2dxyz << "\n\n";
    cout << duvdd.transpose() << endl;

    return 0;
}
