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

    float a = 4.2108e+06;
    unsigned char *argb = 0;
    // void *p = &a;
    argb = (unsigned char*)(&a);
    printf("%u %u %u %u\n", argb[0], argb[1], argb[2], argb[3]);

    uint8_t r = 128;
    uint8_t g = 128;
    uint8_t b = 224;
    int32_t rgb = (r << 16) | (g << 8) | b; 
    a = *(float *)(&rgb); // makes the point red
    cout << a << endl;

    return -1;

    pal_init("/home/hwj23/Dataset/PAL/calib_results_fish.txt"); 

	Eigen::Matrix<float, 2, 6> dx2dSE;
	Eigen::Matrix<float, 2, 3> duv2dxyz;

    Vec3f pt(0.04, -1.03, 0.98);
	pal_model_g->jacobian_xyz2uv(pt, dx2dSE, duv2dxyz);
    cout << "p = " << pt.transpose() << endl;
    cout << duv2dxyz << endl;

    return 0;
}
