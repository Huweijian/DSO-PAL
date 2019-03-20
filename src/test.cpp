#include <iostream>
#include <cstring>
#include <Eigen/Core>
// #include <MatrixAccumulators.h>
#include <OptimizationBackend/MatrixAccumulators.h>

using namespace std;
using namespace Eigen;
using namespace dso;

using namespace std;

class A{
    public:
    int x, y;
};

istream  & operator >> (istream &is, A &a){
    is >> a.x >> a.y;
}


int main(void){
    A a;
    cin >> a;
    cout << a.x << " " << a.y << endl;


    return 0;
}
