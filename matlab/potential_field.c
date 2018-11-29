#include "mex.h"
#include "matrix.h"

extern void multiply(double* a, double* b, double* c, long elements);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double* a;
    double* b;
    double* c;

    mwSize elements;

    if (nrhs != 2) {
        mexErrMsgTxt("Wrong number of input args");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Wrong number of output args");
    }

    a = mxGetPr(prhs[0]);
    b = mxGetPr(prhs[1]);
    elements = mxGetM(prhs[0]);

    plhs[0] = mxCreateDoubleMatrix(elements, 1, mxREAL);
    c = mxGetPr(plhs[0]);

    multiply(a, b, c, elements);
}

//int main() {
//    mxArray *c = NULL;
//    mxArray *plhs[1], *prhs[2];
//    prhs[0] = mxCreateDoubleMatrix(3, 1, mxREAL);
//    prhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
//    mexFunction(1, plhs, 2, prhs);
//}
