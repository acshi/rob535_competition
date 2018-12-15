#include "mex.h"
#include "matrix.h"

extern void potential_fields(
        double* bl,
        double* br,
        double* cline,
        double* theta,
        int len,
        double* XObs,
        int nObs,
        double* u,
        int* u_len);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double* bl;
    double* br;
    double* cline;
    double* theta;
    double* XObs;

    mwSize elements;

    if (nrhs != 5) {
        mexErrMsgTxt("Wrong number of input args");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Wrong number of output args");
    }


    bl = mxGetPr(prhs[0]);
    br = mxGetPr(prhs[1]);
    cline = mxGetPr(prhs[2]);
    theta = mxGetPr(prhs[3]);
    XObs = mxGetPr(prhs[4]);

    int len = mxGetN(prhs[0]);
    int nObs = mxGetM(prhs[4])/8;

    double* u = malloc(8000*sizeof(double));
    int u_len = 0;
    potential_fields(bl, br, cline, theta, len, XObs, nObs, u, &u_len);

    mexPrintf("Length: %d\n", u_len);
    plhs[0] = mxCreateDoubleMatrix(2, u_len/2, mxREAL);
    double* u_matlab = mxGetPr(plhs[0]);
    for (int i = 0; i < u_len; i++) {
        u_matlab[i] = u[i];
    }
}
