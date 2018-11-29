#include "mex.h"

// on windows needs some lib's explicitly specified. not sure why it doesn't auto-detect...
// mex -g rob535_competition_mex.c target/debug/rob535_competition.lib WS2_32.Lib Userenv.lib

extern void trajectory_for_obstacles(int n_obs, const double *obs_x, const double *obs_y,
                                     int n_controls, double *controls);

#define MAX_CONTROL_STEPS 1024

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgTxt("Wrong number of input args");
    }
    if (nlhs != 1) {
        mexErrMsgTxt("Wrong number of output args");
    }

    double *obs_x = mxGetPr(prhs[0]);
    double *obs_y = mxGetPr(prhs[1]);
    int n_obs = mxGetM(prhs[0]);

    plhs[0] = mxCreateDoubleMatrix(MAX_CONTROL_STEPS, 2, mxREAL);
    double *controls = mxGetPr(plhs[0]);

    printf("Before rust call\n");
    trajectory_for_obstacles(n_obs, obs_x, obs_y, MAX_CONTROL_STEPS, controls);
    printf("After rust call\n");
}
