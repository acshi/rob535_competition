#include "mex.h"
#include "matrix.h"

// on windows needs some lib's explicitly specified. not sure why it doesn't auto-detect...
// mex -R2018a -g rob535_competition_mex.c target/debug/rob535_competition.lib WS2_32.Lib Userenv.lib

extern void solve_obstacle_problem(int n_track, const double *bl, const double *br,
                                   const double *cline, const double *thetas,
                                   int n_obs, const double *obs_x, const double *obs_y,
                                   int *n_controls, double *controls);

#define MAX_CONTROL_STEPS 1024

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 6) {
        mexErrMsgTxt("Wrong number of input args, expected 6");
    }
    if (nlhs != 1) {
        mexErrMsgTxt("Wrong number of output args, expected 1");
    }

    double *bl = mxGetDoubles(prhs[0]);
    double *br = mxGetDoubles(prhs[1]);
    double *cline = mxGetDoubles(prhs[2]);
    double *theta = mxGetDoubles(prhs[3]);
    double *obs_x = mxGetDoubles(prhs[4]);
    double *obs_y = mxGetDoubles(prhs[5]);
    int n_track = mxGetM(prhs[0]);
    int n_obs_len = mxGetM(prhs[4]);
    int n_obs = n_obs_len / 4;

    printf("Got %d track elements\n", n_track);
    printf("Got %d obstacles (%d total points)\n", n_obs, n_obs_len);

    plhs[0] = mxCreateDoubleMatrix(MAX_CONTROL_STEPS, 2, mxREAL);
    double *controls = mxGetDoubles(plhs[0]);

    int n_controls = MAX_CONTROL_STEPS;
    solve_obstacle_problem(n_track, bl, br, cline, theta,
                           n_obs, obs_x, obs_y,
                           &n_controls, controls);
    mxSetM(plhs[0], n_controls);
}
