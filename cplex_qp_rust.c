#include <stdio.h>
#include <stdlib.h>
#include <ilcplex/cplex.h>

int cplex_check_error(CPXENVptr env, int status)
{
    if (status) {
       char error_msg[CPXMESSAGEBUFSIZE];
       CPXgeterrorstring(env, status, error_msg);
       fprintf(stderr, "%s", error_msg);
       return 1;
    }
    return 0;
}

// MATLAB-style call
// min_x 1/2 x^T H x + f^T x, such that A x <= b, A_eq x == b_eq, lb <= x <= ub
int quadprog(size_t n, double *H, double *f,
             size_t n_le, double *A, double *b,
             size_t n_eq, double *A_eq, double *b_eq,
             double *lb, double *ub,
             double *obj_val, double *x_out)
{
    int status = 0;
    CPXENVptr env = CPXopenCPLEX(&status);
    if (cplex_check_error(env, status)) { return status; }

    // screen update thing
    // status = CPXsetintparam(env, CPXPARAM_ScreenOutput, CPX_ON);
    // if (cplex_check_error(env, status)) { return status; }

    // data checking (not sure what for)
    // status = CPXsetintparam(env, CPXPARAM_Read_DataCheck, CPX_DATACHECK_WARN);
    // if (cplex_check_error(env, status)) { return status; }

    CPXLPptr qp = CPXcreateprob(env, &status, "Test QP");
    if (cplex_check_error(env, status)) { return status; }

    // set to minimization problem
    status = CPXchgobjsen(env, qp, CPX_MIN);
    if (cplex_check_error(env, status)) { return status; }

    // Set the linear part of the objective function and inform CPLEX
    // about the number of varialbes (columns) that we are using
    status = CPXnewcols(env, qp, n, f, lb, ub, NULL, NULL);
    if (cplex_check_error(env, status)) { return status; }

    // Set the quadratic objective matrix H (or Q)
    // This is general enough to allow sparse matrices
    // Where the desired non-zero rows of Q/H are in qmatind
    // qmatbeg marks the start of each column in qmatind/qmatval
    // qmatcnt is the number of entries for each column in qmatind/qmatval

    // The full MATLAB quadprog does not allow sparse representation so we specify that here
    // and we use H directly as qmatval
    int *qmatbeg = malloc(sizeof(int) * n);
    int *qmatcnt = malloc(sizeof(int) * n);
    int *qmatind = malloc(sizeof(int) * n * n);
    for (int i = 0; i < n; i++) {
        qmatbeg[i] = i * n;
        qmatcnt[i] = n;
        for (int j = 0; j < n; j++) {
            qmatind[i * n + j] = j;
        }
    }
    status = CPXcopyquad(env, qp, qmatbeg, qmatcnt, qmatind, H);
    free(qmatbeg);
    free(qmatcnt);
    free(qmatind);
    if (cplex_check_error(env, status)) { return status; }

    // Now we add the constraints
    // and we can do these with first the <= terms and then the == terms
    // if either of these are not null, that is!
    if (n_le > 0 && A && b) {
        // Here, there is no rmatcnt, because it is fixed at the value of nzcnt
        // which we have set to be n * n_le
        char *sense = malloc(sizeof(char) * n_le);
        int *rmatbeg = malloc(sizeof(int) * n_eq);
        int *rmatind = malloc(sizeof(int) * n * n_le);
        for (int i = 0; i < n_le; i++) {
            sense[i] = 'L';
            rmatbeg[i] = i * n;
            for (int j = 0; j < n; j++) {
                rmatind[i * n + j] = j;
            }
        }
        status = CPXaddrows(env, qp, 0, n_le, n * n_le, b, sense, rmatbeg, rmatind, A, NULL, NULL);
        free(sense);
        free(rmatbeg);
        free(rmatind);
        if (cplex_check_error(env, status)) { return status; }
    }

    if (n_eq > 0 && A_eq && b_eq) {
        // Here, there is no rmatcnt, because it is fixed at the value of nzcnt
        // which we have set to be n * n_eq
        char *sense = malloc(sizeof(char) * n_eq);
        int *rmatbeg = malloc(sizeof(int) * n_eq);
        int *rmatind = malloc(sizeof(int) * n * n_eq);
        for (int i = 0; i < n_eq; i++) {
            sense[i] = 'E';
            rmatbeg[i] = i * n;
            for (int j = 0; j < n; j++) {
                rmatind[i * n + j] = j;
            }
        }
        status = CPXaddrows(env, qp, 0, n_eq, n * n_eq, b_eq, sense, rmatbeg, rmatind, A_eq, NULL, NULL);
        free(sense);
        free(rmatbeg);
        free(rmatind);
        if (cplex_check_error(env, status)) { return status; }
    }

    status = CPXqpopt(env, qp);
    if (cplex_check_error(env, status)) { return status; }

    int solstat;
    status = CPXsolution(env, qp, &solstat, obj_val, x_out, NULL, NULL, NULL);
    if (cplex_check_error(env, status)) { return status; }

    status = CPXfreeprob(env, &qp);
    if (cplex_check_error(env, status)) { return status; }

    status = CPXcloseCPLEX(&env);
    if (cplex_check_error(env, status)) { return status; }

    return solstat;
}
