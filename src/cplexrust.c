#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
int quadprog(int n, double *H, double *f,
             int n_le, double *A, double *b,
             int n_eq, double *A_eq, double *b_eq,
             double *lb, double *ub,
             double *obj_val, double *x_out)
{
    int status = 0;
    CPXENVptr env = CPXopenCPLEX(&status);
    if (cplex_check_error(env, status)) { return status; }

    // screen update thing
    // status = CPXsetintparam(env, CPXPARAM_ScreenOutput, CPX_ON);
    // if (cplex_check_error(env, status)) { return status; }

    // data checking
    // status = CPXsetintparam(env, CPXPARAM_Read_DataCheck, CPX_DATACHECK_WARN);
    // if (cplex_check_error(env, status)) { return status; }

    CPXLPptr qp = CPXcreateprob(env, &status, "One-Step QP");
    if (cplex_check_error(env, status)) { return status; }

    // set to minimization problem
    status = CPXchgobjsen(env, qp, CPX_MIN);
    if (cplex_check_error(env, status)) { return status; }

    // Set the linear part of the objective function and inform CPLEX
    // about the number of variables (columns) that we are using
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

    // 1 indicates a solution
    // translate to 0 as the error code for "no error"
    if (solstat == 1) {
        return 0;
    }
    return -1;
}

int qp_create(int n, double *f, double *lb, double *ub, int n_le, int n_eq, CPXENVptr *env_out, CPXLPptr *qp_out) {
    int status = 0;
    CPXENVptr env = CPXopenCPLEX(&status);
    if (cplex_check_error(env, status)) { return status; }

    CPXsetintparam(env, CPXPARAM_QPMethod, CPX_ALG_PRIMAL); // BARRIER is ~4x slower than the other algs
    CPXsetintparam(env, CPX_PARAM_FEASOPTMODE, CPX_FEASOPT_OPT_QUAD);
    // CPXsetdblparam(env, CPXPARAM_Simplex_Tolerances_Optimality, 1e-14);

    // screen update thing
    // status = CPXsetintparam(env, CPXPARAM_ScreenOutput, CPX_ON);
    // if (cplex_check_error(env, status)) { return status; }

    // data checking
    // status = CPXsetintparam(env, CPXPARAM_Read_DataCheck, CPX_DATACHECK_WARN);
    // if (cplex_check_error(env, status)) { return status; }

    CPXLPptr qp = CPXcreateprob(env, &status, "Full QP");
    if (cplex_check_error(env, status)) { return status; }

    // set to minimization problem
    status = CPXchgobjsen(env, qp, CPX_MIN);
    if (cplex_check_error(env, status)) { return status; }

    // Set the linear part of the objective function and inform CPLEX
    // about the number of variables (columns) that we are using
    status = CPXnewcols(env, qp, n, f, lb, ub, NULL, NULL);
    if (cplex_check_error(env, status)) { return status; }

    if (n_le > 0) {
        char *sense = malloc(sizeof(char) * n_le);
        for (int i = 0; i < n_le; i++) {
            sense[i] = 'L';
        }
        status = CPXnewrows(env, qp, n_le, NULL, sense, NULL, NULL);
        free(sense);
        if (cplex_check_error(env, status)) { return status; }
    }

    if (n_eq > 0) {
        char *sense = malloc(sizeof(char) * n_eq);
        for (int i = 0; i < n_eq; i++) {
            sense[i] = 'E';
        }
        status = CPXnewrows(env, qp, n_eq, NULL, sense, NULL, NULL);
        free(sense);
        if (cplex_check_error(env, status)) { return status; }
    }

    *env_out = env;
    *qp_out = qp;

    return 0;
}

int qp_linear_cost(CPXENVptr env, CPXLPptr qp, int n, double *f) {
    int *idxs = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        idxs[i] = i;
    }
    CPXchgobj(env, qp, n, idxs, f);
    free(idxs);

    return 0;
}

int qp_bounds(CPXENVptr env, CPXLPptr qp, double *lb, double *ub) {
    int n = CPXgetnumcols(env, qp);
    char *bound_kind = malloc(sizeof(char) * n);
    int *idxs = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        bound_kind[i] = 'L';
        idxs[i] = i;
    }
    int status = CPXchgbds(env, qp, n, idxs, bound_kind, lb);
    if (!status) {
        for (int i = 0; i < n; i++) {
            bound_kind[i] = 'U';
        }
        status = CPXchgbds(env, qp, n, idxs, bound_kind, ub);
    }
    free(idxs);
    free(bound_kind);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

int qp_diagonal_quadratic_cost(CPXENVptr env, CPXLPptr qp, double *q_diag) {
    return CPXcopyqpsep(env, qp, q_diag);
}

int qp_dense_quadratic_cost(CPXENVptr env, CPXLPptr qp, double *q) {
    // The full MATLAB quadprog does not allow sparse representation so we specify that here
    // and we use H directly as qmatval
    int n = CPXgetnumcols(env, qp);
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
    int status = CPXcopyquad(env, qp, qmatbeg, qmatcnt, qmatind, q);
    free(qmatbeg);
    free(qmatcnt);
    free(qmatind);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

int qp_sparse_quadratic_cost(CPXENVptr env, CPXLPptr qp, int *coefs_per_row, int *column_indices, double *q) {
    int n = CPXgetnumcols(env, qp);
    int *qmatbeg = malloc(sizeof(int) * n);
    int offset = 0;
    for (int i = 0; i < n; i++) {
        qmatbeg[i] = offset;
        offset += coefs_per_row[i];
    }
    int status = CPXcopyquad(env, qp, qmatbeg, coefs_per_row, column_indices, q);
    free(qmatbeg);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

int qp_dense_le_constraints(CPXENVptr env, CPXLPptr qp, int n_le, double *coefs, double *rhs) {
    // we organize the <= constraints first in the list
    // so this is just a direct call
    int n = CPXgetnumcols(env, qp);
    int *constraint_indices = malloc(sizeof(int) * n  * n_le);
    int *column_indices = malloc(sizeof(int) * n  * n_le);
    int idx = 0;
    for (int i = 0; i < n_le; i++) {
        for (int j = 0; j < n; j++) {
            constraint_indices[idx] = i;
            column_indices[idx] = j;
            idx++;
        }
    }
    int status = CPXchgcoeflist(env, qp, n * n_le, constraint_indices, column_indices, coefs);
    if (status == 0) {
        for (int i = 0; i < n_le; i++) {
            constraint_indices[i] = i;
        }
        status = CPXchgrhs(env, qp, n_le, constraint_indices, rhs);
    }
    free(constraint_indices);
    free(column_indices);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

int qp_sparse_le_constraints(CPXENVptr env, CPXLPptr qp, int n_le, int n_coefs, int *constraint_indices, int *column_indices, double *coefs, double *rhs) {
    // we organize the <= constraints first in the list
    // so this is just a direct call
    int status = CPXchgcoeflist(env, qp, n_coefs, constraint_indices, column_indices, coefs);
    if (cplex_check_error(env, status)) { return status; }

    int *rhs_indices = malloc(sizeof(int) * n_le);
    for (int i = 0; i < n_le; i++) {
        rhs_indices[i] = i;
    }
    status = CPXchgrhs(env, qp, n_le, rhs_indices, rhs);
    free(rhs_indices);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

int qp_dense_eq_constraints(CPXENVptr env, CPXLPptr qp, int n_eq, double *coefs, double *rhs) {
    // we organize the == constraints after then <= ones
    // so we have to offset by the number of <= constraints
    int n_le = CPXgetnumrows(env, qp) - n_eq;
    int n = CPXgetnumcols(env, qp);
    int *constraint_indices = malloc(sizeof(int) * n * n_eq);
    int *column_indices = malloc(sizeof(int) * n * n_eq);
    int idx = 0;
    for (int i = 0; i < n_eq; i++) {
        for (int j = 0; j < n; j++) {
            constraint_indices[idx] = i + n_le;
            column_indices[idx] = j;
            idx++;
        }
    }
    int status = CPXchgcoeflist(env, qp, n * n_eq, constraint_indices, column_indices, coefs);
    if (status == 0) {
        for (int i = 0; i < n_eq; i++) {
            constraint_indices[i] = i + n_le;
        }
        status = CPXchgrhs(env, qp, n_eq, constraint_indices, rhs);
    }
    free(constraint_indices);
    free(column_indices);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

int qp_quadratic_geq(CPXENVptr env, CPXLPptr qp, int n_lin, int n_quad, double rhs,
                     int *lin_idxs, double *lin_vals,
                     int *quad_rows, int *quad_cols, double *quad_vals) {
    int status = CPXchgprobtype(env, qp, CPXPROB_QCP);
    if (cplex_check_error(env, status)) { return status; }

    status = CPXaddqconstr(env, qp, n_lin, n_quad, rhs, 'G', lin_idxs, lin_vals, quad_rows, quad_cols, quad_vals, NULL);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

// inclusive range
int qp_delete_quadratic_geqs(CPXENVptr env, CPXLPptr qp, int idx_low, int idx_high) {
    int status = CPXdelqconstrs(env, qp, idx_low, idx_high);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

// the right-hand side should be included with column indices of -1.
int qp_sparse_eq_constraints(CPXENVptr env, CPXLPptr qp, int n_eq, int n_coefs, int *constraint_indices, int *column_indices, double *coefs, double *rhs) {
    // we organize the == constraints after then <= ones
    // so we have to offset by the number of <= constraints
    int n_le = CPXgetnumrows(env, qp) - n_eq;
    for (int i = 0; i < n_eq; i++) {
        constraint_indices[i] += n_le;
    }
    int status = CPXchgcoeflist(env, qp, n_coefs, constraint_indices, column_indices, coefs);
    for (int i = 0; i < n_eq; i++) {
        constraint_indices[i] -= n_le;
    }

    if (cplex_check_error(env, status)) { return status; }

    int *rhs_indices = malloc(sizeof(int) * n_eq);
    for (int i = 0; i < n_eq; i++) {
        rhs_indices[i] = i + n_le;
    }
    status = CPXchgrhs(env, qp, n_eq, rhs_indices, rhs);
    free(rhs_indices);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}

int qp_run(CPXENVptr env, CPXLPptr qp, double *obj_val, double *x_out) {
    int status = CPXqpopt(env, qp);
    if (cplex_check_error(env, status)) { return status; }
    int solstat;
    status = CPXsolution(env, qp, &solstat, obj_val, x_out, NULL, NULL, NULL);
    if (status || solstat != CPX_STAT_OPTIMAL) {
        printf(".");
        fflush(stdout);

        int n_constraints = CPXgetnumrows(env, qp);
        double *rhs_relaxation_cost = malloc(sizeof(double) * n_constraints);
        // initial condition constraints can not be relaxed
        for (int i = 0; i < 3; i++) {
            rhs_relaxation_cost[i] = 0;
        }
        // but all other integration constraints can be relaxed equally
        for (int i = 3; i < n_constraints; i++) {
            rhs_relaxation_cost[i] = 1;
        }
        status = CPXfeasopt(env, qp, rhs_relaxation_cost, NULL, NULL, NULL);
        free(rhs_relaxation_cost);

        status = CPXsolution(env, qp, &solstat, obj_val, x_out, NULL, NULL, NULL);
    }
    if (cplex_check_error(env, status)) { return status; }

    // translate to 0 as the error code for "no error"
    if (solstat == CPX_STAT_OPTIMAL) {
        return 0;
    }
    // did not need relaxation
    if (solstat == CPX_STAT_FEASIBLE) {
        return 0;
    }
    // found relaxed solution
    if (solstat == CPX_STAT_FEASIBLE_RELAXED_QUAD) {
        return 0;
    }
    printf("Solution code: %d\n", solstat);
    return -1;
}

int qp_destroy(CPXENVptr env, CPXLPptr qp) {
    int status = CPXfreeprob(env, &qp);
    if (cplex_check_error(env, status)) { return status; }

    status = CPXcloseCPLEX(&env);
    if (cplex_check_error(env, status)) { return status; }

    return 0;
}
