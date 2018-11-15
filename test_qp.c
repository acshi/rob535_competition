#include <stdio.h>
#include <stdlib.h>

int n = 3;
double H[] = {1, -1, 1, -1, 2, -2, 1, -2, 4};
double f[] = {2, -3, 1};
int n_le = 0;
double A[] = {1, 1, -1, 2, 2, 1};
double b[] = {2, 2, 3};
int n_eq = 1;
double Aeq[] = {1, 1, 1};
double beq[] = {0.5};
double lb[] = {0, 0, 0};
double ub[] = {1, 1, 1};
double obj_val = 0;
double x[] = {0, 0, 0};

int quadprog(int n, double *H, double *f,
             int n_le, double *A, double *b,
             int n_eq, double *Aeq, double *beq,
             double *lb, double *ub,
             double *obj_val, double *x_out);

struct CPXENV;
typedef struct CPXENV* CPXENVptr;

struct CPXLP;
typedef struct CPXLP* CPXLPptr;

int qp_create(int n, double *f, double *lb, double *ub,
              int n_le, int n_eq,
              CPXENVptr *env_out, CPXLPptr *qp_out);

int qp_diagonal_quadratic_cost(CPXENVptr env, CPXLPptr qp, double *q_diag);
int qp_dense_quadratic_cost(CPXENVptr env, CPXLPptr qp, double *q);
int qp_dense_le_constraints(CPXENVptr env, CPXLPptr qp, int n_le, double *coefs, double *rhs);
int qp_sparse_le_constraints(CPXENVptr env, CPXLPptr qp, int n_le, int n_coefs, int *constraint_indices, int *column_indices, double *coefs, double *rhs);
int qp_dense_eq_constraints(CPXENVptr env, CPXLPptr qp, int n_eq, double *coefs, double *rhs);
int qp_sparse_eq_constraints(CPXENVptr env, CPXLPptr qp, int n_eq, int n_coefs, int *constraint_indices, int *column_indices, double *coefs, double *rhs);
int qp_run(CPXENVptr env, CPXLPptr qp, double *obj_val, double *x_out);
int qp_destroy(CPXENVptr env, CPXLPptr qp);

void test1() {
    int sol_stat = quadprog(n, H, f, n_le, A, b, n_eq, Aeq, beq, lb, ub, &obj_val, x);
    printf("\nSolution status = %d\n", sol_stat);
    printf("Solution value  = %f\n\n", obj_val);
    for (int j = 0; j < n; j++) {
       printf ("Column %d: value = %10f\n", j, x[j]);
    }
    printf("\nFinished!\n");
}

void test2() {
    CPXENVptr env = NULL;
    CPXLPptr qp = NULL;
    qp_create(n, f, lb, ub, n_le, n_eq, &env, &qp);
    double H_diag[] = {1, 2, 4};
    // qp_diagonal_quadratic_cost(env, qp, H_diag);
    qp_dense_quadratic_cost(env, qp, H);
    qp_dense_eq_constraints(env, qp, n_eq, Aeq, beq);

    int sol_stat = qp_run(env, qp, &obj_val, x);
    printf("\nSolution status = %d\n", sol_stat);
    printf("Solution value  = %f\n\n", obj_val);
    for (int j = 0; j < n; j++) {
       printf ("Column %d: value = %10f\n", j, x[j]);
    }
    printf("\nFinished!\n");

    int status = qp_destroy(env, qp);
    printf("status: %d\n", status);
}

int main(int argc, char** argv) {
    test1();
    // test2();
    return 0;
}
