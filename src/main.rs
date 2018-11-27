extern crate libc;
use libc::c_void;
use std::ptr;
use std::fs::File;
use std::io::Read;
use std::f64;

extern crate lapack;
extern crate blas;
extern crate openblas_src;
use lapack::*;
use blas::*;

extern crate rustplotlib;
use rustplotlib::{Axes2D, Line2D, Backend};
use rustplotlib::backend::Matplotlib;

extern crate time;
use time::precise_time_s;

#[link(name = "cplex", kind = "static")]
#[link(name = "cplexrust", kind = "static")]
#[allow(dead_code)]
extern {
    fn quadprog(n: i32, H: *const f64, f: *const f64,
                n_le: i32, A: *const f64, b: *const f64,
                n_eq: i32, A_eq: *const f64, b_eq: *const f64,
                lb: *const f64, ub: *const f64,
                obj_val: *mut f64, x_out: *mut f64) -> i32;

    fn qp_create(n: i32, f: *const f64, lb: *const f64, ub: *const f64,
                  n_le: i32, n_eq: i32,
                  env_out: *mut *mut c_void, qp_out: *mut *mut c_void) -> i32;

    fn qp_bounds(env: *mut c_void, qp: *mut c_void, lb: *const f64, ub: *const f64) -> i32;

    fn qp_diagonal_quadratic_cost(env: *mut c_void, qp: *mut c_void, q_diag: *const f64) -> i32;
    fn qp_dense_quadratic_cost(env: *mut c_void, qp: *mut c_void, q: *const f64) -> i32;
    fn qp_dense_le_constraints(env: *mut c_void, qp: *mut c_void, n_le: i32, coefs: *const f64, rhs: *const f64) -> i32;
    fn qp_sparse_le_constraints(env: *mut c_void, qp: *mut c_void, n_le: i32, n_coefs: i32,
                                constraint_indices: *const i32, column_indices: *const i32,
                                coefs: *const f64, rhs: *const f64) -> i32;
    fn qp_dense_eq_constraints(env: *mut c_void, qp: *mut c_void, n_eq: i32, coefs: *const f64, rhs: *const f64) -> i32;
    fn qp_sparse_eq_constraints(env: *mut c_void, qp: *mut c_void, n_eq: i32, n_coefs: i32,
                                constraint_indices: *const i32, column_indices: *const i32,
                                coefs: *const f64, rhs: *const f64) -> i32;
    fn qp_quadratic_geq(env: *mut c_void, qp: *mut c_void, n_lin: i32, n_quad: i32, rhs: f64,
                         lin_idxs: *const i32, lin_vals: *const f64,
                         quad_rows: *const i32, quad_cols: *const i32, quad_vals: *const f64) -> i32;
    fn qp_delete_quadratic_geqs(env: *mut c_void, qp: *mut c_void, idx_low: i32, idx_high: i32) -> i32;
    fn qp_run(env: *mut c_void, qp: *mut c_void, obj_val: *mut f64, x_out: *mut f64) -> i32;
    fn qp_destroy(env: *mut c_void, qp: *mut c_void) -> i32;
}

#[allow(dead_code)]
struct QuadProg {
    env: *mut c_void,
    qp: *mut c_void,
    n: i32,
    n_le: i32,
    n_eq: i32,
    n_quad_geqs: i32,
}

#[allow(dead_code)]
impl QuadProg {
    fn new(n: usize, f: &[f64], lb: &[f64], ub: &[f64], n_le: usize, n_eq: usize) -> QuadProg {
        assert_eq!(f.len(), n);
        assert_eq!(lb.len(), n);
        assert_eq!(ub.len(), n);

        let n = n as i32;
        let n_le = n_le as i32;
        let n_eq = n_eq as i32;

        let mut env = ptr::null_mut();
        let mut qp = ptr::null_mut();
        unsafe {
            let status = qp_create(n, f.as_ptr(), lb.as_ptr(), ub.as_ptr(), n_le, n_eq, &mut env, &mut qp);
            if status != 0 {
                panic!();
            }
        }
        QuadProg { env, qp, n, n_le, n_eq, n_quad_geqs: 0 }
    }

    fn bounds(&mut self, lb: &[f64], ub: &[f64]) {
        assert_eq!(lb.len(), self.n as usize);
        assert_eq!(ub.len(), self.n as usize);
        unsafe {
            let status = qp_bounds(self.env, self.qp, lb.as_ptr(), ub.as_ptr());
            if status != 0 {
                panic!();
            }
        }
    }

    fn diagonal_quadratic_cost(&mut self, q_diag: &[f64]) {
        assert_eq!(q_diag.len(), self.n as usize);
        unsafe {
            let status = qp_diagonal_quadratic_cost(self.env, self.qp, q_diag.as_ptr());
            if status != 0 {
                panic!();
            }
        }
    }

    fn dense_quadratic_cost(&mut self, q: &[f64]) {
        assert_eq!(q.len(), (self.n * self.n) as usize);
        unsafe {
            let status = qp_dense_quadratic_cost(self.env, self.qp, q.as_ptr());
            if status != 0 {
                panic!();
            }
        }
    }

    fn dense_le_constraints(&mut self, coefs: &[f64], rhs: &[f64]) {
        assert_eq!(coefs.len(), (self.n * self.n_le) as usize);
        assert_eq!(rhs.len(), self.n_le as usize);
        unsafe {
            let status = qp_dense_le_constraints(self.env, self.qp, self.n_le, coefs.as_ptr(), rhs.as_ptr());
            if status != 0 {
                panic!();
            }
        }
    }

    fn sparse_le_constraints(&mut self, constraint_idxs: &[i32], column_idxs: &[i32], coefs: &[f64], rhs: &[f64]) {
        let n_coefs = coefs.len();
        assert_eq!(constraint_idxs.len(), n_coefs);
        assert_eq!(column_idxs.len(), n_coefs);
        assert_eq!(rhs.len(), self.n_le as usize);
        unsafe {
            let status = qp_sparse_le_constraints(self.env, self.qp, self.n_le, n_coefs as i32,
                                                  constraint_idxs.as_ptr(), column_idxs.as_ptr(),
                                                  coefs.as_ptr(), rhs.as_ptr());
            if status != 0 {
                panic!();
            }
        }
    }

    fn dense_eq_constraints(&mut self, coefs: &[f64], rhs: &[f64]) {
        assert_eq!(coefs.len(), (self.n * self.n_eq) as usize);
        assert_eq!(rhs.len(), self.n_eq as usize);
        unsafe {
            let status = qp_dense_eq_constraints(self.env, self.qp, self.n_eq, coefs.as_ptr(), rhs.as_ptr());
            if status != 0 {
                panic!();
            }
        }
    }

    fn sparse_eq_constraints(&mut self, constraint_idxs: &[i32], column_idxs: &[i32], coefs: &[f64], rhs: &[f64]) {
        let n_coefs = coefs.len();
        assert_eq!(constraint_idxs.len(), n_coefs);
        assert_eq!(column_idxs.len(), n_coefs);
        assert_eq!(rhs.len(), self.n_eq as usize);
        unsafe {
            let status = qp_sparse_eq_constraints(self.env, self.qp, self.n_eq, n_coefs as i32,
                                                 constraint_idxs.as_ptr(), column_idxs.as_ptr(),
                                                 coefs.as_ptr(), rhs.as_ptr());
            if status != 0 {
                panic!();
            }
        }
    }

    fn quadratic_geq(&mut self, rhs: f64, lin_idxs: &[i32], lin_vals: &[f64],
                     quad_rows: &[i32], quad_cols: &[i32], quad_vals: &[f64]) {
        let n_lin = lin_idxs.len();
        assert_eq!(lin_vals.len(), n_lin);
        let n_quad = quad_rows.len();
        assert_eq!(quad_cols.len(), n_quad);
        assert_eq!(quad_vals.len(), n_quad);
        unsafe {
            let status = qp_quadratic_geq(self.env, self.qp, n_lin as i32, n_quad as i32, rhs,
                                          lin_idxs.as_ptr(), lin_vals.as_ptr(),
                                          quad_rows.as_ptr(), quad_cols.as_ptr(), quad_vals.as_ptr());

            if status != 0 {
                panic!();
            }
        }

        self.n_quad_geqs += 1;
    }

    fn clear_quadratic_geqs(&mut self) {
        unsafe {
            let status = qp_delete_quadratic_geqs(self.env, self.qp, 0, self.n_quad_geqs);
            if status != 0 {
                panic!();
            }
        }
        self.n_quad_geqs = 0;
    }

    // even on failure, xs may be mutated
    fn run(&mut self, xs: &mut [f64]) -> Option<f64> {
        assert_eq!(xs.len(), self.n as usize);
        unsafe {
            let mut obj_val = 0.0;
            let status = qp_run(self.env, self.qp, &mut obj_val, xs.as_mut_ptr());
            if status == 0 {
                return Some(obj_val);
            }
        }
        None
    }
}

impl Drop for QuadProg {
    fn drop(&mut self) {
        unsafe {
            let status = qp_destroy(self.env, self.qp);
            if status != 0 {
                panic!();
            }
        }
    }
}

fn solve_quadprog(h_mat: &[f64], f: &[f64],
                  a_mat: &[f64], b: &[f64],
                  a_eq_mat: &[f64], b_eq: &[f64],
                  lb: &[f64], ub: &[f64]) -> Option<(f64, Vec<f64>)> {
    let n = f.len();
    assert_eq!(h_mat.len(), n * n);
    let n_le = b.len();
    assert_eq!(a_mat.len(), b.len() * n);
    let n_eq = b_eq.len();
    assert_eq!(a_eq_mat.len(), b_eq.len() * n);
    assert_eq!(lb.len(), n);
    assert_eq!(ub.len(), n);
    let mut x = vec![0.0; n];
    let mut obj_val = 0.0;
    unsafe {
        let status = quadprog(n as i32, h_mat.as_ptr(), f.as_ptr(),
                              n_le as i32, a_mat.as_ptr(), b.as_ptr(),
                              n_eq as i32, a_eq_mat.as_ptr(), b_eq.as_ptr(),
                              lb.as_ptr(), ub.as_ptr(),
                              &mut obj_val, x.as_mut_ptr());
        if status != 0 {
            return None;
        }
    }
    Some((obj_val, x))
}

// m is rows of a_mat
// https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
fn matmult_alloc(m: usize, a_mat: &[f64], b_mat: &[f64]) -> Vec<f64> {
    // multiply m*k by k*n for m*n result
    let m = m as i32;
    let k = a_mat.len() as i32 / m;
    let n = b_mat.len() as i32 / k;
    let mut c_mat = vec![0.0; (m * n) as usize];
    unsafe {
        dgemm(b'n', b'n', n, m, k, 1.0, b_mat, n, a_mat, k, 0.0, &mut c_mat, n);
    }
    c_mat
}

// m is rows of a_mat
// https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
fn matmult(m: usize, a_mat: &[f64], b_mat: &[f64], c_mat: &mut [f64]) {
    // multiply m*k by k*n for m*n result
    let m = m as i32;
    let k = a_mat.len() as i32 / m;
    let n = b_mat.len() as i32 / k;
    assert_eq!(c_mat.len(), (m * n) as usize);
    unsafe {
        dgemm(b'n', b'n', n, m, k, 1.0, b_mat, n, a_mat, k, 0.0, c_mat, n);
    }
}

// k is rows of a_mat
fn matmult_atb(k: usize, a_mat: &[f64], b_mat: &[f64], c_mat: &mut [f64]) {
    // multiply m*k by k*n for m*n result
    let k = k as i32;
    let m = a_mat.len() as i32 / k;
    let n = b_mat.len() as i32 / k;
    assert_eq!(c_mat.len(), (m * n) as usize);
    unsafe {
        dgemm(b'n', b't', n, m, k, 1.0, b_mat, n, a_mat, m, 0.0, c_mat, n);
    }
}

// fn matmult_abt(m: usize, a_mat: &[f64], b_mat: &[f64], c_mat: &mut [f64]) {
//     matmult_transpose(m, a_mat, false, b_mat, true, c_mat);
// }
//
// fn matmult_atbt(m: usize, a_mat: &[f64], b_mat: &[f64], c_mat: &mut [f64]) {
//     matmult_transpose(m, a_mat, true, b_mat, true, c_mat);
// }

fn solve_ax_b(n: usize, a_mat: &mut [f64], b: &mut [f64]) {
    let mut pivots = vec![0; n];
    let n = n as i32;
    let n_rhs = b.len() as i32 / n;
    let mut output_status = 0;
    unsafe {
        dgesv(n, n_rhs, a_mat, n, &mut pivots, b, n, &mut output_status);
    }
    if output_status != 0 {
        panic!("Ax=b solver failed: {}", output_status);
    }
}

fn lqr_ltv<F, G>(n: usize, a_fun: &F, b_fun: &G, q: Vec<f64>, mut r: Vec<f64>, t_span: &[f64]) -> Vec<Vec<f64>>
where F: Fn(usize, &mut [f64]),
      G: Fn(usize, &mut [f64]) {
    let n_steps = t_span.len();
    let mut p = vec![0.0; q.len()];
    let mut ks = vec![vec![0.0; n * r.len()]; n_steps];

    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n];

    let mut p_a = vec![0.0; n * n];
    let mut at_p = vec![0.0; n * n];
    let mut p_b = vec![0.0; n];
    let mut term3 = vec![0.0; n * n];

    for i in (0..n_steps-1).rev() {
        a_fun(i, &mut a);
        b_fun(i, &mut b);

        // P{i} = P_ + (tSpan(i+1)-tSpan(i)) * ( A_*P_ + A_'*P_ - P_*B_*(R\(B_'*P_)) + Q);
        let dt = t_span[i + 1] - t_span[i];
        matmult(n, &p, &a, &mut p_a);
        // println!("{:?} * {:?} = {:?}", &p, &a, &p_a);
        matmult_atb(n, &a, &p, &mut at_p);
        // println!("{:?}^T * {:?} = {:?}", &a, &p, &at_p);
        matmult(n, &p, &b, &mut p_b);
        // println!("{:?} * {:?} = {:?}", &p, &b, &p_b);
        // K{i} = R\(B_'*P_);
        matmult_atb(n, &b, &p, &mut ks[i]);
        // println!("{:?}^T * {:?} = {:?}", &b, &p, &ks[i]);
        // println!("Solving {:?} x = {:?}", &r, &ks[i]);
        solve_ax_b(r.len(), &mut r, &mut ks[i]);
        // println!("Got ks[{}] = {:?}\n", i + 1, &ks[i]);
        matmult(n, &p_b, &ks[i], &mut term3);
        // println!("{:?} * {:?} = {:?}", &p_b, &ks[i], &term3);
        for i in 0..n*n {
            p[i] += dt * (p_a[i] + at_p[i] - term3[i] + q[i]);
        }
        // println!("p_old + dt * (p_a + at_p + term3 + q) = {:?}", &p);
    }

    ks
}

fn ode1<F>(mut f: F, t_span: &[f64], x0: &[f64]) -> Vec<Vec<f64>>
where F: FnMut(usize, &[f64], &mut [f64]) {
    let n_steps = t_span.len();
    let mut xs = vec![vec![0.0; x0.len()]; n_steps];
    let mut x_new = x0.to_vec();
    let mut f_val = vec![0.0; x0.len()];
    xs[0].copy_from_slice(&x_new);
    for i in 1..n_steps {
        let dt = t_span[i] - t_span[i - 1];
        f(i, &xs[i - 1], &mut f_val);
        for j in 0..x0.len() {
            x_new[j] += dt * f_val[j];
        }
        xs[i].copy_from_slice(&x_new);
    }
    xs
}

// copy matrix b (whatever * m) to row, col in matrix a (whatever * n).
fn copy_mat(n: usize, a: &mut [f64], row: usize, col: usize, m: usize, b: &[f64]) {
    let mut target_i = n * row + col;
    let rows_b = b.len() / m;
    for j in 0..rows_b {
        a[target_i..target_i+m].copy_from_slice(&b[(j * m)..(j * m + m)]);
        target_i += n;
    }
}

// for RK4 we have
// X_i+1 = X_i + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
// k1 = dt * f(t, X_i)
// k2 = dt * f(t + dt/2, X_i + k1/2)
// k3 = dt * f(t + dt/2, X_i + k2/2)
// k4 = dt * f(t + dt, X_i + k3)
//
// k1 = dt * (A_i * X_i + B_i * U_i)
// k2 = dt * (A_i+.5 * (X_i + k1/2) + B_i+.5 * U_i)
// k3 = dt * (A_i+.5 * (X_i + k2/2) + B_i+.5 * U_i)
// k4 = dt * (A_i+1 * (X_i + k3) + B_i+1 * U_i)
//
// k2 = dt * (A_i+.5 * (X_i + dt/2 * A_i * X_i + dt/2 * B_i * U_i) + B_i+.5 * U_i)
// k2 = dt * (A_i+.5 * X_i + dt/2 * A_i+.5 * A_i * X_i + dt/2 * A_i+.5 * B_i * U_i + B_i+.5 * U_i)
// k2 = dt * (A_i+.5 + dt/2 * A_i+.5 * A_i) * X_i + dt * (dt/2 * A_i+.5 * B_i + B_i+.5) * U_i
//
/*
k1 := h * (A1 * X1 + B1 * U1);
k2 := h * (A2 * (X1 + k1/2) + B2 * U1);
k3 := h * (A2 * (X1 + k2/2) + B2 * U1);
k4 := h * (A3 * (X1 + k3) + B3 * U1);
X2 := X1 + 1/6 * (k1 + 2*k2 + 2*k3 + k4);
XPart := Coef(X2, X1, 1);
UPart := Coef(X2, U1, 1);

0 = XPart * X_i + UPart * U_i - X_i+1

Simplify(XPart)
(h^4*A1*A2^2*A3+2*h^3*A1*A2^2+2*h^3*A2^2*A3+4*h^2*A1*A2+4*h^2*A2^2+4*h^2*A2*A3+4*h*A1+16*h*A2+4*h*A3+24)/24
Simplify(UPart)
(h^4*B1*A2^2*A3+2*h^3*B1*A2^2+2*h^3*A2*B2*A3+4*h^2*B1*A2+4*h^2*A2*B2+4*h^2*B2*A3+4*h*B1+16*h*B2+4*h*B3)/24
*/
#[allow(dead_code)]
fn rk4_constraints<F, G>(n: usize, m: usize, idx: usize, horizon: usize, dt: f64,
                         a_fun: &F, b_fun: &G,
                         aeq_row_idxs: &mut [i32], aeq_col_idxs: &mut [i32],
                         aeq_coefs: &mut [f64])
where F: Fn(f64, &mut [f64]),
      G: Fn(f64, &mut [f64]) {
    let mut xpart = vec![0.0; n * n];
    let mut upart = vec![0.0; n * m];

    let mut a1 = vec![0.0; n * n];
    let mut a2 = vec![0.0; n * n];
    let mut a3 = vec![0.0; n * n];

    let mut a2a1 = vec![0.0; n * n];
    let mut a2a2 = vec![0.0; n * n];
    let mut a3a2 = vec![0.0; n * n];
    let mut a2a2a1 = vec![0.0; n * n];
    let mut a3a2a2 = vec![0.0; n * n];
    let mut a3a2a2a1 = vec![0.0; n * n];

    let mut b1 = vec![0.0; n * m];
    let mut b2 = vec![0.0; n * m];
    let mut b3 = vec![0.0; n * m];

    let mut a2b1 = vec![0.0; n * m];
    let mut a2b2 = vec![0.0; n * m];
    let mut a3b2 = vec![0.0; n * m];
    let mut a2a2b1 = vec![0.0; n * m];
    let mut a3a2b2 = vec![0.0; n * m];
    let mut a3a2a2b1 = vec![0.0; n * m];

    let mut aeq_i = n; // after initial conditions
    for i in 0..horizon-1 {
        // - X_i+1
        let row = i * n + n;
        for j in 0..n {
            aeq_row_idxs[aeq_i] = (row + j) as i32;
            aeq_col_idxs[aeq_i] = (row + j) as i32;
            aeq_coefs[aeq_i] = -1.0;
            aeq_i += 1;
        }

        // XPart * X_i
        a_fun((i + idx) as f64, &mut a1);
        a_fun((i + idx) as f64 + 0.5, &mut a2);
        a_fun((i + idx) as f64 + 1.0, &mut a3);

        // (4*h*A1 + 16*h*A2 + 4*h*A3)/24
        for k in 0..n*n {
            xpart[k] = dt * (4.0 / 24.0) * (a1[k] + 4.0 * a2[k] + a3[k]);
        }
        // 24/24
        for k in 0..n {
            xpart[k*n + k] += 1.0;
        }

        // (4*h^2*A2*A1 + 4*h^2*A2^2 + 4*h^2*A3*A2)/24
        matmult(n, &a2, &a1, &mut a2a1);
        matmult(n, &a2, &a2, &mut a2a2);
        matmult(n, &a3, &a2, &mut a3a2);
        for k in 0..n*n {
            xpart[k] += dt * dt * (4.0 / 24.0) * (a2a1[k] + a2a2[k] + a3a2[k]);
        }

        // (2*h^3*A2^2*A1 + 2*h^3*A3*A2^2)/24
        matmult(n, &a2a2, &a1, &mut a2a2a1);
        matmult(n, &a3, &a2a2, &mut a3a2a2);
        for k in 0..n*n {
            xpart[k] += dt.powi(3) * (2.0 / 24.0) * (a2a2a1[k] + a3a2a2[k]);
        }

        // h^4*A3*A2^2*A1/24
        matmult(n, &a3a2a2, &a1, &mut a3a2a2a1);
        for k in 0..n*n {
            xpart[k] += dt.powi(4) * a3a2a2a1[k] * (1.0 / 24.0);
        }

        for j in 0..n {
            for k in 0..n {
                aeq_row_idxs[aeq_i] = (row + j) as i32;
                aeq_col_idxs[aeq_i] = (row - n + k) as i32;
                aeq_coefs[aeq_i] = xpart[j * n + k];
                aeq_i += 1;
            }
        }

        // UPart
        b_fun((i + idx) as f64, &mut b1);
        b_fun((i + idx) as f64 + 0.5, &mut b2);
        b_fun((i + idx) as f64 + 1.0, &mut b3);

        // (4*h*B1 + 16*h*B2 + 4*h*B3)/24
        for k in 0..n*m {
            upart[k] = dt * (4.0 / 24.0) * (b1[k] + 4.0 * b2[k] + b3[k]);
        }

        // (4*h^2*A2*B1 + 4*h^2*A2*B2 + 4*h^2*A3*B2)/24
        matmult(n, &a2, &b1, &mut a2b1);
        matmult(n, &a2, &b2, &mut a2b2);
        matmult(n, &a3, &b2, &mut a3b2);
        for k in 0..n*m {
            upart[k] += dt * dt * (4.0 / 24.0) * (a2b1[k] + a2b2[k] + a3b2[k]);
        }

        // (2*h^3*A2^2*B1 + 2*h^3*A3*A2*B2)/24
        matmult(n, &a2a2, &b1, &mut a2a2b1);
        matmult(n, &a3a2, &b2, &mut a3a2b2);
        for k in 0..n*m {
            upart[k] += dt.powi(3) * (2.0 / 24.0) * (a2a2b1[k] + a3a2b2[k]);
        }

        // (h^4*A3*A2^2*B1)/24
        matmult(n, &a3a2a2, &b1, &mut a3a2a2b1);
        for k in 0..n*m {
            upart[k] += dt.powi(4) * (1.0 / 24.0) * a3a2a2b1[k];
        }

        for j in 0..n {
            for k in 0..m {
                aeq_row_idxs[aeq_i] = (row + j) as i32;
                aeq_col_idxs[aeq_i] = (i * m + n * horizon + k) as i32;
                aeq_coefs[aeq_i] = upart[j * m + k];
                aeq_i += 1;
            }
        }
    }
}

// for RK2 we have
// X_i+1 = X_i + k2
// k1 = h * f(t, X_i)
// k2 = h * f(t + h/2, X_i + k1/2)
//
// k1 = h * (A_i * X_i + B_i * U_i)
// k2 = h * (A_i+.5 * (X_i + k1/2) + B_i+.5 * U_i)
/*
k1 := h * (A1 * X1 + B1 * U1);
k2 := h * (A2 * (X1 + k1/2) + B2 * U1);
X2 := X1 + k2;
XPart := Coef(X2, X1, 1);
UPart := Coef(X2, U1, 1);

0 = XPart * X_i + UPart * U_i - X_i+1

Simplify(XPart)
(h^2*A2*A1 + 2*h*A2+2)/2
Simplify(UPart)
(h^2*A2*B1 + 2*h*B2)/2
*/
#[allow(dead_code)]
fn rk2_constraints<F, G>(n: usize, m: usize, idx: usize, horizon: usize, dt: f64,
                         a_fun: &F, b_fun: &G,
                         aeq_row_idxs: &mut [i32], aeq_col_idxs: &mut [i32],
                         aeq_coefs: &mut [f64])
where F: Fn(f64, &mut [f64]),
      G: Fn(f64, &mut [f64]) {
    let mut xpart = vec![0.0; n * n];
    let mut upart = vec![0.0; n * m];

    let mut a1 = vec![0.0; n * n];
    let mut a2 = vec![0.0; n * n];

    let mut a2a1 = vec![0.0; n * n];

    let mut b1 = vec![0.0; n * m];
    let mut b2 = vec![0.0; n * m];

    let mut a2b1 = vec![0.0; n * m];

    let mut aeq_i = n; // after initial conditions
    for i in 0..horizon-1 {
        // - X_i+1
        let row = i * n + n;
        for j in 0..n {
            aeq_row_idxs[aeq_i] = (row + j) as i32;
            aeq_col_idxs[aeq_i] = (row + j) as i32;
            aeq_coefs[aeq_i] = -1.0;
            aeq_i += 1;
        }

        // XPart * X_i
        a_fun((i + idx) as f64, &mut a1);
        a_fun((i + idx) as f64 + 0.5, &mut a2);

        // h*A2
        for k in 0..n*n {
            xpart[k] = dt * a2[k];
        }
        // 2/2
        for k in 0..n {
            xpart[k*n + k] += 1.0;
        }

        // (h^2*A2*A1)/2
        matmult(n, &a2, &a1, &mut a2a1);
        for k in 0..n*n {
            xpart[k] += dt * dt * 0.5 * a2a1[k];
        }

        for j in 0..n {
            for k in 0..n {
                aeq_row_idxs[aeq_i] = (row + j) as i32;
                aeq_col_idxs[aeq_i] = (row - n + k) as i32;
                aeq_coefs[aeq_i] = xpart[j * n + k];
                aeq_i += 1;
            }
        }

        // UPart
        b_fun((i + idx) as f64, &mut b1);
        b_fun((i + idx) as f64 + 0.5, &mut b2);

        // h*B2
        for k in 0..n*m {
            upart[k] = dt * b2[k];
        }

        // (h^2*A2*B1)/2
        matmult(n, &a2, &b1, &mut a2b1);
        for k in 0..n*m {
            upart[k] += dt * dt * 0.5 * a2b1[k];
        }

        for j in 0..n {
            for k in 0..m {
                aeq_row_idxs[aeq_i] = (row + j) as i32;
                aeq_col_idxs[aeq_i] = (i * m + n * horizon + k) as i32;
                aeq_coefs[aeq_i] = upart[j * m + k];
                aeq_i += 1;
            }
        }
    }
}

#[allow(dead_code)]
fn euler_constraints<F, G>(n: usize, m: usize, idx: usize, horizon: usize, dt: f64,
                           a_fun: &F, b_fun: &G,
                           aeq_row_idxs: &mut [i32], aeq_col_idxs: &mut [i32],
                           aeq_coefs: &mut [f64])
where F: Fn(f64, &mut [f64]),
      G: Fn(f64, &mut [f64]) {
    let mut a_mat = vec![0.0; n * n];
    let mut b_mat = vec![0.0; n * m];
    // euler integration equality constraints
    // X_i+1 = X_i + dt * f(t, X_i)
    // X_i+1 = X_i + dt * (A_i * X_i + B_i * U_i)
    // X_i + dt * (A_i * X_i + B_i * U_i) - X_i+1 = 0
    let mut aeq_i = n; // after initial conditions
    for i in 0..horizon-1 {
        // - X_i+1
        let row = i * n + n;
        for j in 0..n {
            aeq_row_idxs[aeq_i] = (row + j) as i32;
            aeq_col_idxs[aeq_i] = (row + j) as i32;
            aeq_coefs[aeq_i] = -1.0;
            aeq_i += 1;
        }

        // dt * A_i * X_i
        a_fun((i + idx) as f64, &mut a_mat);
        for k in 0..n*n {
            a_mat[k] *= dt;
        }
        // X_i
        for k in 0..n {
            a_mat[k * n + k] += 1.0;
        }
        for j in 0..n {
            for k in 0..n {
                aeq_row_idxs[aeq_i] = (row + j) as i32;
                aeq_col_idxs[aeq_i] = (row - n + k) as i32;
                aeq_coefs[aeq_i] = a_mat[j * n + k];
                aeq_i += 1;
            }
        }

        // dt * B_i * U_i
        b_fun((i + idx) as f64, &mut b_mat);
        for k in 0..n*m {
            b_mat[k] *= dt;
        }
        for j in 0..n {
            for k in 0..m {
                aeq_row_idxs[aeq_i] = (row + j) as i32;
                aeq_col_idxs[aeq_i] = (i * m + n * horizon + k) as i32;
                aeq_coefs[aeq_i] = b_mat[j * m + k];
                aeq_i += 1;
            }
        }
    }
}

fn mpc_ltv<F, G, H>(a_fun: &F, b_fun: &G, q_diag: &[f64], r_diag: &[f64],
                    t_span: &[f64], horizon: usize,
                    ref_x: &[f64], ref_u: &[f64], ode_fun: &H,
                    a_x_constraints: &[f64], b_x_constraints: &[f64],
                    a_u_constraints: &[f64], b_u_constraints: &[f64],
                    x_lb: &[f64], x_ub: &[f64],
                    u_lb: &[f64], u_ub: &[f64], x0: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>)
where F: Fn(f64, &mut [f64]),
      G: Fn(f64, &mut [f64]),
      H: Fn(f64, &[f64], &[f64], &mut[f64]) {
    let n = x0.len();
    assert_eq!(q_diag.len(), n);

    let m = r_diag.len();
    let dt = t_span[1] - t_span[0]; // assume fixed
    let n_steps = t_span.len();

    // states at each time step
    let mut xs = vec![vec![0.0; n]; n_steps];
    // control inputs at each time step
    let mut us = vec![vec![0.0; m]; n_steps - 1];

    // decision variables will correspond first to the xs and then to the us
    // not over the whole time span, but just over the horizon
    let n_dec = n * horizon + m * (horizon - 1);
    let mut quadratic_cost_diag = vec![0.0; n_dec];
    for i in 0..horizon {
        quadratic_cost_diag[i*n..i*n+n].copy_from_slice(&q_diag);
    }
    // same but with r instead of q, offset in the matrix after all the q stuff
    for i in 0..horizon-1 {
        let idx = horizon * n + i * m;
        quadratic_cost_diag[idx..idx+m].copy_from_slice(&r_diag);
    }

    // figure out the number of inequalities per decision variable
    let ineq_per_x = a_x_constraints.len() / n / n;
    assert_eq!(a_x_constraints.len(), ineq_per_x * n * n);
    assert_eq!(b_x_constraints.len(), ineq_per_x * n);

    let ineq_per_u = a_u_constraints.len() / m / m;
    assert_eq!(a_u_constraints.len(), ineq_per_u * m * m);
    assert_eq!(b_u_constraints.len(), ineq_per_u * m);

    let n_ineq = ineq_per_x * n * horizon + ineq_per_u * m * (horizon - 1);
    let mut a_ineq = vec![0.0; n_ineq * n_dec];
    let mut b_ineq = vec![0.0; n_ineq];
    for i in 0..horizon {
        let row = ineq_per_x * n * i;
        let col = n * i;
        copy_mat(n_dec, &mut a_ineq, row, col, n, &a_x_constraints);
        copy_mat(1, &mut b_ineq, row, 0, 1, &b_x_constraints);
    }
    for i in 0..horizon-1 {
        let row = ineq_per_x * n * horizon + ineq_per_u * m * i;
        let col = n * horizon + m * i;
        copy_mat(n_dec, &mut a_ineq, row, col, m, &a_u_constraints);
        copy_mat(1, &mut b_ineq, row, 0, 1, &b_u_constraints);
    }

    // set bounds to infinities for the default
    let cplex_inf = 1.0e+20;
    let mut lb = vec![-cplex_inf; n_dec];
    let mut ub = vec![cplex_inf; n_dec];

    // can be length zero to leave unbounded
    assert!(x_lb.len() == n || x_lb.len() == 0);
    assert_eq!(x_ub.len(), x_lb.len());
    assert!(u_lb.len() == m || u_lb.len() == 0);
    assert_eq!(u_ub.len(), u_lb.len());

    // Finally, for each time step, we set up equality constraints on the states
    // (only as far as our forward horizon)
    // and solve that for the next state and action inputs
    // The first step of the horizon sets up the initial conditions
    // and then each step relates with the previous through euler integration

    let n_eq = n * horizon;
    let mut b_eq = vec![0.0; n_eq];

    xs[0].copy_from_slice(x0);
    let mut x_new = vec![0.0; n];

    let linear_cost = vec![0.0; n_dec];
    let mut qp = QuadProg::new(n_dec, &linear_cost, &lb, &ub, 0, n_eq);
    qp.diagonal_quadratic_cost(&quadratic_cost_diag);

    let mut solved_vars = vec![0.0; n_dec];

    // n coefs for the initial condition identity
    // then for each state variable, each step in the remaining horizon...
    // there are n coefficients from A_i, m from B_i, and 1 from -X_i+1
    // note: rows correspond to constraints, columns to decision variables
    let aeq_coefs_n = n + (n * (horizon - 1)) * (n + m + 1);
    let mut aeq_row_idxs = vec![0; aeq_coefs_n];
    let mut aeq_col_idxs = vec![0; aeq_coefs_n];
    let mut aeq_coefs = vec![0.0; aeq_coefs_n];

    // initial conditions identity
    for i in 0..n {
        aeq_row_idxs[i] = i as i32;
        aeq_col_idxs[i] = i as i32;
        aeq_coefs[i] = 1.0;
    }

    for k in 0..n_steps-1 {
        let horizon_left = n_steps - k;
        let valid_horizon = if horizon_left < horizon { horizon_left } else { horizon };

        // for initial conditions, all other values stay at 0
        b_eq[0..n].copy_from_slice(&xs[k]);
        for i in 0..n {
            b_eq[i] -= ref_x[(n_steps * i + k) * 2];
        }
        // zero invalid horizon coefficients
        if valid_horizon < horizon {
            let start_invalid = n + (n * (valid_horizon - 1)) * (n + m + 1);
            for i in start_invalid..aeq_coefs_n {
                aeq_coefs[i] = 0.0;
            }
            // for i in valid_horizon..horizon {
            //     for j in 0..n {
            //         quadratic_cost_diag[i * n + j] = 0.0;
            //     }
            //     let idx = horizon * n + (i - 1) * m;
            //     for j in 0..m {
            //         quadratic_cost_diag[idx + j] = 0.0;
            //     }
            // }
        }
        euler_constraints(n, m, k, valid_horizon, dt, a_fun, b_fun,
                          &mut aeq_row_idxs, &mut aeq_col_idxs, &mut aeq_coefs);
        // rk4_constraints(n, m, k, valid_horizon, dt, a_fun, b_fun,
        //                 &mut aeq_row_idxs, &mut aeq_col_idxs, &mut aeq_coefs);
        // rk2_constraints(n, m, k, valid_horizon, dt, a_fun, b_fun,
        //                 &mut aeq_row_idxs, &mut aeq_col_idxs, &mut aeq_coefs);

        // if k == 0 {
        //     let mut test_mat = vec![0.0; n_dec * n_eq];
        //     for i in 0..aeq_coefs_n {
        //         test_mat[aeq_row_idxs[i] as usize * n_dec + aeq_col_idxs[i] as usize] = aeq_coefs[i];
        //     }
        //     print!("[");
        //     for i in 0..n_eq {
        //         for j in 0..n_dec {
        //             print!("{} ", test_mat[i * n_dec + j]);
        //         }
        //         println!("; ");
        //     }
        //     println!("]");
        //     panic!("Done for now");
        // }

        qp.sparse_eq_constraints(&aeq_row_idxs, &aeq_col_idxs, &aeq_coefs, &b_eq);

        // bounds are in terms of the reference
        if x_lb.len() > 0 {
            for i in 0..valid_horizon {
                for j in 0..n {
                    lb[n * i + j] = x_lb[j] - ref_x[(n_steps * j + k + i) * 2];
                    ub[n * i + j] = x_ub[j] - ref_x[(n_steps * j + k + i) * 2];
                }
            }
            for i in valid_horizon..horizon {
                for j in 0..n {
                    lb[n * i + j] = -cplex_inf;
                    ub[n * i + j] = cplex_inf;
                }
            }
        }
        if u_lb.len() > 0 {
            for i in 0..valid_horizon-1 {
                let idx = n * horizon + m * i;
                for j in 0..m {
                    lb[idx + j] = u_lb[j] - ref_u[(n_steps * j + k + i) * 2];
                    ub[idx + j] = u_ub[j] - ref_u[(n_steps * j + k + i) * 2];
                }
            }
            for i in valid_horizon-1..horizon-1 {
                let idx = n * horizon + m * i;
                for j in 0..m {
                    lb[idx + j] = -cplex_inf;
                    ub[idx + j] = cplex_inf;
                }
            }
        }
        qp.bounds(&lb, &ub);

        // print!("[");
        // for i in 0..n_dec {
        //     print!("{}; ", ub[i]);
        // }
        // println!("]");
        // panic!("Done for now");

        if let Some(_) = qp.run(&mut solved_vars) {
            // println!("{}", obj_val);
            // println!("{:?}", solved_vars);

            // we actually don't care much where the quadprog thinks we end up
            // xs[k + 1].copy_from_slice(&solved_vars[n..2*n]);
            us[k].copy_from_slice(&solved_vars[n*horizon..n*horizon+m]);
            for i in 0..m {
                us[k][i] += ref_u[(n_steps * i + k) * 2];
            }

            // print!("{:.10} ", us[k][0]);

            // pass through "real world" model to get our next state
            let mut integrate_fun = |t: f64, x: &[f64], x_new: &mut[f64]| {
                ode_fun(k as f64 * dt + t, x, &us[k], x_new);
            };
            rk4_integrate(dt / 8.0, 8, &mut integrate_fun, &xs[k], &mut x_new);
            xs[k + 1].copy_from_slice(&x_new);
        } else {
            panic!("Quadratic program could not be solved on timestep {}", k);
        }
    }

    (xs, us)
}

// x_out can either be the same length as x0, or n_steps times that length
fn rk4_integrate<F>(h: f64, n_steps: usize, f: &mut F, x0: &[f64], x_out: &mut [f64])
where F: FnMut(f64, &[f64], &mut [f64]) {
    let n = x0.len();
    let just_x_result = x_out.len() == n;
    if !just_x_result {
        assert_eq!(n * (n_steps + 1), x_out.len());
    }

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];

    let mut x_tmp = vec![0.0; n];
    let mut x_old = vec![0.0; n];

    x_old.copy_from_slice(x0);
    x_out[0..n].copy_from_slice(x0);

    let mut t = 0.0;
    for i in 0..n_steps {
        {
            f(t, &x_old, &mut k1);

            for j in 0..n { x_tmp[j] = x_old[j] + 0.5 * h * k1[j]; }
            f(t + h * 0.5, &x_tmp, &mut k2);

            for j in 0..n { x_tmp[j] = x_old[j] + 0.5 * h * k2[j]; }
            f(t + h * 0.5, &x_tmp, &mut k3);

            for j in 0..n { x_tmp[j] = x_old[j] + h * k3[j]; }
            f(t + h, &x_tmp, &mut k4);

            for j in 0..n { x_tmp[j] = x_old[j] + h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]); }
        }
        x_old.copy_from_slice(&x_tmp);
        if !just_x_result {
            x_out[i*n+n..i*n+n*2].copy_from_slice(&x_tmp);
        }

        t += h;
    }

    if just_x_result {
        x_out.copy_from_slice(&x_old);
    }
}

fn forward_integrate_bike(controls: &[f64]) -> Vec<f64> {
    let n = 6;
    let m = 2;
    let x0 = [287.0, 5.0, -176.0, 0.0, 2.0, 0.0];
    let dt = 0.01;
    let rk4_steps = 10;
    let all_steps = (controls.len() / m - 1) * rk4_steps;
    let mut x_all = vec![0.0; n * all_steps + n];

    let mut ode_fun = |t: f64, x: &[f64], dx: &mut [f64]| {
        let idx = (t / 0.01 + 0.0001) as usize;
        let idx = if idx == controls.len() / m { idx - 1 } else { idx };
        bike_fun(x, controls[idx * m], controls[idx * m + 1], dx);
    };
    rk4_integrate(dt / rk4_steps as f64, all_steps, &mut ode_fun, &x0, &mut x_all);

    x_all
}

fn rad2deg(r: f64) -> f64 {
    return r * 180.0 / f64::consts::PI;
}

fn bike_fun(x: &[f64], delta_f: f64, mut f_x: f64, dx: &mut [f64]) {
    let nw = 2.;
    let f = 0.01;
    let iz = 2667.0;
    let a = 1.35;
    let b = 1.45;
    let by = 0.27;
    let cy = 1.2;
    let dy = 0.7;
    let ey = -1.6;
    let shy = 0.0;
    let svy = 0.0;
    let m = 1400.0;
    let g = 9.806;

    // slip angle functions in degrees
    let a_f = rad2deg(delta_f - (x[3]+a*x[5]).atan2(x[1]));
    let a_r = rad2deg(-(x[3]-b*x[5]).atan2(x[1]));

    // Nonlinear Tire dynamics
    let phi_yf = (1.0-ey)*(a_f+shy)+(ey/by)*(by*(a_f+shy)).atan();
    let phi_yr = (1.0-ey)*(a_r+shy)+(ey/by)*(by*(a_r+shy)).atan();

    let f_zf=b/(a+b)*m*g;
    let f_yf=f_zf*dy*(cy*(by*phi_yf).atan()).sin()+svy;

    let f_zr=a/(a+b)*m*g;
    let mut f_yr=f_zr*dy*(cy*(by*phi_yr).atan()).sin()+svy;

    let f_total=((nw*f_x).powi(2)+f_yr*f_yr).sqrt();
    let f_max=0.7*m*g;

    if f_total > f_max {
        f_x=f_max/f_total*f_x;
        f_yr=f_max/f_total*f_yr;
    }

    // vehicle dynamics
    dx[0] = x[1]*x[4].cos()-x[3]*x[4].sin();
    dx[1] = (-f*m*g+nw*f_x-f_yf*delta_f.sin())/m+x[3]*x[5];
    dx[2] = x[1]*x[4].sin()+x[3]*x[4].cos();
    dx[3] = (f_yf*delta_f.cos()+f_yr)/m-x[1]*x[5];
    dx[4] = x[5];
    dx[5] = (f_yf*a*delta_f.cos()-f_yr*b)/iz;
}

fn quadprog_test() {
    let h_mat = [1., -1., 1., -1., 2., -2., 1., -2., 4.];
    let f = [2., -3., 1.];
    let a_mat = [];
    let b = [];
    let a_eq_mat = [1., 1., 1.];
    let b_eq = [0.5];
    let lb = [0., 0., 0.];
    let ub = [1., 1., 1.];
    if let Some((obj_val, x)) = solve_quadprog(&h_mat, &f, &a_mat, &b, &a_eq_mat, &b_eq, &lb, &ub) {
        println!("{}", obj_val);
        println!("{:?}", x);
    }
}

fn matmult_test() {
    let a_mat = [1., 1., 0., 0.];
    let b_mat = [1., 2., 3., 4.];
    let c_mat = matmult_alloc(1, &a_mat, &b_mat);
    println!("{:?}", c_mat);
    let c_mat = matmult_alloc(4, &b_mat, &a_mat);
    println!("{:?}", c_mat);
}

fn axb_test() {
    let mut a = [8., 1., 6.,
                 3., 5., 7.,
                 4., 9., 2.];
    let mut b = [15., 15., 15.];
    solve_ax_b(3, &mut a, &mut b);
    println!("{:?}", b);
}

fn lqr_test() {
    let a_fun = |_: usize, a: &mut [f64]| a.copy_from_slice(&[1., 1., 0., 1.]);
    let b_fun = |_: usize, b: &mut [f64]| b.copy_from_slice(&[0., 1.]);
    let x0 = [-1., -1.];
    let n_steps = 10000;
    let t_span = (0..n_steps).map(|i: usize| i as f64 * 5. / (n_steps - 1) as f64).collect::<Vec<_>>();
    let q = vec![1000., 0., 0., 1000.];
    let r = vec![0.1];
    let n = 2;
    let start_time = precise_time_s();
    let ks = lqr_ltv(n, &a_fun, &b_fun, q, r, &t_span);
    println!("LQR took: {} seconds", precise_time_s() - start_time);
    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n];
    let mut b_k = vec![0.0; n * n];
    let ode_fun = move |i: usize, x: &[f64], r: &mut [f64]| {
        // (a_fun(i) - b_fun(i) * ks[i]) * x
        a_fun(i, &mut a);
        b_fun(i, &mut b);
        matmult(n, &b, &ks[i], &mut b_k);
        for j in 0..n*n {
            a[j] -= b_k[j];
        }
        matmult(n, &a, &x, r);
    };
    let start_time = precise_time_s();
    let xs_lqr = ode1(ode_fun, &t_span, &x0);
    println!("Euler took: {} seconds", precise_time_s() - start_time);

    if false {
        let mut x0s = vec![0.0; n_steps];
        let mut x1s = vec![0.0; n_steps];
        for i in 0..n_steps {
            x0s[i] = xs_lqr[i][0];
            x1s[i] = xs_lqr[i][1];
        }

        let ax = Axes2D::new()
            .add(Line2D::new("")
              .data(&x0s, &x1s)
              .color("blue")
              // .marker("x")
              .linestyle("-")
              .linewidth(1.0))
            .xlabel("X")
            .ylabel("Y");
        let mut mpl = Matplotlib::new().unwrap();
        ax.apply(&mut mpl).unwrap();
        mpl.show().unwrap();
        mpl.wait().unwrap();
    }
}

fn is_float_digit(c: char) -> bool {
    return c.is_numeric() || c == '.' || c == '-';
}

fn fill_from_csv(filename: &str, vals: &mut [f64]) -> std::io::Result<()> {
    let mut f = File::open(filename)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let mut buf_i = 0;
    for i in 0..vals.len() {
        let mut next_nondigit = buf_i;
        while is_float_digit(buf[next_nondigit] as char) {
            next_nondigit += 1;
        }
        vals[i] = String::from_utf8(buf[buf_i..next_nondigit].to_owned()).unwrap().parse().unwrap();
        buf_i = next_nondigit;
        while buf_i < buf.len() && !is_float_digit(buf[buf_i] as char) {
            buf_i += 1;
        }
    }

    Ok(())
}

fn load_test_track() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let track_n = 246;
    let mut bl = vec![0.0; 2 * track_n];
    let mut br = vec![0.0; 2 * track_n];
    let mut cline = vec![0.0; 2 * track_n];
    let mut theta = vec![0.0; track_n];
    fill_from_csv("bl.csv", &mut bl).unwrap();
    fill_from_csv("br.csv", &mut br).unwrap();
    fill_from_csv("cline.csv", &mut cline).unwrap();
    fill_from_csv("theta.csv", &mut theta).unwrap();

    (bl, br, cline, theta)
}

fn trajectory_stays_on_track(x_all: &[f64]) -> bool {
    let n = 6;
    let n_steps = x_all.len() / n;
    let (bls, brs, clines, thetas) = load_test_track();
    let track_n = thetas.len();
    let mut trajectory_i = 0;
    let mut track_i = 0;

    let mut trajectory_x = vec![0.0; n_steps];
    let mut trajectory_y = vec![0.0; n_steps];
    for i in 0..n_steps {
        trajectory_x[i] = x_all[i * n];
        trajectory_y[i] = x_all[i * n + 2];
    }

    let mut status = true;

    while trajectory_i < n_steps {
        let x = x_all[trajectory_i * n];
        let y = x_all[trajectory_i * n + 2];
        let theta = thetas[track_i];
        let (sint, cost) = theta.sin_cos();
        // project center of mass onto track axis
        let car_forward = sint * y + cost * x;
        let car_sideways = sint * x + cost * y;

        // and then the sides of the section of track
        let l_sideways = sint * bls[track_i] + cost * bls[track_n + track_i];
        let r_sideways = sint * brs[track_i] + cost * brs[track_n + track_i];
        let forward = sint * clines[track_n + track_i] + cost * clines[track_i];
        // and where the next section starts in the forward dimension
        let next_forward = sint * clines[track_n + track_i + 1] + cost * clines[track_i + 1];

        if car_forward > next_forward {
            track_i += 1;
            continue;
        } else if car_forward < forward {
            track_i -= 1;
            continue;
        }
        if car_sideways < l_sideways || car_sideways > r_sideways {
            status = false;
            break;
        }
        trajectory_i += 1;
    }

    println!("Stays on track: {}", status);

    let ax = Axes2D::new()
        .add(Line2D::new("")
            .data(&bls[0..track_n], &bls[track_n..track_n*2])
            .color("blue")
            .linestyle("-"))
        .add(Line2D::new("")
            .data(&brs[0..track_n], &brs[track_n..track_n*2])
            .color("blue")
            .linestyle("-"))
        .add(Line2D::new("")
            .data(&trajectory_x, &trajectory_y)
            .color("red")
            .linestyle("-"))
        .xlabel("x")
        .ylabel("y");
    let mut mpl = Matplotlib::new().unwrap();
    ax.apply(&mut mpl).unwrap();
    mpl.show().unwrap();
    mpl.wait().unwrap();

    status
}

fn mpc_test() {
    let n = 3;
    let m = 2;
    let total_time = 6.0;
    let ref_steps = 601;
    // extra 2x factor for the half-steps needed by rk2/rk4
    let itermediate_steps = 2;
    let base_step_factor = 1;
    let step_factor = base_step_factor * itermediate_steps;
    let total_steps = ref_steps * step_factor;
    let n_steps = total_steps / itermediate_steps;
    let horizon = 11 * base_step_factor;

    let mut base_ref_x = vec![0.0; n * ref_steps];
    fill_from_csv("ref_x.csv", &mut base_ref_x).unwrap();
    let mut ref_x = vec![0.0; n * total_steps];
    let mut idx = 0;
    for _i in 0..n {
        for _j in 0..ref_steps-1 {
            let ref_idx = idx / step_factor;
            let ref_val = base_ref_x[ref_idx];
            let delta = (base_ref_x[ref_idx + 1] - ref_val) / step_factor as f64;
            for k in 0..step_factor {
                ref_x[idx] = ref_val + k as f64 * delta;
                idx += 1;
            }
        }
        // last element just repeats
        let ref_val = base_ref_x[idx / step_factor];
        for _k in 0..step_factor {
            ref_x[idx] = ref_val;
            idx += 1;
        }
    }

    let mut base_ref_u = vec![0.0; m * ref_steps];
    fill_from_csv("ref_u.csv", &mut base_ref_u).unwrap();
    let mut ref_u = vec![0.0; m * total_steps];
    let mut idx = 0;
    for _i in 0..m {
        for _j in 0..ref_steps-1 {
            let ref_idx = idx / step_factor;
            let ref_val = base_ref_u[ref_idx];
            let delta = (base_ref_u[ref_idx + 1] - ref_val) / step_factor as f64;
            for k in 0..step_factor {
                ref_u[idx] = ref_val + k as f64 * delta;
                idx += 1;
            }
        }
        // last element just repeats
        let ref_val = base_ref_u[idx / step_factor];
        for _k in 0..step_factor {
            ref_u[idx] = ref_val;
            idx += 1;
        }
    }

    let r_xs = &ref_x[0..total_steps];
    let r_ys = &ref_x[total_steps..total_steps*2];
    let r_psis = &ref_x[total_steps*2..total_steps*3];
    let r_us = &ref_u[0..total_steps];
    let r_deltas = &ref_u[total_steps..total_steps*2];

    let l = 3.0; // wheelbase
    let rb = 1.5; // rear wheel to center of mass

    let a_fun = |idx: f64, a: &mut [f64]| {
        let idx = (idx * 2.0).round() as usize;
        for i in 0..9 { a[i] = 0.0; }
        // let u_val = r_us[idx];
        // let psi_val = r_psis[idx];
        // let delta_val = r_deltas[idx];
        // let f1 = -r_psis[idx].sin();
        // let f2 = r_deltas[idx].tan();
        // let f3 = r_psis[idx].cos();
        a[2] = r_us[idx] * (-r_psis[idx].sin() - rb / l * r_deltas[idx].tan() * r_psis[idx].cos()); // dpsi/du
        a[5] = r_us[idx] * (r_psis[idx].cos() - rb / l * r_deltas[idx].tan() * r_psis[idx].sin()); // dpsi/ddelta
    };
    let b_fun = |idx: f64, b: &mut [f64]| {
        let idx = (idx * 2.0).round() as usize;
        b[0] = r_psis[idx].cos() - rb / l * r_deltas[idx].tan() * r_psis[idx].sin(); // dx/du
        b[2] = r_psis[idx].sin() + rb / l * r_deltas[idx].tan() * r_psis[idx].cos(); // dy/du
        b[1] = -rb / l * r_us[idx] * r_psis[idx].sin() / r_deltas[idx].cos().powi(2); // dx/ddelta
        b[3] = rb / l * r_us[idx] * r_psis[idx].cos() / r_deltas[idx].cos().powi(2); // dy/ddelta
        b[4] = r_deltas[idx].tan() / l; // dpsi/du
        b[5] = r_us[idx] / (r_deltas[idx].cos().powi(2) * l); // dpsi/ddelta
    };
    let ode_fun = |_t: f64, x: &[f64], u: &[f64], x_new: &mut[f64]| {
        let u_val = u[0];
        let delta_val = u[1];
        x_new[0] = u_val * (x[2].cos() - rb / l * delta_val.tan() * x[2].sin());
        x_new[1] = u_val * (x[2].sin() + rb / l * delta_val.tan() * x[2].cos());
        x_new[2] = u_val / l * delta_val.tan();
    };

    let x0 = [0.25, -0.25, -0.1];

    let t_span = (0..n_steps).map(|i: usize| i as f64 * total_time / (n_steps - 1) as f64).collect::<Vec<_>>();
    let q = vec![1., 1., 0.5];
    let r = vec![0.1, 0.01];

    let a_x_constraints = vec![];
    let b_x_constraints = vec![];
    let a_u_constraints = vec![];
    let b_u_constraints = vec![];

    let x_lb = vec![];
    let x_ub = vec![];
    let u_lb = vec![0.0, -0.5];
    let u_ub = vec![1.0, 0.5];

    let start_time = precise_time_s();
    let (xs, _us) = mpc_ltv(&a_fun, &b_fun, &q, &r, &t_span, horizon, &ref_x, &ref_u, &ode_fun,
                          &a_x_constraints, &b_x_constraints, &a_u_constraints, &b_u_constraints,
                          &x_lb, &x_ub, &u_lb, &u_ub, &x0);
    println!("MPC took: {} seconds", precise_time_s() - start_time);

    // caclulate max distance error between the actual and nominal trajectories,
    // when the x value of the actual trajectory is between or equal to 3 and 4 m
    let mut dist_errors = vec![0.0; n_steps];
    let mut max_dist_error = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i][0] - r_xs[i*2];
        let dy = xs[i][1] - r_ys[i*2];
        let dist_err = (dx * dx + dy * dy).sqrt();
        dist_errors[i] = dist_err;
        if xs[i][0] >= 3.0 && xs[i][0] <= 4.0 {
            if dist_err > max_dist_error {
                max_dist_error = dist_err;
            }
        }
    }
    println!("Max distance error: {}", max_dist_error);

    if false {
        let ax = Axes2D::new()
            .add(Line2D::new("")
                .data(&t_span[199..599], &dist_errors[199..599])
                .color("blue")
                .marker("+")
                .linestyle("-")
                .linewidth(1.0))
            .xlabel("time")
            .ylabel("error");
        let mut mpl = Matplotlib::new().unwrap();
        ax.apply(&mut mpl).unwrap();
        mpl.show().unwrap();
        mpl.wait().unwrap();
    }

    if true {
        let mut xs1 = vec![0.0; n_steps];
        let mut ys1 = vec![0.0; n_steps];
        for i in 0..n_steps {
            xs1[i] = xs[i][0];
            ys1[i] = xs[i][1];
        }

        let ax = Axes2D::new()
            .add(Line2D::new("")
                .data(&xs1, &ys1)
                .color("blue")
                // .marker("+")
                .linestyle("-")
                .linewidth(1.0))
            .add(Line2D::new("")
                .data(&r_xs, &r_ys)
                .color("orange")
                .linestyle("-")
                .linewidth(1.0))
            .xlabel("X")
            .ylabel("Y");
        let mut mpl = Matplotlib::new().unwrap();
        ax.apply(&mut mpl).unwrap();
        mpl.show().unwrap();
        mpl.wait().unwrap();
    }
}

fn bike_test() {
    let controls = vec![0.0; 20];
    let x_all = forward_integrate_bike(&controls);
    let n = 6;
    let n_steps = x_all.len() / n;
    for i in (0..n_steps).step_by(10) {
        for j in 0..n {
            print!("{:9.4}  ", x_all[i * n + j]);
        }
        println!("");
    }
    trajectory_stays_on_track(&x_all);
}

fn main() {
    if false {
        quadprog_test();
        matmult_test();
        axb_test();
    }
    if false {
        lqr_test();
    }
    if false {
        mpc_test();
    }
    bike_test();
}
