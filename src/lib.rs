extern crate libc;
use libc::c_void;
use std::ptr;
use std::fs::File;
use std::io::Read;
use std::f64;
use std::slice;

extern crate lapack;
extern crate blas;
use lapack::*;
use blas::*;

extern crate rustplotlib;
use rustplotlib::{Axes2D, Line2D, Backend};
use rustplotlib::backend::Matplotlib;

extern crate time;
use time::precise_time_s;

#[cfg_attr(target_os = "linux", link(name = "cplex", kind = "static"))]
#[cfg_attr(target_os = "windows", link(name = "cplex1280", kind = "static"))]
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

    fn qp_linear_cost(env: *mut c_void, qp: *mut c_void, n: i32, f: *const f64) -> i32;

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

    fn linear_cost(&mut self, f: &[f64]) {
        assert_eq!(f.len(), self.n as usize);
        unsafe {
            let status = qp_linear_cost(self.env, self.qp, self.n, f.as_ptr());
            if status != 0 {
                panic!();
            }
        }
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
#[allow(dead_code)]
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
fn euler_constraints<F>(n: usize, m: usize, idx: usize, horizon: usize, dt: f64,
                           ab_fun: &F,
                           aeq_row_idxs: &mut [i32], aeq_col_idxs: &mut [i32],
                           aeq_coefs: &mut [f64])
where F: Fn(f64, &mut [f64], &mut [f64]) {
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
        ab_fun((i + idx) as f64, &mut a_mat, &mut b_mat);
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

// currently expects ref_x and ref_u to be twice as long as needed for the number of time steps
// because that is how they would already need to be for a_fun and b_fun in the rk2/rk4 scenario
fn mpc_ltv<F, G, H>(ab_fun: &F, q_diag: &[f64], r_diag: &[f64],
                   t_span: &[f64], horizon: usize,
                   obj_x: &[f64], ref_x: &[f64], ref_u: &[f64], ode_fun: &H,
                   x_constraints_fun: &G,
                   x_lb: &[f64], x_ub: &[f64],
                   u_lb: &[f64], u_ub: &[f64], x0: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>)
where F: Fn(f64, usize, &[f64], &mut [f64], &mut [f64]),
      G: Fn(usize, usize, &[f64], &mut [f64], &mut [f64]),
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

    // set bounds to infinities for the default
    let cplex_inf = 1.0e+20;
    let mut lb = vec![-cplex_inf; n_dec];
    let mut ub = vec![cplex_inf; n_dec];

    // can be length zero to leave unbounded
    assert!(x_lb.len() == n || x_lb.len() == 0);
    assert_eq!(x_ub.len(), x_lb.len());
    assert!(u_lb.len() == m || u_lb.len() == 0);
    assert_eq!(u_ub.len(), u_lb.len());

    // one-time static matrices for setting inequality constraints
    let n_ineq = horizon * 2; // each x, y pair will have linearized limits on two sides
    let mut constraint_idxs = vec![0; n_ineq * 2];
    let mut column_idxs = vec![0; n_ineq * 2];
    let mut le_coefs = vec![0.0; n_ineq * 2];
    let mut le_rhs = vec![0.0; n_ineq];
    for i in 0..horizon {
        for j in 0..=1 {
            let idx = i * 2 + j;
            constraint_idxs[idx * 2] = idx as i32;
            constraint_idxs[idx * 2 + 1] = idx as i32;
            column_idxs[idx * 2] = (n * i) as i32; // x
            column_idxs[idx * 2 + 1] = (n * i + 2) as i32; // y
        }
    }

    // Finally, for each time step, we set up equality constraints on the states
    // (only as far as our forward horizon)
    // and solve that for the next state and action inputs
    // The first step of the horizon sets up the initial conditions
    // and then each step relates with the previous through euler integration

    let n_eq = n * horizon;
    let mut b_eq = vec![0.0; n_eq];

    xs[0].copy_from_slice(x0);
    let mut x_new = vec![0.0; n];

    let mut linear_cost = vec![0.0; n_dec];
    let mut qp = QuadProg::new(n_dec, &linear_cost, &lb, &ub, n_ineq, n_eq);
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

    let mut sum_obj_val = 0.0;

    for k in 0..n_steps-1 {
        let horizon_left = n_steps - k;
        let valid_horizon = if horizon_left < horizon { horizon_left } else { horizon };

        // set up linearized constraints
        for i in 0..valid_horizon {
            x_constraints_fun(k + i, k, &xs[k], &mut le_coefs[i*4..i*4+4], &mut le_rhs[i*2..i*2+2]);
        }
        for i in valid_horizon..horizon {
            le_coefs[i*4..i*4+4].copy_from_slice(&[0.0; 4]);
            le_rhs[i*2..i*2+2].copy_from_slice(&[0.0; 2]);
        }
        qp.sparse_le_constraints(&constraint_idxs, &column_idxs, &le_coefs, &le_rhs);

        // for initial conditions, all other values stay at 0
        b_eq[0..n].copy_from_slice(&xs[k]);
        for i in 0..n {
            b_eq[i] -= ref_x[(n_steps * i + k) * 2];
        }

        // linear cost allows the objective values function to be different from the reference
        // trajectory values
        for i in 0..valid_horizon {
            for j in 0..n {
                let ref_idx = (n_steps * j + k + i) * 2;
                let mut obj_idx = ref_idx;
                if false && i >= horizon - 4 && horizon_left > i + 1 {
                    obj_idx += 2;
                }
                linear_cost[i*n+j] = 2.0 * (obj_x[obj_idx] - ref_x[ref_idx]);
            }
        }
        qp.linear_cost(&linear_cost);

        // zero invalid horizon coefficients
        // perhaps not helpful!! -- without this the last constraint will become duplicated
        // as if we want the 'vehicle' to stop at the end state... which may be a good thing!
        // for whatever reason, zero-ing here causes trouble...
        // if valid_horizon < horizon {
        //     let start_invalid = n + (n * (valid_horizon - 1)) * (n + m + 1);
        //     for i in start_invalid..aeq_coefs_n {
        //         aeq_coefs[i] = 0.0;
        //     }
        // }
        {
            let ab_fun = &|idx: f64, a: &mut [f64], b: &mut [f64]| ab_fun(idx, k, &xs[k], a, b);
            euler_constraints(n, m, k, valid_horizon, dt, ab_fun,
                              &mut aeq_row_idxs, &mut aeq_col_idxs, &mut aeq_coefs);
            // rk4_constraints(n, m, k, valid_horizon, dt, a_fun, b_fun,
            //                 &mut aeq_row_idxs, &mut aeq_col_idxs, &mut aeq_coefs);
            // rk2_constraints(n, m, k, valid_horizon, dt, a_fun, b_fun,
            //                 &mut aeq_row_idxs, &mut aeq_col_idxs, &mut aeq_coefs);
        }
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

        // if ref_u[0].abs() > 0.101 {
        //     print!("[");
        //     for i in 0..n_dec {
        //         print!("{}; ", ub[i]);
        //     }
        //     println!("]");
        //     std::process::exit(0);
        // }

        if let Some(obj_val) = qp.run(&mut solved_vars) {
            sum_obj_val += obj_val;
            // println!("{:?}", solved_vars);

            // we actually don't care much where the quadprog thinks we end up
            // xs[k + 1].copy_from_slice(&solved_vars[n..2*n]);
            us[k].copy_from_slice(&solved_vars[n*horizon..n*horizon+m]);
            // if ref_u[0].abs() > 0.101 {
            //     println!("{:?}", us[k]);
            // }
            for i in 0..m {
                us[k][i] += ref_u[(n_steps * i + k) * 2];
            }

            if false && k == 50 && ref_x[0] != obj_x[0] {
                let total_steps = n_steps * 2;
                let r_xs = &ref_x[0..total_steps];
                let r_ys = &ref_x[total_steps*2..total_steps*3];

                let mut horizon_xs = vec![0.0; horizon];
                let mut horizon_ys = vec![0.0; horizon];

                for i in 0..horizon {
                    horizon_xs[i] = solved_vars[n*i] + r_xs[(k + i) * 2];
                    horizon_ys[i] = solved_vars[n*i+2] + r_ys[(k + i) * 2];
                }

                let ax = Axes2D::new()
                    .add(Line2D::new("")
                        .data(&horizon_xs, &horizon_ys)
                        .color("blue")
                        .linestyle("-")
                        .linewidth(1.0))
                    .add(Line2D::new("")
                        // .data(r_xs, r_ys)
                        .data(&obj_x[0..total_steps], &obj_x[total_steps*2..total_steps*3])
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

    println!("Mean obj val: {:.2}", sum_obj_val / (n_steps-1) as f64);

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

fn is_float_digit(c: char) -> bool {
    return c.is_numeric() || c == '.' || c == '-';
}

fn fill_from_csv(filename: &str, vals: &mut [f64]) -> std::io::Result<()> {
    let mut f = File::open(filename).expect(&format!("Could not open {}", filename)) ;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let mut buf_i = 0;
    for i in 0..vals.len() {
        let mut next_nondigit = buf_i;
        loop {
            if next_nondigit >= buf.len() {
                panic!("Only got {} values from {}, expected {}", i, filename, vals.len());
            }
            if !is_float_digit(buf[next_nondigit] as char) {
                break;
            }
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
    let mut thetas = vec![0.0; track_n];
    fill_from_csv("bl.csv", &mut bl).unwrap();
    fill_from_csv("br.csv", &mut br).unwrap();
    fill_from_csv("cline.csv", &mut cline).unwrap();
    fill_from_csv("theta.csv", &mut thetas).unwrap();

    (bl, br, cline, thetas)
}

fn trajectory_stays_on_track(x_all: &[f64]) -> bool {
    let n = 6;
    let n_steps = x_all.len() / n;
    let (bl, br, cline, thetas) = load_test_track();
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
        let l_sideways = sint * bl[track_i] + cost * bl[track_n + track_i];
        let r_sideways = sint * br[track_i] + cost * br[track_n + track_i];
        let forward = sint * cline[track_n + track_i] + cost * cline[track_i];
        // and where the next section starts in the forward dimension
        let next_forward = sint * cline[track_n + track_i + 1] + cost * cline[track_i + 1];

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
            .data(&bl[0..track_n], &bl[track_n..track_n*2])
            .color("blue")
            .linestyle("-"))
        .add(Line2D::new("")
            .data(&br[0..track_n], &br[track_n..track_n*2])
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

#[allow(dead_code)]
struct TrackProblem {
    bl: Vec<f64>,
    br: Vec<f64>,
    cline: Vec<f64>,
    thetas: Vec<f64>,
    n_obs: usize,
    obs_x: Vec<f64>,
    obs_y: Vec<f64>,
}

#[no_mangle]
pub extern fn solve_obstacle_problem(n_track: i32, bl: *const f64, br: *const f64,
                                     cline: *const f64, thetas: *const f64,
                                     n_obs: i32, obs_x: *const f64, obs_y: *const f64,
                                     n_controls: *mut i32, controls: *mut f64) {
    let n_track = n_track as usize;
    let n_obs = n_obs as usize;
    let n_max_controls = unsafe { *n_controls as usize };

    let bl = unsafe { slice::from_raw_parts(bl, n_track * 2).to_vec() };
    let br = unsafe { slice::from_raw_parts(br, n_track * 2).to_vec() };
    let cline = unsafe { slice::from_raw_parts(cline, n_track * 2).to_vec() };
    let thetas = unsafe { slice::from_raw_parts(thetas, n_track).to_vec() };
    let obs_x = unsafe { slice::from_raw_parts(obs_x, n_obs * 4).to_vec() };
    let obs_y = unsafe { slice::from_raw_parts(obs_y, n_obs * 4).to_vec() };
    let controls = unsafe { slice::from_raw_parts_mut(controls, n_max_controls * 2) };

    let tp = TrackProblem { bl, br, cline, thetas, n_obs, obs_x, obs_y };

    let n_used_controls = solve_control_problem(tp, controls);
    unsafe {
        *n_controls = n_used_controls as i32;
    }
}

fn bike_derivatives(state: &[f64], controls: &[f64], a_mat: &mut [f64], b_mat: &mut [f64]) {
    let a_rows = 6;
    let b_rows = 2;
    // Constants
    let m = 1400.0;
    let nw = 2.0;
    let iz = 2667.0;
    let a = 1.35;
    let b = 1.45;
    let by = 0.27;
    let cy = 1.2;
    let dy = 0.7;
    let ey = -1.6;
    let shy = 0.0;
    let svy = 0.0;
    let g = 9.806;

    let x_ind = 0;
    let u_ind = 1;
    let y_ind = 2;
    let v_ind = 3;
    let psi_ind = 4;
    let r_ind = 5;

    let u = state[u_ind];
    let v = state[v_ind];
    let psi = state[psi_ind];
    let r = state[r_ind];

    let df_ind = 0;
    let fx_ind = 1;

    let df = controls[df_ind];
    let fx = controls[fx_ind];

    for i in 0..a_rows*a_rows {
        a_mat[i] = 0.0;
    }
    for i in 0..b_rows*a_rows {
        b_mat[i] = 0.0;
    }

    a_mat[x_ind * a_rows + u_ind] = psi.cos();
    a_mat[x_ind * a_rows + v_ind] = -psi.sin();
    a_mat[x_ind * a_rows + psi_ind] = -u*psi.sin() - v*psi.cos();

    a_mat[y_ind * a_rows + u_ind] = psi.sin();
    a_mat[y_ind * a_rows + v_ind] = psi.cos();
    a_mat[y_ind * a_rows + psi_ind] = u*psi.cos() - v*psi.sin();

    a_mat[psi_ind * a_rows + r_ind] = 1.0;

    let d_rad2deg = 180.0/f64::consts::PI;

    let d_alpha_f_d_df = d_rad2deg;
    let d_alpha_f_d_v = d_rad2deg * -u/(u*u + (v + a*r).powi(2));
    let d_alpha_f_d_u = d_rad2deg * (v + a*r)/(u*u + (v + a*r).powi(2));
    let d_alpha_f_d_r = d_rad2deg * -a*u/(u*u +(v + a*r).powi(2));

    let d_alpha_r_d_v = d_rad2deg * -u/(u*u + (v - b*r).powi(2));
    let d_alpha_r_d_u = d_rad2deg * (v - b*r)/(u*u + (v - b*r).powi(2));
    let d_alpha_r_d_r = d_rad2deg * b*u/(u*u + (v - b*r).powi(2));

    let alpha_f = d_rad2deg * (df - (v + a*r).atan2(u));
    let d_phi_d_alpha_f = (1.0 - ey) + ey/(1.0 + (by*(alpha_f + shy)).powi(2));
    let alpha_r = d_rad2deg * (-(v - b*r).atan2(u));
    let d_phi_d_alpha_r = (1.0 - ey) + ey/(1.0 + (by*(alpha_r + shy)).powi(2));

    let fzr = a*m*g/(a + b);
    let fzf = b*m*g/(a + b);
    let phi_r = (1.0 - ey)*(alpha_r + shy) + (ey/by)*(by*(alpha_r + shy)).atan();
    let d_fyr_d_phi = fzr*dy*(cy*(by*phi_r).atan()).cos()*cy*by/(1.0 + (by*phi_r).powi(2));
    let phi_f = (1.0 - ey)*(alpha_f + shy) + (ey/by)*(by*(alpha_f + shy)).atan();
    let d_fyf_d_phi = fzf*dy*(cy*(by*phi_f).atan()).cos()*cy*by/(1.0 + (by*phi_f).powi(2));

    let fyr = fzr*dy*(cy*(by*phi_r).atan()).sin() + svy;
    let fyf = fzf*dy*(cy*(by*phi_f).atan()).sin() + svy;

    let fmax = 0.7*m*g;
    let ftotal = ((nw*fx).powi(2) + fyr*fyr).sqrt();

    let (n, d_n_d_fx, d_n_d_fyr);
    if ftotal > fmax {
        n = fmax/ftotal;
        let real_part = -fmax*nw*nw*fx;
        let possibly_nan = ((nw*fx).powi(2) + fyr*fyr).powf(-1.5);
        d_n_d_fx = if real_part == 0.0 { 0.0 } else { real_part * possibly_nan };
        let real_part = -fmax*fyr;
        d_n_d_fyr = if real_part == 0.0 { 0.0 } else { real_part * possibly_nan };
    } else {
        n = 1.0;
        d_n_d_fx = 0.0;
        d_n_d_fyr = 0.0;
    }

    let d_fsyr_d_fx = d_n_d_fx*fyr;
    let d_fsx_d_fx = d_n_d_fx*fx + n;
    let d_fsyr_d_fyr = d_n_d_fyr*fyr + n;
    let d_fsx_d_fyr = d_n_d_fyr*fx;

    let d_fyr_d_r = d_fyr_d_phi*d_phi_d_alpha_r*d_alpha_r_d_r;
    let d_fyr_d_u = d_fyr_d_phi*d_phi_d_alpha_r*d_alpha_r_d_u;
    let d_fyr_d_v = d_fyr_d_phi*d_phi_d_alpha_r*d_alpha_r_d_v;
    let d_fsyr_d_r = d_fsyr_d_fyr*d_fyr_d_r;
    let d_fsyr_d_u = d_fsyr_d_fyr*d_fyr_d_u;
    let d_fsyr_d_v = d_fsyr_d_fyr*d_fyr_d_v;

    let d_fyf_d_r = d_fyf_d_phi*d_phi_d_alpha_f*d_alpha_f_d_r;
    let d_fyf_d_u = d_fyf_d_phi*d_phi_d_alpha_f*d_alpha_f_d_u;
    let d_fyf_d_v = d_fyf_d_phi*d_phi_d_alpha_f*d_alpha_f_d_v;
    let d_fyf_d_df = d_fyf_d_phi*d_phi_d_alpha_f*d_alpha_f_d_df;

    a_mat[r_ind * a_rows + r_ind] = (1.0/iz)*(a*df.cos()*d_fyf_d_r - b*d_fsyr_d_r);
    a_mat[r_ind * a_rows + u_ind] = (1.0/iz)*(a*df.cos()*d_fyf_d_u - b*d_fsyr_d_u);
    a_mat[r_ind * a_rows + v_ind] = (1.0/iz)*(a*df.cos()*d_fyf_d_v - b*d_fsyr_d_v);
    b_mat[r_ind * b_rows + fx_ind] = (-b/iz)*d_fsyr_d_fx;
    b_mat[r_ind * b_rows + df_ind] = (1.0/iz)*(a*df.cos()*d_fyf_d_df - a*fyf*df.sin());

    a_mat[u_ind * a_rows + u_ind] = (1.0/m)*(nw*d_fsx_d_fyr*d_fyr_d_u - d_fyf_d_u*df.sin());
    a_mat[u_ind * a_rows + v_ind] = (1.0/m)*(nw*d_fsx_d_fyr*d_fyr_d_v - d_fyf_d_v*df.sin()) + r;
    a_mat[u_ind * a_rows + r_ind] = (1.0/m)*(nw*d_fsx_d_fyr*d_fyr_d_r - d_fyr_d_r*df.sin()) + v;
    b_mat[u_ind * b_rows + fx_ind] = (1.0/m)*(nw*d_fsx_d_fx);
    b_mat[u_ind * b_rows + df_ind] = (1.0/m)*(-df.sin()*d_fyf_d_df - a*fyf*df.cos());

    a_mat[v_ind * a_rows + u_ind] = (1.0/m)*(df.cos()*d_fyf_d_u + d_fsyr_d_u) - r;
    a_mat[v_ind * a_rows + v_ind] = (1.0/m)*(df.cos()*d_fyf_d_v + d_fsyr_d_v);
    a_mat[v_ind * a_rows + r_ind] = (1.0/m)*(df.cos()*d_fyf_d_r + d_fsyr_d_r) - u;
    b_mat[v_ind * b_rows + fx_ind] = (1.0/m)*d_fsyr_d_fx;
    b_mat[v_ind * b_rows + df_ind] = (1.0/m)*(df.cos()*d_fyf_d_df - df.sin()*fyf);
}

trait FloatIterExt {
    fn float_max(&mut self) -> f64;
    fn float_min(&mut self) -> f64;
}

impl<'a, T> FloatIterExt for T where T: Iterator<Item=&'a f64> {
    fn float_max(&mut self) -> f64 {
        self.fold(f64::NAN, |a: f64, b: &f64| if a > *b { a } else { *b })
    }
    fn float_min(&mut self) -> f64 {
        self.fold(f64::NAN, |a: f64, b: &f64| if a < *b { a } else { *b })
    }
}

fn sq_diff(idx: usize, r_xs: &[f64], r_ys: &[f64], x: f64, y: f64) -> f64 {
    (r_xs[idx] - x).powi(2) + (r_ys[idx] - y).powi(2)
}

fn next_track_idx(old_idx: usize, r_xs: &[f64], r_ys: &[f64], x: f64, y: f64) -> usize {
    // if we are far behind (or ahead of) the trajectory (old_idx), then we should use something closer to
    // us as the reference.
    let dist_sq_lim = 1.0;
    let dist_sq = sq_diff(old_idx, r_xs, r_ys, x, y);
    if dist_sq < dist_sq_lim {
        return old_idx;
    }
    // are we behind or ahead?
    let delta: i32;
    if old_idx == 0 {
        delta = 1;
    } else if old_idx == r_xs.len() - 1 {
        delta = -1;
    } else {
        delta = if sq_diff(old_idx - 1, r_xs, r_ys, x, y) < sq_diff(old_idx + 1, r_xs, r_ys, x, y)
                { -1 } else { 1 };
    }
    let mut last_dist_sq = dist_sq;
    let mut idx = old_idx;
    while idx > 0 && idx < r_xs.len() - 1 {
        idx = (idx as i32 + delta) as usize;
        let dist_sq = sq_diff(idx, r_xs, r_ys, x, y);
        if dist_sq < dist_sq_lim {
            return idx;
        }
        // but don't get worse!
        if dist_sq > last_dist_sq {
            // println!("Don't get worse from {} to {}", last_dist_sq, dist_sq);
            return (idx as i32 - delta) as usize;
        }
        last_dist_sq = dist_sq;
    }
    return idx;
}

fn plot_trajectory(tp: &TrackProblem, n_steps: usize, xs: &[Vec<f64>], r_xs: &[f64], r_ys: &[f64]) {
    let ref_steps = tp.thetas.len();
    let mut xs1 = vec![0.0; n_steps];
    let mut ys1 = vec![0.0; n_steps];
    for i in 0..n_steps {
        xs1[i] = xs[i][0];
        ys1[i] = xs[i][2];
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
        .add(Line2D::new("")
            .data(&tp.bl[0..ref_steps], &tp.bl[ref_steps..ref_steps*2])
            .color("green")
            .linestyle("-"))
        .add(Line2D::new("")
            .data(&tp.br[0..ref_steps], &tp.br[ref_steps..ref_steps*2])
            .color("green")
            .linestyle("-"))
        .xlabel("X")
        .ylabel("Y");
    let mut mpl = Matplotlib::new().unwrap();
    ax.apply(&mut mpl).unwrap();
    mpl.show().unwrap();
    mpl.wait().unwrap();
}

fn report_trajectory_error(n_steps: usize, xs: &[Vec<f64>], r_xs: &[f64], r_ys: &[f64]) {
    // caclulate max distance error between the actual and nominal trajectories,
    // when the x value of the actual trajectory is between or equal to 3 and 4 m
    let mut dist_errors = vec![0.0; n_steps];
    let mut max_dist_error = 0.0;
    let mut max_dist_i = 0;
    let mut max_dist_idx = 0;
    let mut mean_square_error = 0.0;
    for i in 0..xs.len() {
        if i == 737 {
            let _i = 692;
        }
        let idx = next_track_idx(i*2, r_xs, r_ys, xs[i][0], xs[i][2]);
        let dx = xs[i][0] - r_xs[idx];
        let dy = xs[i][2] - r_ys[idx];
        let sq_err = dx * dx + dy * dy;
        mean_square_error += sq_err;
        let dist_err = sq_err.sqrt();
        dist_errors[i] = dist_err;
        if dist_err > max_dist_error {
            max_dist_error = dist_err;
            max_dist_i = i;
            max_dist_idx = idx;
        }
    }
    mean_square_error /= xs.len() as f64;
    println!("Max distance error: {}", max_dist_error);
    println!(" at {}: ({}, {}) with ref ({}, {})", max_dist_i, xs[max_dist_i][0], xs[max_dist_i][2], r_xs[max_dist_idx], r_ys[max_dist_idx]);
    println!("Mean square error: {}\n", mean_square_error);
}

fn lin_interp(pos: f64, vals: &[f64]) -> f64 {
    let idx2 = ((pos + 1.0) as usize).min(vals.len() - 1).max(1);
    let idx1 = idx2 - 1;
    let alpha = idx2 as f64 - pos;
    vals[idx1] * alpha + (1.0 - alpha) * vals[idx2]
}

fn solve_mpc_iteration(tp: &TrackProblem, n: usize, dt: f64, horizon: usize, n_steps: usize, improving: bool,
                       q: &[f64], r: &[f64], obj_x: &[f64], ref_x: &[f64], ref_u: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let total_steps = ref_x.len() / n;
    let r_xs = &ref_x[0..total_steps];
    let r_us = &ref_x[total_steps..total_steps*2];
    let r_ys = &ref_x[total_steps*2..total_steps*3];
    let r_vs = &ref_x[total_steps*3..total_steps*4];
    let r_psis = &ref_x[total_steps*4..total_steps*5];
    let r_rs = &ref_x[total_steps*5..total_steps*6];

    let r_deltas = &ref_u[0..total_steps];
    let r_fxs = &ref_u[total_steps..total_steps*2];

    let constraints_fun = |idx: usize, k: usize, x: &[f64], coefs: &mut [f64], rhs: &mut [f64]| {
        let old_idx = idx * 2;
        let horizon_idx = idx - k;
        if horizon_idx >= 1 {
            return;
        }

        let idx = if false && improving {
            old_idx
        } else {
            let mut new_idx = next_track_idx(old_idx, r_xs, r_ys, x[0], x[2]) + 2 * horizon_idx;
            if new_idx >= r_xs.len() {
                new_idx = r_xs.len() - 1;
            }
            new_idx
        };
        let track_n = tp.thetas.len();
        // let track_pos = (idx * track_n) as f64 / n_steps as f64;
        let theta = tp.thetas[idx];//lin_interp(track_pos, &tp.thetas);
        let bl_x = tp.bl[idx];//lin_interp(track_pos, &tp.bl[0..track_n]);
        let bl_y = tp.bl[track_n+idx];//lin_interp(track_pos, &tp.bl[track_n..track_n*2]);
        let br_x = tp.br[idx];//lin_interp(track_pos, &tp.br[0..track_n]);
        let br_y = tp.br[track_n+idx];//lin_interp(track_pos, &tp.br[track_n..track_n*2]);

        let (sint, cost) = theta.sin_cos();

        // and then the sides of the section of track
        let l_bound = sint * bl_x - cost * bl_y;
        let r_bound = sint * br_x - cost * br_y;

        // left bound becomes a >= constraint
        coefs[0] = -sint;
        coefs[1] = cost;
        coefs[2] = sint;
        coefs[3] = -cost;
        rhs[0] = -l_bound;
        rhs[1] = r_bound;
    };

    if true {
        let track_n = tp.thetas.len();
        let mut small_lines = Vec::new();
        let mut ax = Axes2D::new()
            .add(Line2D::new("")
                .data(&tp.cline[0..track_n], &tp.cline[track_n..track_n*2])
                .color("orange")
                .linestyle("-")
                .linewidth(1.0))
            .add(Line2D::new("")
                .data(&tp.bl[0..track_n], &tp.bl[track_n..track_n*2])
                .color("green")
                .linestyle("-"))
            .add(Line2D::new("")
                .data(&tp.br[0..track_n], &tp.br[track_n..track_n*2])
                .color("green")
                .linestyle("-"))
            .xlabel("X")
            .ylabel("Y");
        for i in (0..track_n).step_by(20) {
            let cx = tp.cline[i];
            let cy = tp.cline[track_n + i];
            let mut coefs = [0.0; 4];
            let mut rhs = [0.0; 2];
            constraints_fun(i/2, i/2, &[cx, 0.0, cy], &mut coefs, &mut rhs);
            let d = cx * coefs[0] + cy * coefs[1];
            for j in 0..=1 {
                let s = if j == 1 { -1.0 } else { 1.0 };
                let x1 = cx + coefs[j*2] * (rhs[j] - s * d) - 0.01 * coefs[j*2+1];
                let x2 = cx + coefs[j*2] * (rhs[j] - s * d) + 0.01 * coefs[j*2+1];
                let y1 = cy + coefs[j*2+1] * (rhs[j] - s * d) - 0.01 * coefs[j*2];
                let y2 = cy + coefs[j*2+1] * (rhs[j] - s * d) + 0.01 * coefs[j*2];
                let x_vals = [x1, x2];
                let y_vals = [y1, y2];
                small_lines.push(x_vals);
                small_lines.push(y_vals);
            }
        }
        for i in 0..small_lines.len()/2 {
            ax = ax.add(Line2D::new("")
                   .data(&small_lines[i*2], &small_lines[i*2 + 1])
                   .color("blue")
                   .linestyle("-")
                   .marker("+")
                   .linewidth(1.0));
        }
        let mut mpl = Matplotlib::new().unwrap();
        ax.apply(&mut mpl).unwrap();
        mpl.show().unwrap();
        mpl.wait().unwrap();
    }

    let ab_fun = |idx: f64, k: usize, x: &[f64], a: &mut [f64], b: &mut [f64]| {
        let old_idx = (idx * 2.0) as usize;
        let idx = if false && improving {
            old_idx
        } else {
            let mut new_idx = next_track_idx(old_idx, r_xs, r_ys, x[0], x[2]) + old_idx - 2 * k;
            if new_idx >= r_xs.len() {
                new_idx = r_xs.len() - 1;
            }
            new_idx
        };
        let state = [r_xs[idx], r_us[idx], r_ys[idx], r_vs[idx], r_psis[idx], r_rs[idx]];
        let controls = [r_deltas[idx], r_fxs[idx]];
        bike_derivatives(&state, &controls, a, b);
    };
    let ode_fun = |_t: f64, x: &[f64], u: &[f64], x_new: &mut[f64]| {
        bike_fun(x, u[0], u[1], x_new);
    };

    let x0 = [287.0, 5.0, -176.0, 0.0, 2.0, 0.0];

    let t_span = (0..n_steps).map(|i: usize| i as f64 * dt).collect::<Vec<_>>();

    let u_lb = vec![-0.5, -5000.0];
    let u_ub = vec![0.5, 2500.0];

    let start_time = precise_time_s();
    let (xs, us) = mpc_ltv(&ab_fun, &q, &r, &t_span, horizon, obj_x, ref_x, ref_u, &ode_fun,
                           &constraints_fun, &[], &[], &u_lb, &u_ub, &x0);
    println!("MPC took: {} seconds", precise_time_s() - start_time);
    report_trajectory_error(n_steps, &xs, &obj_x[0..total_steps], &obj_x[total_steps*2..total_steps*3]);

    (xs, us)
}

#[allow(dead_code)]
fn resample_track(tp: &TrackProblem, n_divisions: usize) -> TrackProblem {
    // first determine spacing between points on the center line
    // and sum the values
    let n_points = tp.thetas.len();
    let cxs = &tp.cline[0..n_points];
    let cys = &tp.cline[n_points..n_points*2];
    let mut sum_distances = vec![0.0; n_points];
    for i in 1..n_points {
        let d = ((cxs[i] - cxs[i-1]).powi(2) + (cys[i] - cys[i-1]).powi(2)).sqrt();
        sum_distances[i] = sum_distances[i - 1] + d;
    }
    let total_dist = sum_distances.last().unwrap();
    println!("Total track length: {}", total_dist);
    println!("Number of reference points: {}\n", n_points);

    let bl_xs = &tp.bl[0..n_points];
    let bl_ys = &tp.bl[n_points..n_points*2];
    let br_xs = &tp.br[0..n_points];
    let br_ys = &tp.br[n_points..n_points*2];

    let mut new_clines = vec![0.0; 2 * n_divisions];
    let mut new_bl = vec![0.0; 2 * n_divisions];
    let mut new_br = vec![0.0; 2 * n_divisions];
    let mut new_thetas = vec![0.0; n_divisions];
    for i in 0..n_divisions {
        let target_d = total_dist / ((n_divisions - 1) as f64) * i as f64;
        let idx2 = sum_distances.iter().position(|&d| d >= target_d).unwrap_or(sum_distances.len() - 1).max(1);
        let idx1 = idx2 - 1;
        let diff = sum_distances[idx2] - sum_distances[idx1];
        let alpha = (sum_distances[idx2] - target_d) / diff;
        new_clines[i] = cxs[idx1] * alpha + (1.0 - alpha) * cxs[idx2];
        new_clines[i+n_divisions] = cys[idx1] * alpha + (1.0 - alpha) * cys[idx2];

        new_bl[i] = bl_xs[idx1] * alpha + (1.0 - alpha) * bl_xs[idx2];
        new_bl[i+n_divisions] = bl_ys[idx1] * alpha + (1.0 - alpha) * bl_ys[idx2];

        new_br[i] = br_xs[idx1] * alpha + (1.0 - alpha) * br_xs[idx2];
        new_br[i+n_divisions] = br_ys[idx1] * alpha + (1.0 - alpha) * br_ys[idx2];

        new_thetas[i] = tp.thetas[idx1] * alpha + (1.0 - alpha) * tp.thetas[idx2];
    }
    TrackProblem {bl: new_bl, br: new_br, cline: new_clines, thetas: new_thetas,
                  n_obs: tp.n_obs, obs_x: tp.obs_x.to_vec(), obs_y: tp.obs_y.to_vec() }
}

fn solve_control_problem(tp: TrackProblem, controls: &mut [f64]) -> usize {
    // general ideas...
    // solve the trajectory in local sections, and for each section...
    // start with a trajectory following the center line
    // (or from a potential field trajectory that avoids the obstacles)
    // then iteratively run the approximate quadratically constrained quadratic program
    // really think about initial conditions based on obvious ideas (need to turn? start turning!)
    // use line-like ellipses to approximate boundaries and obstacles
    // really think about _scaling_ of your values for the optimization

    let n = 6;
    let m = 2;
    // let ref_steps = tp.thetas.len();
    // extra 2x factor for the half-steps needed by rk2/rk4
    // let itermediate_steps = 2;
    // let base_step_factor = 6;
    // let step_factor = base_step_factor * itermediate_steps;
    // let total_steps = ref_steps * step_factor;
    // let n_steps = total_steps / itermediate_steps;
    let n_steps = 1379;
    let total_steps = 1379*2;
    let horizon = 8*2;//11 * base_step_factor;

    let q = vec![2., 0., 2., 0., 1., 0.];
    let r = vec![1000.0, 0.00001];

    let tp = resample_track(&tp, total_steps);
    // let (cline, thetas) = resample_center_line(&tp.cline, &tp.thetas, total_steps);

    // let mut ref_x = vec![0.0; total_steps * n];
    // ref_x[0..total_steps].copy_from_slice(&cline[0..total_steps]);
    // ref_x[total_steps..total_steps*2].copy_from_slice(&vec![20.0; total_steps]);
    // ref_x[total_steps*2..total_steps*3].copy_from_slice(&cline[total_steps..total_steps*2]);
    // ref_x[total_steps*3..total_steps*4].copy_from_slice(&vec![0.0; total_steps]);
    // ref_x[total_steps*4..total_steps*5].copy_from_slice(&thetas);
    // ref_x[total_steps*5..total_steps*6].copy_from_slice(&vec![0.0; total_steps]);
    let mut base_ref_x = vec![0.0; n_steps * n];
    fill_from_csv("matlab/max_ref_x.csv", &mut base_ref_x).unwrap();
    let ref_x = lin_upsample(n, &base_ref_x, 2);
    // let ref_x = ref_x;
    // let mut ref_u = vec![0.0f64; total_steps * m];
    // ref_u[0..total_steps].copy_from_slice(&vec![0.0; total_steps]);
    // ref_u[total_steps..total_steps*2].copy_from_slice(&vec![0.0; total_steps]);
    let mut base_ref_u = vec![0.0; n_steps * m];
    fill_from_csv("matlab/max_ref_u.csv", &mut base_ref_u).unwrap();
    let ref_u = lin_upsample(m, &base_ref_u, 2);
    let ref_u = ref_u;
    let dt = 0.1; //0.0986;
    let (mut xs, mut us) = solve_mpc_iteration(&tp, n, dt, horizon, n_steps, false, &q, &r, &ref_x, &ref_x, &ref_u);

    for _k in 0..0 {
        let mut new_ref_x = vec![0.0; total_steps * n];
        for i in 0..n_steps {
            for j in 0..n {
                new_ref_x[total_steps * j + i * 2] = xs[i][j];
                new_ref_x[total_steps * j + i * 2 + 1] = xs[i][j];
            }
        }
        let mut new_ref_u = vec![0.0; total_steps * m];
        for i in 0..n_steps-1 {
            for j in 0..m {
                new_ref_u[total_steps * j + i * 2] = us[i][j];
                new_ref_u[total_steps * j + i * 2 + 1] = us[i][j];
            }
        }
        let new_ref_u = new_ref_u;

        let mut obj_x = new_ref_x.clone();
        obj_x[0..total_steps].copy_from_slice(&ref_x[0..total_steps]);
        obj_x[total_steps*2..total_steps*3].copy_from_slice(&ref_x[total_steps*2..total_steps*3]);
        // obj_x[total_steps*4..total_steps*5].copy_from_slice(&thetas);

        let q = vec![1., 0., 1., 0., 0., 0.];
        let r = vec![1000.0e6, 0.0001e6];
        // let dt = 0.099 - 0.001/4.0 * k as f64;
        let (new_xs, new_us) = solve_mpc_iteration(&tp, n, dt, 12, n_steps, true, &q, &r, &obj_x, &new_ref_x, &new_ref_u);

        xs = new_xs;
        us = new_us;
    }

    if true {
        let r_xs = &ref_x[0..total_steps];
        let r_ys = &ref_x[total_steps*2..total_steps*3];
        plot_trajectory(&tp, n_steps, &xs, r_xs, r_ys);
    }

    for i in 0..n_steps-1 {
        controls[i] = us[i][0];
        controls[i + n_steps - 1] = us[i][1];
    }

    return n_steps - 1;
}

pub fn run_problem1() {
    let (bl, br, cline, thetas) = load_test_track();
    let tp = TrackProblem { bl, br, cline, thetas,
                            n_obs: 0, obs_x: vec![], obs_y: vec![] };
    let mut controls = vec![0.0; 2048 * 2];
    solve_control_problem(tp, &mut controls);
}

pub fn quadprog_test() {
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

pub fn matmult_test() {
    let a_mat = [1., 1., 0., 0.];
    let b_mat = [1., 2., 3., 4.];
    let c_mat = matmult_alloc(1, &a_mat, &b_mat);
    println!("{:?}", c_mat);
    let c_mat = matmult_alloc(4, &b_mat, &a_mat);
    println!("{:?}", c_mat);
}

pub fn axb_test() {
    let mut a = [8., 1., 6.,
                 3., 5., 7.,
                 4., 9., 2.];
    let mut b = [15., 15., 15.];
    solve_ax_b(3, &mut a, &mut b);
    println!("{:?}", b);
}

pub fn lqr_test() {
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

fn lin_upsample(n: usize, base_vals: &[f64], step_factor: usize) -> Vec<f64> {
    let ref_steps = base_vals.len() / n;
    let total_steps = ref_steps * step_factor;
    let mut vals = vec![0.0; n * total_steps];
    let mut idx = 0;
    for _i in 0..n {
        for _j in 0..ref_steps-1 {
            let ref_idx = idx / step_factor;
            let ref_val = base_vals[ref_idx];
            let delta = (base_vals[ref_idx + 1] - ref_val) / step_factor as f64;
            for k in 0..step_factor {
                vals[idx] = ref_val + k as f64 * delta;
                idx += 1;
            }
        }
        // last element just repeats
        let ref_val = base_vals[idx / step_factor];
        for _k in 0..step_factor {
            vals[idx] = ref_val;
            idx += 1;
        }
    }
    vals
}

pub fn mpc_test() {
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
    let ref_x = lin_upsample(n, &base_ref_x, step_factor);

    let mut base_ref_u = vec![0.0; m * ref_steps];
    fill_from_csv("ref_u.csv", &mut base_ref_u).unwrap();
    let ref_u = lin_upsample(m, &base_ref_u, step_factor);

    let r_xs = &ref_x[0..total_steps];
    let r_ys = &ref_x[total_steps..total_steps*2];
    let r_psis = &ref_x[total_steps*2..total_steps*3];
    let r_us = &ref_u[0..total_steps];
    let r_deltas = &ref_u[total_steps..total_steps*2];

    let l = 3.0; // wheelbase
    let rb = 1.5; // rear wheel to center of mass

    let ab_fun = |idx: f64, _: usize, _: &[f64], a: &mut [f64], b: &mut [f64]| {
        let idx = (idx * 2.0).round() as usize;
        for i in 0..9 { a[i] = 0.0; }
        a[2] = r_us[idx] * (-r_psis[idx].sin() - rb / l * r_deltas[idx].tan() * r_psis[idx].cos()); // dpsi/du
        a[5] = r_us[idx] * (r_psis[idx].cos() - rb / l * r_deltas[idx].tan() * r_psis[idx].sin()); // dpsi/ddelta

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

    let x_lb = vec![];
    let x_ub = vec![];
    let u_lb = vec![0.0, -0.5];
    let u_ub = vec![1.0, 0.5];

    let no_constraints_fun = |_idx: usize, _k: usize, _x: &[f64], _coefs: &mut [f64], _rhs: &mut [f64]| {};

    let start_time = precise_time_s();
    let (xs, _us) = mpc_ltv(&ab_fun, &q, &r, &t_span, horizon, &ref_x, &ref_x, &ref_u, &ode_fun,
                            &no_constraints_fun, &x_lb, &x_ub, &u_lb, &u_ub, &x0);
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

pub fn bike_test() {
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

pub fn bike_derivative_test() {
    let state = [287.0, 5.0, -176.0, 0.0, 2.0, 0.0];
    let controls = [0.0, 0.0];
    let mut analytic_a = [0.0; 6*6];
    let mut analytic_b = [0.0; 2*6];
    bike_derivatives(&state, &controls, &mut analytic_a, &mut analytic_b);

    let delta = f64::EPSILON.sqrt();

    let mut numeric_a = [0.0; 6*6];
    let mut numeric_b = [0.0; 2*6];
    let mut dx1 = [0.0; 6];
    let mut dx2 = [0.0; 6];
    let mut state_mut = [0.0; 6];
    for i in 0..6 {
        state_mut.copy_from_slice(&state);
        let diff = state[i] * delta + delta;
        state_mut[i] -= diff;
        bike_fun(&state_mut, 0.0, 0.0, &mut dx1);
        state_mut[i] += 2.0 * diff;
        bike_fun(&state_mut, 0.0, 0.0, &mut dx2);

        for j in 0..6 {
            numeric_a[j * 6 + i] = (dx2[j] - dx1[j]) / (2.0 * diff);
        }
    }

    let mut dx0 = [0.0; 6];
    bike_fun(&state, 0.0, 0.0, &mut dx0);

    for i in 0..2 {
        let delta_f = if i == 0 { -delta } else { 0.0 };
        let f_x = if i == 1 { -delta } else { 0.0 };
        bike_fun(&state, delta_f, f_x, &mut dx1);
        let delta_f = if i == 0 { delta } else { 0.0 };
        let f_x = if i == 1 { delta } else { 0.0 };
        bike_fun(&state, delta_f, f_x, &mut dx2);

        for j in 0..6 {
            numeric_b[j * 2 + i] = (dx2[j] - dx1[j]) / (2.0 * delta);
        }
    }

    println!("Analytic A       Numeric A");
    for i in 0..6*6 {
        println!("{:10.6}\t{:10.6}", analytic_a[i], numeric_a[i]);
        if i > 0 && (i % 6) == 5 {
            println!();
        }
    }
    println!("Analytic B       Numeric B");
    for i in 0..2*6 {
        println!("{:10.6}\t{:10.6}", analytic_b[i], numeric_b[i]);
        if i > 0 && (i % 2) == 1 {
            println!();
        }
    }
}
