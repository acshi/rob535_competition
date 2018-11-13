extern crate libc;
use libc::size_t;

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
#[link(name = "cplexqp", kind = "static")]
extern {
    fn quadprog(n: size_t, H: *const f64, f: *const f64,
                n_le: size_t, A: *const f64, b: *const f64,
                n_eq: size_t, A_eq: *const f64, b_eq: *const f64,
                lb: *const f64, ub: *const f64,
                obj_val: *mut f64, x_out: *mut f64) -> i32;
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
        let status = quadprog(n as size_t, h_mat.as_ptr(), f.as_ptr(),
                              n_le as size_t, a_mat.as_ptr(), b.as_ptr(),
                              n_eq as size_t, a_eq_mat.as_ptr(), b_eq.as_ptr(),
                              lb.as_ptr(), ub.as_ptr(),
                              &mut obj_val, x.as_mut_ptr());
        if status != 1 {
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

// COLUMN-MAJOR copy matrix b (whatever * m) to row, col in matrix a (whatever * n).
// fn copy_cmat(n: usize, a: &mut [f64], row: usize, col: usize, m: usize, b: &[f64]) {
//     let rows_a = a.len() / n;
//     let mut target_i = rows_a * col + row;
//     let p = b.len() / m;
//     for j in 0..m {
//         a[target_i..target_i+p].copy_from_slice(&b[(j * p)..(j * p + p)]);
//         target_i += rows_a;
//     }
// }

fn mpc_ltv<F, G>(a_fun: &F, b_fun: &G, q: &[f64], r: &[f64], t_span: &[f64],
                 horizon: usize, a_x_constraints: &[f64], b_x_constraints: &[f64],
                 a_u_constraints: &[f64], b_u_constraints: &[f64], x0: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>)
where F: Fn(usize, &mut [f64]),
      G: Fn(usize, &mut [f64]) {
    let n = x0.len();
    assert_eq!(q.len(), n * n);

    let m = r.len();
    let dt = t_span[1] - t_span[0]; // assume fixed
    let n_steps = t_span.len();

    // states at each time step
    let mut xs = vec![vec![0.0; n]; n_steps];
    // control inputs at each time step
    let mut us = vec![vec![0.0; m]; n_steps - 1];

    // decision variables will correspond first to the xs and then to the us
    // not over the whole time span, but just over the horizon
    let n_dec = n * horizon + m * (horizon - 1);
    let mut quadratic_cost = vec![0.0; n_dec * n_dec];
    let linear_cost = vec![0.0; n_dec]; // stays zeros
    for i in 0..horizon {
        // copy matrix q onto diagonal of cost matrix
        copy_mat(n_dec, &mut quadratic_cost, i * n, i * n, n, &q);
    }
    // same but with r instead of q, offset in the matrix after all the q stuff
    for i in 0..horizon-1 {
        let row_col = horizon * n + i * m;
        copy_mat(n_dec, &mut quadratic_cost, row_col, row_col, m, &r);
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

    // set bounds to infinity
    let cplex_inf = 1.0e+20;
    let lb = vec![-cplex_inf; n_dec];
    let ub = vec![cplex_inf; n_dec];

    // Finally, for each time step, we set up equality constraints on the states
    // (only as far as our forward horizon)
    // and solve that for the next state and action inputs
    // The first step of the horizon sets up the initial conditions
    // and then each step relates with the previous through euler integration

    let n_eq = n * horizon;
    let mut a_eq = vec![0.0; n_eq * n_dec];
    let mut b_eq = vec![0.0; n_eq];

    let mut a_mat = vec![0.0; n * n];
    let mut b_mat = vec![0.0; n];

    // for intial conditions, identity
    // also setup convenience neg_identity matrix
    let mut neg_identity = vec![0.0; n * n];
    for i in 0..n {
        a_eq[i * n_dec + i] = 1.0;
        neg_identity[i * n + i] = -1.0;
    }
    xs[0].copy_from_slice(x0);

    for j in 0..n_steps-1 {
        // for initial conditions
        b_eq[0..n].copy_from_slice(&xs[j]);

        // euler integration equality constraints
        for i in 1..horizon {
            let row = i * n;
            copy_mat(n_dec, &mut a_eq, row, row, n, &neg_identity);

            a_fun(i, &mut a_mat);
            for k in 0..n*n {
                a_mat[k] *= dt;
            }
            for k in 0..n {
                a_mat[k * n + k] += 1.0;
            }
            copy_mat(n_dec, &mut a_eq, row, row - n, n, &a_mat);

            b_fun(i, &mut b_mat);
            for k in 0..n {
                b_mat[k] *= dt;
            }
            copy_mat(n_dec, &mut a_eq, row, (i - 1) * m + n * horizon, 1, &b_mat);
        }

        // println!("Q: {:?}", &quadratic_cost);
        // println!("A: {:?}", &a_ineq);
        // println!("b: {:?}", &b_ineq);
        // println!("A_eq: {:?}", &a_eq);
        // println!("b_eq: {:?}", &b_eq);

        if let Some((_, x)) = solve_quadprog(&quadratic_cost, &linear_cost, &a_ineq, &b_ineq, &a_eq, &b_eq, &lb, &ub) {
            // println!("{}", obj_val);
            // println!("{:?}", x);
            xs[j + 1].copy_from_slice(&x[n..2*n]);
            us[j].copy_from_slice(&x[n*horizon..n*horizon+m]);
        } else {
            panic!("Quadratic program could not be solved");
        }
    }

    (xs, us)
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

fn mpc_test() {
    // everything here is now in __column-major__ order
    let a_fun = |_: usize, a: &mut [f64]| a.copy_from_slice(&[1., 1., 0., 1.]);
    let b_fun = |_: usize, b: &mut [f64]| b.copy_from_slice(&[0., 1.]);
    let x0 = [-1., -1.];
    let n_steps = 1000;
    let t_span = (0..n_steps).map(|i: usize| i as f64 * 5. / (n_steps - 1) as f64).collect::<Vec<_>>();
    let q = vec![1000., 0., 0., 1000.];
    let r = vec![0.1];

    let horizon = 100;
    let a_x_constraints = vec![1., 0., -1., 0., 0., 1., 0., -1.]; // vec![1., -1., 0., 0., 0., 0., 1., -1.];
    let b_x_constraints = vec![2., 2., 2., 2.];
    let a_u_constraints = vec![1., -1.];
    let b_u_constraints = vec![10., 10.];

    let start_time = precise_time_s();
    let (xs, _) = mpc_ltv(&a_fun, &b_fun, &q, &r, &t_span, horizon, &a_x_constraints, &b_x_constraints, &a_u_constraints, &b_u_constraints, &x0);
    println!("MPC took: {} seconds", precise_time_s() - start_time);

    if true {
        let mut x0s = vec![0.0; n_steps];
        let mut x1s = vec![0.0; n_steps];
        for i in 0..n_steps {
            x0s[i] = xs[i][0];
            x1s[i] = xs[i][1];
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

fn main() {
    if false {
        quadprog_test();
        matmult_test();
        axb_test();
    }
    if false {
        lqr_test();
    }
    mpc_test();
}
