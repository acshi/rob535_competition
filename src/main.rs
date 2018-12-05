extern crate libc;
extern crate lapack;
extern crate blas;
//extern crate openblas_src;
extern crate rustplotlib;
extern crate time;

mod lib;
pub use lib::solve_obstacle_problem;
use lib::*;

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
    if false {
        bike_test();
    }
    if false {
        bike_derivative_test();
    }
    if true {
        run_problem1();
    }
}
