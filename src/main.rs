extern crate libc;
extern crate lapack;
extern crate blas;
//extern crate openblas_src;
extern crate rustplotlib;
extern crate time;

mod lib;
pub use lib::trajectory_for_obstacles;
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
    if true {
        mpc_test();
    }
    if false {
        bike_test();
    }
}
