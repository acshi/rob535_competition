extern crate rustplotlib;
extern crate time;

mod lib;
pub use crate::lib::solve_obstacle_problem;
use crate::lib::*;

fn main() {
    if true {
        run_problem1();
    }
    if false {
        run_refine_solution();
    }
}
