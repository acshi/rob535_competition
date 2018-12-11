extern crate rustplotlib;
extern crate time;
extern crate argmin;

mod lib;
pub use lib::solve_obstacle_problem;
use lib::*;

fn main() {
    if true {
        run_problem1();
    }
    if false {
        run_refine_solution();
    }
}
