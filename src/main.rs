extern crate libc;
extern crate rustplotlib;
extern crate time;
extern crate potential_field;

mod lib;
use lib::*;

fn main() {
    if false {
        bike_test();
    }
    for i in 0..10 {
        potential_field::test_traj();
    }
}
