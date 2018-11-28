extern crate cc;

use std::fs;
use std::process::Command;

// invoked by cargo to compile/manage native dependencies
fn main() {
    let cplex_dir = if cfg!(windows) {
        "C:/Program Files/IBM/ILOG/CPLEX_Studio128/cplex"
    } else {
        "/opt/ibm/ILOG/CPLEX_Studio128/cplex"
    };

    let lib_dir = if cfg!(windows) {
        "/lib/x64_windows_vs2017/stat_mda/"
    } else {
        "/lib/x86-64_linux/static_pic"
    };

    // compile our cplex interface
    cc::Build::new()
        .file("src/cplexrust.c")
        .include(format!("{}/include/", cplex_dir))
        .compile("cplexrust");

    // provide location of cplex library
    println!("cargo:rustc-link-search=native={}{}", cplex_dir, lib_dir);

    if cfg!(windows) {
        println!("cargo:rustc-link-lib=static=cplex1280");
    } else {
        println!("cargo:rustc-link-lib=static=cplex");
    }

    if cfg!(windows) {
        // help find the openblas library on windows
        fs::create_dir_all(".\\target\\debug\\deps").unwrap();
        fs::create_dir_all(".\\target\\release\\deps").unwrap();
        // let output = Command::new("cmd").arg("dir").output().unwrap();
        // panic!("{:?}", str::from_utf8(&output.stdout).unwrap())
        Command::new("cmd").arg("/C").arg("mklink")
                           .arg(".\\target\\debug\\deps\\libopenblas.a")
                           .arg("C:\\OpenBLAS-v0.2.19-Win64-int32\\lib\\libopenblas.a")
                           .output().unwrap();
        Command::new("cmd").arg("/C").arg("mklink")
                           .arg(".\\target\\release\\deps\\libopenblas.a")
                           .arg("C:\\OpenBLAS-v0.2.19-Win64-int32\\lib\\libopenblas.a")
                           .output().unwrap();
    }
}
