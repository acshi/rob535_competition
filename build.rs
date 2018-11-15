extern crate cc;

// invoked by cargo to compile/manage native dependencies
fn main() {
    let cplex_dir = "/opt/ibm/ILOG/CPLEX_Studio128/cplex";

    // compile our cplex interface
    cc::Build::new()
        .file("src/cplexrust.c")
        .include(format!("{}/include/", cplex_dir))
        .compile("cplexrust");

    // provide location of cplex library
    println!("cargo:rustc-link-search=native={}/lib/x86-64_linux/static_pic", cplex_dir);
}
