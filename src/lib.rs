extern crate libc;
use libc::c_void;
use std::ptr;
use std::fs::File;
use std::io::Read;
use std::f64;
use std::slice;

extern crate lapack;
extern crate blas;

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

trait FastMath {
    fn fast_atan(self) -> f64;
    fn fast_atan2_inner(self, f64) -> f64;
    fn fast_atan2(self, f64) -> f64;
}

impl FastMath for f64 {
    fn fast_atan(self) -> f64 {
        let z = self;
        // if z >= -1.0 && z <= 1.0 {
        //     let abs_val = z.abs();
        //     let res = z * f64::consts::FRAC_PI_4 - z * (abs_val - 1.0) * (0.2447 + 0.0663 * abs_val);
        //     return res;
        // }
        // print!("{}, ", self);
        z.atan()
    }

    fn fast_atan2(self, x: f64) -> f64 {
        let val1 = self.fast_atan2_inner(x);
        // let val2 = self.atan2(x);
        // let err = val2 - val1;
        // print!("{}, ", err);
        // if err.abs() > 0.01 {
        //     println!("BAD!");
        // }

        val1
    }

    fn fast_atan2_inner(self, x: f64) -> f64 {
        let y = self;
        if x != 0.0 {
            if x.abs() > y.abs() {
                let z = y / x;
                if x > 0.0 {
                    z.fast_atan()
                } else if y >= 0.0 {
                    z.fast_atan() + f64::consts::PI
                } else {
                    z.fast_atan() - f64::consts::PI
                }
            } else {
                let z = x / y;
                if y > 0.0 {
                    -z.fast_atan() + f64::consts::FRAC_PI_2
                } else {
                    -z.fast_atan() - f64::consts::FRAC_PI_2
                }
            }
        } else {
            if y > 0.0 {
                f64::consts::FRAC_PI_2
            } else if y < 0.0 {
                -f64::consts::FRAC_PI_2
            } else {
                0.0
            }
        }
    }
}

fn bike_fun(x: &[f64], delta_f: f64, mut f_x: f64, dx: &mut [f64]) {
    let nw = 2.0;
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
    let a_f = rad2deg(delta_f - (x[3]+a*x[5]).fast_atan2(x[1]));
    let a_r = rad2deg(-(x[3]-b*x[5]).fast_atan2(x[1]));

    // Nonlinear Tire dynamics
    let phi_yf = (1.0-ey)*(a_f+shy)+(ey/by)*(by*(a_f+shy)).fast_atan();
    let phi_yr = (1.0-ey)*(a_r+shy)+(ey/by)*(by*(a_r+shy)).fast_atan();

    let f_zf=b/(a+b)*m*g;
    let f_yf=f_zf*dy*(cy*(by*phi_yf).fast_atan()).sin()+svy;

    let f_zr=a/(a+b)*m*g;
    let mut f_yr=f_zr*dy*(cy*(by*phi_yr).fast_atan()).sin()+svy;

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

    let n_used_controls = shooting_solve(tp, controls);
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
    while (idx > 0 || delta == 1) && (idx < r_xs.len() - 1 || delta == -1) {
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

fn plot_trajectory(tp: &TrackProblem, n_steps: usize, xs: &[f64]) {
    let track_n = tp.thetas.len();
    let ax = Axes2D::new()
        .add(Line2D::new("")
            .data(&xs[0..n_steps], &xs[n_steps*2..n_steps*3])
            .color("red")
            .linestyle("-")
            .linewidth(1.0))
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
    let mut mpl = Matplotlib::new().unwrap();
    ax.apply(&mut mpl).unwrap();
    mpl.show().unwrap();
    mpl.wait().unwrap();
}

#[allow(dead_code)]
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

fn shooting_obj(tp: &TrackProblem, old_track_i: usize, dt: f64, x0: &[f64], deltas: &[f64], fxs: &[f64]) -> f64 {
    let n = x0.len();
    let track_n = tp.thetas.len();
    let ref_x = &tp.cline[0..track_n];
    let ref_y = &tp.cline[track_n..track_n*2];
    let ref_t = &tp.thetas;
    let mut track_i = next_track_idx(old_track_i, ref_x, ref_y, x0[0], x0[2]);

    // score following the center line
    let mut obj_val = 0.0;

    let mut old_xs = x0.to_vec();
    let mut new_xs = vec![0.0; n];
    let mut discount = 1.0;
    for k in 0..deltas.len() {
        let mut integrate_fun = |_t: f64, x: &[f64], x_new: &mut[f64]| {
            bike_fun(x, deltas[k], fxs[k], x_new);
        };
        rk4_integrate(dt / 4.0, 4, &mut integrate_fun, &old_xs, &mut new_xs);
        old_xs.copy_from_slice(&new_xs);

        track_i = next_track_idx(track_i, ref_x, ref_y, new_xs[0], new_xs[2]);
        let completion = track_i as f64 / (track_n - 1) as f64 * 100.0;
        if completion < 95.0 {
            let dist_err = (ref_x[track_i] - new_xs[0]).powi(2) +
                           (ref_y[track_i] - new_xs[2]).powi(2);
            let obs_err = (dist_err - 9.0f64.powi(2)).max(0.0).powi(4);
            let new_err = dist_err + obs_err + ((ref_t[track_i] - new_xs[4]) * 2.0).powi(2);

            obj_val += new_err * discount;
        }
        obj_val += (1.0 - (new_xs[1] / 30.0).min(1.0)).powi(2) * 10.0 * discount;
        obj_val -= completion * 10.0 * discount;
        // obj_val += new_xs[1].min(0.0).abs() * 10.0;
        // obj_val += deltas[k].powi(2) * 100.0;
        // obj_val += fxs[k].min(0.0).abs() // penalize breaking
        discount *= 0.98;
    }

    obj_val
}

fn shooting_best_delta(tp: &TrackProblem, track_i: usize, dt: f64, k: usize, x0: &[f64], deltas: &mut [f64], fxs: &[f64]) -> f64 {
    let mut min_obj = f64::MAX;
    let mut min_delta = 0.0;

    // scan through delta values for the lowest cost
    for &delta in [-0.5, -0.4, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5].iter() {
        deltas[k] = delta;
        // println!("delta: {}", delta);
        let obj = shooting_obj(tp, track_i, dt, x0, deltas, fxs);
        if obj < min_obj {
            min_obj = obj;
            min_delta = delta;
        }
    }

    min_delta
}

fn shooting_best_fx(tp: &TrackProblem, track_i: usize, dt: f64, k: usize, x0: &[f64], deltas: &[f64], fxs: &mut [f64]) -> f64 {
    let mut min_obj = f64::MAX;
    let mut min_fx = 0.0;

    // scan through values for the lowest cost
    for &fx in [-5000.0, -2500.0, -500.0, -250.0, 0.0, 250.0, 500.0, 1000.0, 2500.0].iter() {
        fxs[k] = fx;
        let obj = shooting_obj(tp, track_i, dt, x0, deltas, fxs);
        if obj < min_obj {
            min_obj = obj;
            min_fx = fx;
        }
    }

    min_fx
}

fn shooting_step(tp: &TrackProblem, old_track_i: usize, horizon: usize, dt: f64, x0: &[f64],
                 ref_deltas: &[f64], ref_fxs: &[f64], x_new: &mut [f64], u_new: &mut [f64]) {
    let n = x0.len();

    let track_n = tp.thetas.len();
    let ref_x = &tp.cline[0..track_n];
    let ref_y = &tp.cline[track_n..track_n*2];

    // let u_lb = vec![-0.5, -5000.0];
    // let u_ub = vec![0.5, 2500.0];

    let n_steps = horizon + 1;

    let mut track_i = old_track_i;

    let mut deltas = vec![0.0; horizon];
    deltas.copy_from_slice(ref_deltas);
    let mut fxs = vec![2500.0; horizon];
    fxs.copy_from_slice(ref_fxs);

    let mut xs = vec![0.0; n * n_steps];
    let mut old_xs = x0.to_vec();
    let mut new_xs = vec![0.0; n];
    for k in 0..=1 {
        for i in 0..n {
            xs[i * n_steps + k] = old_xs[i];
        }
        track_i = next_track_idx(track_i, ref_x, ref_y, old_xs[0], old_xs[2]);

        deltas[k] = shooting_best_delta(tp, track_i, dt, k, &old_xs, &mut deltas, &fxs);
        fxs[k] = shooting_best_fx(tp, track_i, dt, k, &old_xs, &deltas, &mut fxs);

        // pass through "real world" model to get our next state
        let mut integrate_fun = |_t: f64, x: &[f64], x_new: &mut[f64]| {
            bike_fun(x, deltas[k], fxs[k], x_new);
        };
        rk4_integrate(dt / 8.0, 8, &mut integrate_fun, &old_xs, &mut new_xs);
        old_xs.copy_from_slice(&new_xs);
    }
    for i in 0..n {
        xs[i * n_steps + horizon] = old_xs[i];
    }

    u_new[0] = deltas[0];
    u_new[1] = fxs[0];
    for i in 0..n {
        x_new[i] = xs[i * n_steps + 1]; // copy time1 values, not time0/x0!
    }

    if false {
        plot_trajectory(tp, n_steps, &xs);
    }
}

fn shooting_solve(tp: TrackProblem, controls: &mut [f64]) -> usize {
    // general idea: essentially a sort of non-linear mpc
    // performed with a sort of tuned coordinate descent.
    let n = 6;
    let m = 2;
    let horizon = 25*2;
    let dt = 0.05;
    let ref_steps = tp.thetas.len();
    let total_steps = ref_steps * 20;
    let tp = resample_track(&tp, total_steps);
    let track_n = tp.thetas.len();
    let ref_x = &tp.cline[0..track_n];
    let ref_y = &tp.cline[track_n..track_n*2];

    let mut xs = vec![0.0; total_steps * n];

    assert!(controls.len() >= total_steps * m);

    let start_time = precise_time_s();
    for solve_i in 0..=3 {
        let mut steps_used = total_steps - 1;
        let mut track_i = 0;
        let mut old_xs = [287.0, 5.0, -176.0, 0.0, 2.0, 0.0];
        let mut new_xs = vec![0.0; n];
        let mut new_us = vec![0.0; m];
        for k in 0..total_steps-1 {
            for i in 0..n {
                xs[i * total_steps + k] = old_xs[i];
            }
            let mut deltas = vec![0.0; horizon];
            let mut fxs = vec![2500.0; horizon];
            if solve_i > 0 {
                deltas.copy_from_slice(&controls[k..k+horizon]);
                fxs.copy_from_slice(&controls[k+total_steps-1..k+total_steps-1+horizon]);
            }
            shooting_step(&tp, track_i, horizon, dt, &old_xs, &deltas, &fxs, &mut new_xs, &mut new_us);
            controls[k] = new_us[0];
            controls[k + total_steps - 1] = new_us[1];

            if k % 10 == 0 {
                let completion = track_i as f64 / (track_n - 1) as f64 * 100.0;
                print!("K: {} C: {:.2}% X: ", k, completion);
                for i in 0..n {
                    print!("{:.2}, ", old_xs[i]);
                }
                println!("{:.2}, {:.2}", controls[k], controls[k + total_steps - 1]);
            }

            old_xs.copy_from_slice(&new_xs);

            track_i = next_track_idx(track_i, ref_x, ref_y, old_xs[0], old_xs[2]);
            let completion = track_i as f64 / (track_n - 1) as f64 * 100.0;
            if completion > 99.8 {
                steps_used = k;
                break;
            }
        }
        for i in 0..n {
            xs[i * total_steps + steps_used] = old_xs[i];
        }
        println!("Completed {:.2}%", track_i as f64 / (track_n - 1) as f64 * 100.0);
        println!("Trajectory time: {}", steps_used as f64 * dt);

        // fill in after with same values
        for k in steps_used+1..total_steps {
            for i in 0..n {
                xs[i * total_steps + k] = old_xs[i];
            }
            controls[k] = new_us[0];
            controls[k + total_steps - 1] = new_us[1];
        }
    }
    println!("Shooting took: {} seconds", precise_time_s() - start_time);

    // print!("[");
    // for i in 0..total_steps {
    //     for j in 0..n {
    //         print!("{}, ", xs[j * total_steps + i]);
    //     }
    //     for j in 0..m {
    //         print!("{}, ", controls[j * (total_steps - 1) + i]);
    //     }
    //     print!("; ");
    // }
    // println!("];");

    plot_trajectory(&tp, total_steps, &xs);

    return total_steps - 1;
}

pub fn run_problem1() {
    let (bl, br, cline, thetas) = load_test_track();
    let tp = TrackProblem { bl, br, cline, thetas,
                            n_obs: 0, obs_x: vec![], obs_y: vec![] };
    let mut controls = vec![0.0; 2048 * 8];
    shooting_solve(tp, &mut controls);
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
