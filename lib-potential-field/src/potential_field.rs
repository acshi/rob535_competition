extern crate libc;
extern crate nalgebra as na;
use na::{Vector2, Vector6, VectorN, DimName, Dim};
use libc::{c_double, c_int};
use std::process::{Command, Stdio};
use std::slice;
use std::mem;
use std::io::Write;
use std::f64;
use std::cmp::min;

type Pt = Vector2<f64>;
type Pts = Vec<Pt>;

fn electric_force_line_charge(x: Pt, p0: Pt, p1: Pt) -> Pt {
    let dir = p1 - p0;
    let L = dir.norm();
    let u_par = dir / L;
    let res = x - p0;
    let a = res.dot(&u_par);
    let perp = res - a*u_par;
    let h = perp.norm();
    let u_perp = perp/h;
    let b = L - a;
    let hyp_a_1 = (a*a + h*h).sqrt();
    let sin_a_1 = a/hyp_a_1;
    let cos_a_1 = h/hyp_a_1;

    let hyp_a_2 = (b*b + h*h).sqrt();
    let sin_a_2 = b/hyp_a_2;
    let cos_a_2 = h/hyp_a_2;

    (1f64/h)*(u_perp*(sin_a_1 + sin_a_2) + u_par*(-cos_a_1 + cos_a_2))
}

fn boundary_electric_force(x: Pt, boundary: &Pts) -> Pt {
    let mut force = Vector2::new(0f64, 0f64);
    for i in 0..boundary.len() - 1 {
        force += electric_force_line_charge(x, boundary[i], boundary[i+1]);
    }
    force /= (boundary.len() - 1) as f64;
    force
}

fn goal_force(x: Pt, cline: &Pts) -> Pt {
    let mut min_dist = f64::MAX;
    let mut idx = 0;
    for i in 0..cline.len() - 1 {
        let dist = (x - cline[i]).norm();
        if dist < min_dist {
            min_dist = dist;
            idx = i;
        }
    }
   let dir = cline[idx+1] - cline[idx];
   let dir = dir.normalize();
   //10.*dir
   dir/100.
}

fn electric_force(x: Pt, bl: &Pts, br: &Pts, cline: &Pts) -> Pt {
    let mut force = Vector2::new(0f64, 0f64);
    force += boundary_electric_force(x, bl);
    force += boundary_electric_force(x, br);
    // force += obstacles...
    force += goal_force(x, cline);
    force
}

fn plot_electric_force(bl: &Pts, br: &Pts, cline: &Pts) {
    let mut bl_x = Vec::new();
    let mut bl_y = Vec::new();
    let mut br_x = Vec::new();
    let mut br_y = Vec::new();
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut u = Vec::new();
    let mut v = Vec::new();
    for i in 0..min(100, bl.len()) {
        let num_points = 11;
        for j in 0..num_points {
            let a1 = (j + 1) as f64/(num_points + 1) as f64;
            let a2 = (num_points - j) as f64/(num_points + 1) as f64;
            let pos = a1*bl[i] + a2*br[i];
            let force = electric_force(pos, bl, br, cline);
            x.push(pos[0]);
            y.push(pos[1]);
            //x.push(bl[i][0]);
            //y.push(bl[i][1]);
            u.push(force[0]);
            v.push(force[1]);
            //u.push(1);
            //v.push(0);
        }
    }
    for i in 0..bl.len() {
        bl_x.push(bl[i][0]);
        bl_y.push(bl[i][1]);
        br_x.push(br[i][0]);
        br_y.push(br[i][1]);
    }
    Command::new("python").arg("-c").arg(format!(
            "import matplotlib.pyplot as plt; plt.plot({:?}, {:?}); plt.plot({:?}, {:?}); plt.quiver({:?}, {:?}, {:?}, {:?}, angles='xy'); plt.show()", bl_x, bl_y, br_x, br_y, x, y, u, v)).output().unwrap();
}

//#[test]
//fn test_e_field() {
//    let bl : Pts = vec![Vector2::new(0f64, 1f64), Vector2::new(1f64, 1f64), Vector2::new(2f64, 2f64), Vector2::new(3f64, 2f64)];
//    let br : Pts = vec![Vector2::new(0f64, 0f64), Vector2::new(1f64, 0f64), Vector2::new(2f64, 1f64), Vector2::new(3., 1.)];
//    let cline: Pts = vec![Vector2::new(0., 0.5), Vector2::new(1., 0.5), Vector2::new(2., 1.5), Vector2::new(3., 1.5)];
//    plot_electric_force(&bl, &br, &cline);
//}

fn plot_traj(traj: &Vec<Vector6<f64>>, bl: &Pts, br: &Pts) {
    let mut bl_x = Vec::new();
    let mut bl_y = Vec::new();
    let mut br_x = Vec::new();
    let mut br_y = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..2000 {
        xs.push(traj[i][0]);
        ys.push(traj[i][2]);
    }
    for i in 0..bl.len() {
        bl_x.push(bl[i][0]);
        bl_y.push(bl[i][1]);
        br_x.push(br[i][0]);
        br_y.push(br[i][1]);
    }
    // Command::new("python").arg("-c").arg(format!(
    //         "import matplotlib.pyplot as plt; plt.plot({:?}, {:?}); plt.plot({:?}, {:?}); plt.plot({:?}, {:?}); plt.show()", bl_x, bl_y, br_x, br_y, xs, ys).output().unwrap();
    let mut p = Command::new("python").stdin(Stdio::piped()).spawn().unwrap();
    {
        let p_in = p.stdin.as_mut().unwrap();
        write!(p_in, "import matplotlib.pyplot as plt; plt.plot({:?}, {:?}); plt.plot({:?}, {:?}); plt.plot({:?}, {:?}); plt.show()", bl_x, bl_y, br_x, br_y, xs, ys).unwrap();
    }
    p.wait().unwrap();
}

#[test]
fn test_traj() {
    let mut bl : Pts = Vec::new();
    let mut br : Pts = Vec::new();
    let mut cl : Pts = Vec::new();
    let mut theta : Vec<f64> = Vec::new();

    for i in 0..20 {
        bl.push(Vector2::new(1.*i as f64, -5.));
        br.push(Vector2::new(1.*i as f64, 5.));
        cl.push((bl[i] + br[i]) / 2.);
        theta.push(0.);
    }


    let x = Vector6::new(0., 0., 0., 0., 3.14/4., 0.);
    let (traj, us) = potential_fields_actual(&x, &bl, &br, &cl, &theta);
    println!("{} {} {} {} {}", us[0], us[1], us[2], us[3], us[4]);
    println!("{} {} {} {} {}", traj[0], traj[1], traj[2], traj[3], traj[4]);
    plot_electric_force(&bl, &br, &cl);
    plot_traj(&traj, &bl, &br);
}


fn potential_fields_actual(x0: &Vector6<f64>, bl: &Pts, br: &Pts, cline: &Pts, theta: &[f64]) -> (Vec<Vector6<f64>>, Pts) {
    //plot_electric_force(bl, br, cline);
    let mut x = x0.clone();
    let mut xs = Vec::new();
    let mut us = Vec::new();

    let k_psi = 10.;
    let k_u = 1000.;
    let u_min = 0.1;
    let u_target = 15.;
    let delta_min = -0.5;
    let delta_max = 0.5;
    let f_x_min = -5000.;
    let f_x_max = 2500.;
    //let dt = 0.01;
    let dt = 0.1;

    for i in 0..2000 {
        xs.push(x);
        let force = electric_force(Vector2::new(x[0], x[2]), bl, br, cline).normalize();
        let dir = Vector2::new(x[4].cos(), x[4].sin());
        let cos = dir.dot(&force);
        let sin = force[0]*dir[1] - force[1]*dir[0];
        let psi_err = sin.atan2(cos);
        //println!("{} {}", force, dir);
        let mut delta = -k_psi*psi_err;
        if delta < delta_min {
            delta = delta_min;
        } else if delta > delta_max {
            delta = delta_max;
        }

        let mut u_d = u_target*(0.5*(cos + 1.)).powi(5);
        if u_d < u_min {
            u_d = u_min;
        }

        let f_x = -k_u*(x[1] - u_d);

        us.push(Vector2::new(delta, f_x));
        //let ode_fun/*: FnMut(&Vector6<f64>) -> Vector6<f64>*/ = |x| bike_fun(x, delta, f_x);
        x = rk4_integrate(dt/100., 100, |x| bike_fun(x, delta, f_x), &x);
        //let y = Vector6::zeros();
        //let f = |z: Vector6<f64>| z.clone();
        //let v = test(|z: &Vector6<f64>| z.clone(), &y);
    }
    (xs, us)
}

fn rad2deg(r: f64) -> f64 {
    return r * 180.0 / f64::consts::PI;
}

fn rk4_integrate<F>(h: f64, n_steps: usize, f: F, x0: &Vector6<f64>) -> Vector6<f64>
        where F: Fn(&Vector6<f64>) -> Vector6<f64> {
    //let mut k1 = vec![0.0; n];
    //let mut k2 = vec![0.0; n];
    //let mut k3 = vec![0.0; n];
    //let mut k4 = vec![0.0; n];

    //let mut x_tmp = vec![0.0; n];
    //let mut x_old = vec![0.0; n];

    //x_old.copy_from_slice(x0);
    //x_out[0..n].copy_from_slice(x0);

    //let mut t = 0.0;
    let mut x_old: Vector6<f64> = x0.clone();
    for i in 0..n_steps {
        //{
            //f(t, &x_old, &mut k1);
            let k1 = f(&x_old);

            //for j in 0..n { x_tmp[j] = x_old[j] + 0.5 * h * k1[j]; }
            let x_tmp = x_old + 0.5*h*k1;

            //f(t + h * 0.5, &x_tmp, &mut k2);
            let k2 = f(&x_tmp);

            //for j in 0..n { x_tmp[j] = x_old[j] + 0.5 * h * k2[j]; }
            let x_tmp = x_old + 0.5*h*k2;

            //f(t + h * 0.5, &x_tmp, &mut k3);
            let k3 = f(&x_tmp);

            //for j in 0..n { x_tmp[j] = x_old[j] + h * k3[j]; }
            let x_tmp = x_old + h*k3;
            //f(t + h, &x_tmp, &mut k4);
            let k4 = f(&x_tmp);

            //for j in 0..n { x_tmp[j] = x_old[j] + h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]); }
            let x_tmp = x_old + h/6. * (k1 + 2.*k2 + 2.*k3 + k4);
        //}
        //x_old.copy_from_slice(&x_tmp);
        x_old = x_tmp;
        //if !just_x_result {
        //    x_out[i*n+n..i*n+n*2].copy_from_slice(&x_tmp);
        //}

        //t += h;
    }

    //if just_x_result {
    //    x_out.copy_from_slice(&x_old);
    //}
    x_old
}

fn bike_fun(x: &Vector6<f64>, delta_f: f64, mut f_x: f64) -> Vector6<f64> {
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
    Vector6::new(
        x[1]*x[4].cos()-x[3]*x[4].sin(),
        (-f*m*g+nw*f_x-f_yf*delta_f.sin())/m+x[3]*x[5],
        x[1]*x[4].sin()+x[3]*x[4].cos(),
        (f_yf*delta_f.cos()+f_yr)/m-x[1]*x[5],
        x[5],
        (f_yf*a*delta_f.cos()-f_yr*b)/iz)
}


fn convert(arr: *mut c_double,  len: usize) -> Pts {
    let mut res = vec![Vector2::new(0f64,0f64); len];
    let arr = unsafe {
        slice::from_raw_parts(arr, 2*len)
    };
    for i in 0..len {
            res[i][0] = arr[2*i];
            res[i][1] = arr[2*i + 1];
    }
    res
}

#[no_mangle]
pub extern "C" fn potential_fields(bl: *mut c_double,
                               br: *mut c_double,
                               cline: *mut c_double,
                               theta: *mut c_double,
                               len: c_int,
                               //XObs: *mut c_double,
                               //nObs: c_int,
                               u: *mut c_double,
                               u_len: *mut c_int) {
    let len = len as usize;
    let bl = convert(bl, len);
    let br= convert(br, len);
    let cline = convert(cline, len);
    let theta = unsafe {
        slice::from_raw_parts(theta, len)
    };
    ////XObs = ...

    let x = Vector6::new(287., 5., -176., 0., 2., 0.);
    let (traj, u_rust) = potential_fields_actual(&x, &bl, &br, &cline, &theta);
    //plot_electric_force(&bl, &br, &cline);
    plot_traj(&traj, &bl, &br);
    let u_rust: Vec<f64> = u_rust.iter().flat_map(|pt| pt.iter()).map(|r| *r).collect();

    unsafe {
        for i in 0..u_rust.len() {
            *u.offset(i as isize) = u_rust[i];
        }
        *u_len = u_rust.len() as i32;
    }
}