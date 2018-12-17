use std::fs::File;
use std::io::Read;
use std::f64;
use std::slice;
use std::io::Write;

extern crate rustplotlib;
use rustplotlib::{Axes2D, Line2D, Backend};
use rustplotlib::backend::Matplotlib;

extern crate time;
use time::precise_time_s;

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

#[allow(dead_code)]
fn forward_integrate_bike(rk4_div: usize, controls: &[f64]) -> Vec<f64> {
    let n = 6;
    let m = 2;
    let n_steps = controls.len() / m;
    let x0 = [287.0, 5.0, -176.0, 0.0, 2.0, 0.0];
    let dt = 0.05;
    let mut x_all = vec![0.0; n * n_steps];

    let mut old_xs = x0.to_vec();
    let mut new_xs = vec![0.0; n];

    for k in 0..n_steps-1 {
        for i in 0..n {
            x_all[i * n_steps + k] = old_xs[i];
        }

        // if k % 50 == 0 {
        //     print!("K: {} ", k);
        //     for i in 0..n {
        //         print!("{:.2}, ", old_xs[i]);
        //     }
        //     println!("{:.2}, {:.2}", controls[k], controls[k + n_steps]);
        // }

        let mut ode_fun = |_t: f64, x: &[f64], dx: &mut [f64]| {
            bike_fun(x, controls[k], controls[k + n_steps], dx);
        };
        rk4_integrate(dt / rk4_div as f64, rk4_div, &mut ode_fun, &old_xs, &mut new_xs);
        old_xs.copy_from_slice(&new_xs);
    }
    for i in 0..n {
        x_all[i * n_steps + n_steps-1] = old_xs[i];
    }

    x_all
}

fn rad2deg(r: f64) -> f64 {
    return r * 180.0 / f64::consts::PI;
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
    while buf_i < buf.len() && !is_float_digit(buf[buf_i] as char) {
        buf_i += 1;
    }

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

#[allow(dead_code)]
fn trajectory_stays_on_track(x_all: &[f64]) -> bool {
    let n = 6;
    let n_steps = x_all.len() / n;
    let (bl, br, cline, thetas) = load_test_track();
    let track_n = thetas.len();
    let mut trajectory_i = 0;
    let mut track_i = 0;

    let xs = &x_all[0..n_steps];
    let ys = &x_all[n_steps*2..n_steps*3];

    let mut status = true;

    while trajectory_i < n_steps {
        let theta = thetas[track_i];
        let (sint, cost) = theta.sin_cos();
        // project center of mass onto track axis
        let car_forward = sint * ys[trajectory_i] + cost * xs[trajectory_i];
        let car_sideways = sint * xs[trajectory_i] + cost * ys[trajectory_i];

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
            .data(&xs, &ys)
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
    obs_p0x: Vec<f64>,
    obs_p0y: Vec<f64>,
    obs_len: Vec<f64>,
    obs_dirx: Vec<f64>,
    obs_diry: Vec<f64>,
}

fn boundaries_to_obs(n_track: usize, bl: &[f64], br: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_obs = (n_track - 1) * 2;
    let mut obs_p0x = vec![0.0; n_obs];
    let mut obs_p0y = vec![0.0; n_obs];
    let mut obs_len = vec![0.0; n_obs];
    let mut obs_dirx = vec![0.0; n_obs];
    let mut obs_diry = vec![0.0; n_obs];
    for i in 0..n_track-1 {
        let (x0, x1) = (br[i], br[i+1]);
        let (y0, y1) = (br[i+n_track], br[i+1+n_track]);
        let (dx, dy) = (x0 - x1, y0 - y1);
        obs_p0x[i] = x1;
        obs_p0y[i] = y1;
        obs_len[i] = (dx*dx + dy*dy).sqrt();
        obs_dirx[i] = dx / obs_len[i];
        obs_diry[i] = dy / obs_len[i];
    }
    for i in 0..n_track-1 {
        let (x0, x1) = (bl[i], bl[i+1]);
        let (y0, y1) = (bl[i+n_track], bl[i+1+n_track]);
        let (dx, dy) = (x1 - x0, y1 - y0);
        let idx = i + n_track-1;
        obs_p0x[idx] = x0;
        obs_p0y[idx] = y0;
        obs_len[idx] = (dx*dx + dy*dy).sqrt();
        obs_dirx[idx] = dx / obs_len[idx];
        obs_diry[idx] = dy / obs_len[idx];
    }
    (obs_p0x, obs_p0y, obs_len, obs_dirx, obs_diry)
}

#[no_mangle]
pub extern fn solve_obstacle_problem(n_track: i32, bl: *const f64, br: *const f64,
                                     cline: *const f64, thetas: *const f64,
                                     _n_box_obs: i32, _obs_x: *const f64, _obs_y: *const f64,
                                     n_controls: *mut i32, controls: *mut f64) {
    let n_track = n_track as usize;
    // let n_box_obs = n_box_obs as usize;
    let n_max_controls = unsafe { *n_controls as usize };

    let bl = unsafe { slice::from_raw_parts(bl, n_track * 2).to_vec() };
    let br = unsafe { slice::from_raw_parts(br, n_track * 2).to_vec() };
    let cline = unsafe { slice::from_raw_parts(cline, n_track * 2).to_vec() };
    let thetas = unsafe { slice::from_raw_parts(thetas, n_track).to_vec() };
    // let obs_x = unsafe { slice::from_raw_parts(obs_x, n_obs * 4).to_vec() };
    // let obs_y = unsafe { slice::from_raw_parts(obs_y, n_obs * 4).to_vec() };
    let controls = unsafe { slice::from_raw_parts_mut(controls, n_max_controls * 2) };

    // put single points from each line segment into p0
    // we skip the last point in bl and the first in br (because it goes the opposite direction)
    // and calculate other quantities accordingly

    let (obs_p0x, obs_p0y, obs_len, obs_dirx, obs_diry) = boundaries_to_obs(n_track, &bl, &br);
    let tp = TrackProblem { bl, br, cline, thetas, obs_p0x, obs_p0y, obs_len, obs_dirx, obs_diry };

    let n_used_controls = shooting_solve(tp, controls);
    unsafe {
        *n_controls = n_used_controls as i32;
    }
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
        // .add(Line2D::new("")
        //     .data(&tp.cline[0..track_n], &tp.cline[track_n..track_n*2])
        //     .color("orange")
        //     .linestyle("-")
        //     .linewidth(1.0))
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

fn report_trajectory_error(n_steps: usize, valid_steps: usize, xs: &[f64], r_xs: &[f64], r_ys: &[f64]) {
    // caclulate max distance error between the actual and nominal trajectories,
    // when the x value of the actual trajectory is between or equal to 3 and 4 m
    let mut max_dist_error = 0.0;
    let mut max_dist_i = 0;
    let mut max_dist_idx = 0;
    let mut mean_square_error = 0.0;
    let mut track_idx = 0;
    for i in 0..valid_steps {
        track_idx = next_track_idx(track_idx, r_xs, r_ys, xs[i], xs[n_steps*2 + i]);
        let dx = xs[i] - r_xs[track_idx];
        let dy = xs[n_steps*2 + i] - r_ys[track_idx];
        let sq_err = dx * dx + dy * dy;
        mean_square_error += sq_err;
        let dist_err = sq_err.sqrt();
        if dist_err > max_dist_error {
            max_dist_error = dist_err;
            max_dist_i = i;
            max_dist_idx = track_idx;
        }
    }
    mean_square_error /= xs.len() as f64;
    println!("Max distance error: {}", max_dist_error);
    println!(" at {}: ({}, {}) with ref ({}, {})", max_dist_i, xs[max_dist_i], xs[n_steps*2 + max_dist_i], r_xs[max_dist_idx], r_ys[max_dist_idx]);
    println!("Mean square error: {}", mean_square_error);
}

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
                  obs_p0x: tp.obs_p0x.to_vec(), obs_p0y: tp.obs_p0y.to_vec(),
                  obs_len: tp.obs_len.to_vec(),
                  obs_dirx: tp.obs_dirx.to_vec(), obs_diry: tp.obs_diry.to_vec() }
}

fn rad_error(a: f64, b: f64) -> f64 {
    let pi = f64::consts::PI;
    (a - b + pi) % (2.0 * pi) - pi
}

// negative indicates intersection
fn closest_obj_signed_sq_dist(tp: &TrackProblem, x: f64, y: f64) -> f64 {
    let mut min_dist_sq = f64::MAX;
    let mut min_i = 0;
    for i in 0..tp.obs_p0x.len() {
        let (p0x, p0y) = (tp.obs_p0x[i], tp.obs_p0y[i]);
        if (p0x - x).abs() > 10.0 || (p0y - y).abs() > 10.0 {
            continue;
        }
        let (dirx, diry) = (tp.obs_dirx[i], tp.obs_diry[i]);
        // s is the distance from p0, along the line direction,
        // to the point that is closest to x, y
        let s = (x - p0x) * dirx + (y - p0y) * diry;
        let s = s.max(0.0).min(tp.obs_len[i]);
        // d_sq is the square distance between x, y and point represented by s
        let d_sq = (s * dirx + p0x - x).powi(2) + (s * diry + p0y - y).powi(2);
        if d_sq < min_dist_sq {
            min_dist_sq = d_sq;
            min_i = i;
        }
    }
    if min_dist_sq >= 1e10 {
        return 1e10;
    }
    let (p0x, p0y) = (tp.obs_p0x[min_i], tp.obs_p0y[min_i]);
    let (dirx, diry) = (tp.obs_dirx[min_i], tp.obs_diry[min_i]);
    let cross = (x - p0x) * diry - (y - p0y) * dirx;
    min_dist_sq * cross.signum()
}

fn shooting_obj(tp: &TrackProblem, solve_i: usize, old_track_i: usize, horizon: usize, dt: f64, rk4_div: usize, x0: &[f64], deltas: &[f64], fxs: &[f64]) -> f64 {
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
    let alpha = 0.985f64.powf(50.0 / horizon as f64);
    for k in 0..deltas.len() {
        let mut integrate_fun = |_t: f64, x: &[f64], dx: &mut[f64]| {
            bike_fun(x, deltas[k], fxs[k], dx);
        };
        rk4_integrate(dt / rk4_div as f64, rk4_div, &mut integrate_fun, &old_xs, &mut new_xs);
        old_xs.copy_from_slice(&new_xs);

        track_i = next_track_idx(track_i, ref_x, ref_y, new_xs[0], new_xs[2]);
        let completion = track_i as f64 / (track_n - 1) as f64 * 100.0;
        if completion < 95.0 {
            let dist_err = (ref_x[track_i] - new_xs[0]).powi(2) +
                           (ref_y[track_i] - new_xs[2]).powi(2);
            let boundary_err = (dist_err - 8.0f64.powi(2)).max(0.0).powi(4);
            let theta_err = (rad_error(ref_t[track_i], new_xs[4]) * 2.0).powi(2);
            let side_err = new_xs[3].powi(4);
            let signed_sq_dist = closest_obj_signed_sq_dist(tp, new_xs[0], new_xs[2]);
            let avoid_dist = 6.0;
            let collision_err = if signed_sq_dist < avoid_dist * avoid_dist {
                let dist = signed_sq_dist.max(0.01).sqrt();
                let repulse = 50.0 * (1.0 / dist - 1.0 / avoid_dist) / (dist * dist);
                // println!("SSD: {} dist_err: {} repulse: {}", signed_sq_dist, dist_err, repulse);
                repulse
            } else {
                0.0
            };
            let collision_err = collision_err * (solve_i as f64 - 3.0).min(3.0).max(0.0) / 10.0;
            // let side_err = side_err * (solve_i as f64 - 3.0).min(3.0).max(1.0) / 10.0;
            //
            // let theta_diff = rad_error(ref_t[track_i], new_xs[4]);
            // let theta_err = (theta_diff * 2.0).powi(4);
            //
            // let new_err = dist_err * 0.1 + collision_err + theta_err * 0.0;
            let dist_err = dist_err * (8.0 - solve_i as f64).max(4.0).min(5.0) / 5.0;
            let boundary_err = boundary_err * (8.0 - solve_i as f64).max(4.0).min(5.0) / 5.0;
            let new_err = if solve_i >= 5 {
                (dist_err + boundary_err) * 0.65 + side_err * 0.05
            } else {
                dist_err + collision_err + theta_err + side_err * 0.05
            };

            obj_val += new_err * discount;
        }
        // obj_val += ((1.0 - (new_xs[1] / 30.0).min(1.0)) * 10.0).powi(3) * discount;
        obj_val += (1.0 - (new_xs[1] / 30.0).min(1.0)).powi(2) * 10.0 * discount;
        obj_val -= completion * 10.0 * discount;
        // obj_val += new_xs[1].min(0.0).abs() * 10.0;
        // obj_val += deltas[k].powi(2) * 100.0;
        // obj_val += fxs[k].min(0.0).abs() // penalize breaking
        discount *= alpha;
    }

    obj_val
}

fn shooting_best_delta(tp: &TrackProblem, solve_i: usize, track_i: usize, horizon: usize, high_res: bool, dt: f64, rk4_div: usize, k: usize, x0: &[f64], deltas: &mut [f64], fxs: &[f64]) -> f64 {
    let mut min_obj = f64::MAX;
    let mut min_delta = 0.0;

    // scan through delta values for the lowest cost
    let iter_vals = if high_res {
        (-20..=20).map(|i| i as f64 / 20.0 * 0.5).collect::<Vec<_>>()
    } else {
        [-0.5, -0.4, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5].to_vec()
    };
    for &delta in iter_vals.iter() {
    // for &delta in [-0.5, -0.25, 0.0, 0.25, 0.5].iter() {
        deltas[k] = delta;
        // println!("delta: {}", delta);
        let obj = shooting_obj(tp, solve_i, track_i, horizon, dt, rk4_div, x0, deltas, fxs);
        if obj < min_obj {
            min_obj = obj;
            min_delta = delta;
        }
    }

    min_delta
}

fn shooting_best_fx(tp: &TrackProblem, solve_i: usize, track_i: usize, horizon: usize, high_res: bool, dt: f64, rk4_div: usize, k: usize, x0: &[f64], deltas: &[f64], fxs: &mut [f64]) -> f64 {
    let mut min_obj = f64::MAX;
    let mut min_fx = 0.0;

    // scan through values for the lowest cost
    let iter_vals = if high_res {
        (-20..=10).map(|i| i as f64 / 20.0 * 5000.0).collect::<Vec<_>>()
    } else {
        [-5000.0, -2500.0, -500.0, -250.0, 0.0, 250.0, 500.0, 1000.0, 2500.0].to_vec()
    };
    for &fx in iter_vals.iter() {
    // for &fx in [-5000.0, -2500.0, 0.0, 1250.0, 2500.0].iter() {
        fxs[k] = fx;
        let obj = shooting_obj(tp, solve_i, track_i, horizon, dt, rk4_div, x0, deltas, fxs);
        if obj < min_obj {
            min_obj = obj;
            min_fx = fx;
        }
    }

    min_fx
}

fn shooting_step(tp: &TrackProblem, solve_i: usize, old_track_i: usize, horizon: usize, high_res: bool, dt: f64, rk4_div: usize, x0: &[f64],
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
    let last_k = 0;
    for k in 0..=last_k {
        for i in 0..n {
            xs[i * n_steps + k] = old_xs[i];
        }
        track_i = next_track_idx(track_i, ref_x, ref_y, old_xs[0], old_xs[2]);

        deltas[k] = shooting_best_delta(tp, solve_i, track_i, horizon, high_res, dt, rk4_div, k, &old_xs, &mut deltas, &fxs);
        fxs[k] = shooting_best_fx(tp, solve_i, track_i, horizon, high_res, dt, rk4_div, k, &old_xs, &deltas, &mut fxs);

        // pass through "real world" model to get our next state
        let mut integrate_fun = |_t: f64, x: &[f64], dx: &mut[f64]| {
            bike_fun(x, deltas[k], fxs[k], dx);
        };
        rk4_integrate(dt / rk4_div as f64, rk4_div, &mut integrate_fun, &old_xs, &mut new_xs);
        old_xs.copy_from_slice(&new_xs);
    }
    for i in 0..n {
        xs[i * n_steps + last_k + 1] = old_xs[i];
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
    let dt = 0.05;
    let rk4_div = 4;
    let ref_steps = tp.thetas.len();
    let total_steps = ref_steps * 20;
    let tp = resample_track(&tp, total_steps);
    let track_n = tp.thetas.len();
    let ref_x = &tp.cline[0..track_n];
    let ref_y = &tp.cline[track_n..track_n*2];

    let refinement_window = 50;

    let mut xs = vec![0.0; total_steps * n];

    let mut best_xs = vec![0.0; total_steps * n];
    let mut best_controls = vec![0.0; total_steps * m];
    let mut best_time = f64::MAX;
    let mut best_steps_used = total_steps;

    assert!(controls.len() >= total_steps * m);

    let start_time = precise_time_s();
    for solve_i in 0..=5 {
        let horizon = 51; //if solve_i <= 3 { 51 } else { 101 };
        let high_res = solve_i >= 5;

        let mut steps_used = total_steps - 1;
        let mut track_i = 0;
        let mut old_xs = [287.0, 5.0, -176.0, 0.0, 2.0, 0.0];
        let mut new_xs = vec![0.0; n];
        let mut new_us = vec![0.0; m];
        let mut last_completion = 0.0;
        for k in 0..total_steps - 1 {
            for i in 0..n {
                xs[i * total_steps + k] = old_xs[i];
            }

            if solve_i == 3 && k >= refinement_window && k % refinement_window / 2 == 0 {
                // local_refinement(&tp, track_i, dt, k - refinement_window, refinement_window, &mut xs, controls);
            }

            let mut deltas = vec![0.0; horizon];
            let mut fxs = vec![2500.0; horizon];
            if solve_i > 0 {
                deltas.copy_from_slice(&controls[k..k+horizon]);
                fxs.copy_from_slice(&controls[k+total_steps..k+total_steps+horizon]);
            }
            shooting_step(&tp, solve_i, track_i, horizon, high_res, dt, rk4_div, &old_xs, &deltas, &fxs, &mut new_xs, &mut new_us);
            controls[k] = new_us[0];
            controls[k + total_steps] = new_us[1];

            if k % 50 == 0 {
                let completion = track_i as f64 / (track_n - 1) as f64 * 100.0;
                print!("K: {} C: {:.2}% X: ", k, completion);
                for i in 0..n {
                    print!("{:.2}, ", old_xs[i]);
                }
                println!("{:.2}, {:.2}", controls[k], controls[k + total_steps]);

                if completion < last_completion {
                    println!("Not making progress, aborting run.");
                    break;
                }
                last_completion = completion;
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
        let solution_time = steps_used as f64 * dt;
        println!("Trajectory time: {}", solution_time);
        report_trajectory_error(total_steps, steps_used + 1, &xs, ref_x, ref_y);

        // fill in after with same values
        for k in steps_used+1..total_steps {
            for i in 0..n {
                xs[i * total_steps + k] = old_xs[i];
            }
            controls[k] = new_us[0];
            controls[k + total_steps] = new_us[1];
        }

        if solution_time < best_time {
            best_time = solution_time;
            best_steps_used = steps_used;
            best_xs.copy_from_slice(&xs);
            best_controls.copy_from_slice(&controls[0..total_steps * m]);
        }
    }
    println!("Shooting took: {:.2} seconds", precise_time_s() - start_time);
    println!("Best solution time: {:.2} seconds", best_time);

    let mut f = File::create("best_xs.txt").unwrap();
    write!(f, "[").unwrap();
    for i in 0..best_steps_used+10 {
        for j in 0..n {
            write!(f, "{:.5}, ", best_xs[j * total_steps + i]).unwrap();
        }
        write!(f, "; ").unwrap();
    }
    writeln!(f, "];").unwrap();

    let mut f = File::create("best_us.txt").unwrap();
    write!(f, "[").unwrap();
    for i in 0..best_steps_used+10 {
        for j in 0..m {
            write!(f, "{:.5}, ", best_controls[j * total_steps + i]).unwrap();
        }
        write!(f, "; ").unwrap();
    }
    writeln!(f, "];").unwrap();

    plot_trajectory(&tp, total_steps, &best_xs);

    if false {
        let x_all = forward_integrate_bike(rk4_div, &best_controls);
        // plot_trajectory(&tp, total_steps, &x_all);
        trajectory_stays_on_track(&x_all);
    }

    return total_steps - 1;
}

fn load_track_problem() -> TrackProblem {
    let (bl, br, cline, thetas) = load_test_track();
    let (obs_p0x, obs_p0y, obs_len, obs_dirx, obs_diry) = boundaries_to_obs(thetas.len(), &bl, &br);
    TrackProblem { bl, br, cline, thetas, obs_p0x, obs_p0y, obs_len, obs_dirx, obs_diry }
}

fn integrate_for_collision(tp: &TrackProblem, steps_used: usize, rk4_div: usize, controls: &[f64], x_end: &mut [f64]) -> (f64, usize, f64) {
    let n = 6;
    let m = 2;
    let n_steps = controls.len() / m;
    let x0 = [287.0, 5.0, -176.0, 0.0, 2.0, 0.0];
    let dt = 0.05;

    let mut old_xs = x0.to_vec();
    let mut new_xs = vec![0.0; n];

    let mut min_sign_sq_dist = f64::MAX;
    let mut min_ssd_i = 0;

    let track_n = tp.thetas.len();
    let ref_x = &tp.cline[0..track_n];
    let ref_y = &tp.cline[track_n..track_n*2];
    let mut track_i = 0;

    for k in 0..steps_used {
        let mut ode_fun = |_t: f64, x: &[f64], dx: &mut [f64]| {
            bike_fun(x, controls[k], controls[k + n_steps], dx);
        };
        rk4_integrate(dt / rk4_div as f64, rk4_div, &mut ode_fun, &old_xs, &mut new_xs);
        old_xs.copy_from_slice(&new_xs);

        track_i = next_track_idx(track_i, ref_x, ref_y, new_xs[0], new_xs[2]);

        let ssd = closest_obj_signed_sq_dist(&tp, new_xs[0], new_xs[2]);
        if ssd < min_sign_sq_dist {
            min_sign_sq_dist = ssd;
            min_ssd_i = k;
        }
        if ssd < 0.0 {
            break;
        }
    }

    x_end.copy_from_slice(&new_xs);

    (min_sign_sq_dist, min_ssd_i, track_i as f64 / (track_n as f64 - 1.0))
}

fn refine_solution(tp: TrackProblem, ref_controls: &[f64]) {
    // the idea for refining the solution will be to take a window
    // and try to perform certain permutations on it
    // for example, can we find braking or turning that isn't necessary and eliminate it?
    // can we find a time step to eliminate entirely?

    // we want our own mutable copy
    let mut ref_controls = ref_controls.to_vec();

    // or... let's try working backwards, maintaining forward success!
    let ref_steps = tp.thetas.len();
    let total_steps = ref_steps * 20;
    let tp = resample_track(&tp, total_steps);
    let track_n = tp.thetas.len();
    let ref_x = &tp.cline[0..track_n];
    let ref_y = &tp.cline[track_n..track_n*2];
    let dt = 0.05;
    let rk4_div = 2;

    let n = 6;
    let m = 2;
    let n_steps = ref_controls.len() / m;
    let mut steps_used = n_steps;
    let mut x_end = vec![0.0; n];
    let (mut min_ssd, mut min_ssd_i, start_completion) = integrate_for_collision(&tp, steps_used, rk4_div, &ref_controls, &mut x_end);
    println!("Start min sign sq dist: {:.2}, {}", min_ssd, min_ssd_i);
    println!("Start completion amount: {:.2}", start_completion);

    let mut controls = ref_controls.to_vec();

    let mut track_i = track_n - 1;

    let mut changes_made = true;
    let mut epoch = 0;
    while changes_made {
        changes_made = false;

        println!("\nStarting epoch {}", epoch);

        // try to whole-house slaughter a time step
        let mut slaughter_controls = ref_controls.to_vec();
        for k in 1..steps_used.min(1150) {
            if k % 50 == 0 {
                println!("SK: {}", k);
            }
            slaughter_controls.clear();
            slaughter_controls.extend_from_slice(&ref_controls);
            slaughter_controls[k - 1] = (slaughter_controls[k - 1] + slaughter_controls[k]).max(-0.5).min(0.5);
            slaughter_controls.remove(k);
            slaughter_controls.insert(n_steps - 1, 0.0);
            slaughter_controls[n_steps + k - 1] = (slaughter_controls[n_steps + k - 1] + slaughter_controls[n_steps + k]).max(-5000.0).min(2500.0);
            slaughter_controls.remove(n_steps + k);
            slaughter_controls.insert(n_steps * 2 - 1, 0.0);

            // is that okay?
            let (new_min_ssd, new_ssd_i, new_completion) = integrate_for_collision(&tp, steps_used, rk4_div, &slaughter_controls, &mut x_end);
            let okay1 = new_min_ssd > 0.25;
            // let check_min_ssd = if okay1 { new_min_ssd } else { integrate_for_collision(&tp, steps_used, rk4_div*2, &slaughter_controls, &mut x_end_check).0 };
            if okay1 && new_completion >= 1.0 {
                changes_made = true;
                min_ssd = new_min_ssd;
                min_ssd_i = new_ssd_i;

                controls.clear();
                controls.extend_from_slice(&slaughter_controls);
                ref_controls.clear();
                ref_controls.extend_from_slice(&slaughter_controls);

                println!("removed timestep {}: {:.2}, {}", k, min_ssd, min_ssd_i);

                track_i = next_track_idx(track_i, ref_x, ref_y, x_end[0], x_end[2]);
                steps_used -= 1;
            }
        }

        // for k in (0..n_steps-1).rev() {
        //     if k % 50 == 0 {
        //         println!("K: {}", k);
        //     }
        //
        //     // try to increase acceleration
        //     for inc in [2500.0, 500.0, 100.0].iter() {
        //         while controls[n_steps + k] < 2500.0 {
        //             controls[n_steps + k] = (controls[n_steps + k] + inc).min(2500.0);
        //
        //             // is that okay?
        //             let (new_min_ssd, new_ssd_i) = integrate_for_collision(&tp, steps_used, rk4_div, &controls, &mut x_end);
        //             let okay1 = new_min_ssd > 0.25;
        //             let check_min_ssd = if okay1 { new_min_ssd } else { integrate_for_collision(&tp, steps_used, rk4_div*2, &controls, &mut x_end_check).0 };
        //             if okay1 && check_min_ssd > 0.25 {
        //                 changes_made = true;
        //                 min_ssd = new_min_ssd;
        //                 min_ssd_i = new_ssd_i;
        //                 ref_controls[n_steps + k] = controls[n_steps + k];
        //                 println!("accel to {:.0}: {:.2}, {:.2}, {}", controls[n_steps + k], min_ssd, check_min_ssd, min_ssd_i);
        //
        //                 track_i = next_track_idx(track_i, ref_x, ref_y, x_end[0], x_end[2]);
        //                 // let completion = track_i as f64 / (track_n - 1) as f64 * 100.0;
        //                 // if completion >= 100.0 {
        //                 //     println!("track already complete ({:.2}, {:.2}); removing last point(s)!", x_end[0], x_end[2]);
        //                 //     steps_used -= 1;
        //                 // }
        //             } else {
        //                 controls[n_steps + k] = ref_controls[n_steps + k];
        //                 break;
        //             }
        //         }
        //     }
        //
        //     // try to reduce steering
        //     for inc in [0.5f64, 0.1, 0.025].iter() {
        //         while controls[k] != 0.0 {
        //             controls[k] -= inc.min(controls[k].abs()) * controls[k].signum();
        //             if controls[k].abs() < 0.025 {
        //                 controls[k] = 0.0;
        //             }
        //             // is that okay?
        //             let (new_min_ssd, new_ssd_i) = integrate_for_collision(&tp, steps_used, rk4_div, &controls, &mut x_end);
        //             let okay1 = new_min_ssd > 0.25 && new_min_ssd >= min_ssd - 0.01;
        //             let check_min_ssd = if okay1 { new_min_ssd } else { integrate_for_collision(&tp, steps_used, rk4_div*2, &controls, &mut x_end_check).0 };
        //             if okay1 && check_min_ssd > 0.25 {
        //                 changes_made = true;
        //                 min_ssd = new_min_ssd;
        //                 min_ssd_i = new_ssd_i;
        //                 println!("steering reduced from {:.2} to {:.2}: {:.2}, {:.2}, {}", ref_controls[k], controls[k], min_ssd, check_min_ssd, min_ssd_i);
        //                 ref_controls[k] = controls[k];
        //
        //                 track_i = next_track_idx(track_i, ref_x, ref_y, x_end[0], x_end[2]);
        //                 // let completion = track_i as f64 / (track_n - 1) as f64 * 100.0;
        //                 // if completion >= 100.0 {
        //                 //     println!("track already complete ({:.2}, {:.2}); removing last point(s)!", x_end[0], x_end[2]);
        //                 //     steps_used -= 1;
        //                 // }
        //             } else {
        //                 controls[k] = ref_controls[k];
        //                 break;
        //             }
        //         }
        //     }
        // }

        let mut continue_clearing = true;
        while min_ssd < 1.0 && continue_clearing {
            continue_clearing = false;
            'outer: for k in (0..min_ssd_i).rev() {
                if k % 50 == 0 {
                    println!("Steer k: {}", k);
                }
                if min_ssd > 1.0 {
                    break;
                }

                // try to increase steering for better clearance
                for &inc in [0.5, 0.1, 0.025, -0.025].iter() {
                    // last negative element only intended for 0.0 control values
                    if controls[k] != 0.0 && inc < 0.0 {
                        break;
                    }
                    while controls[k].abs() < 0.5 && min_ssd < 1.0 {
                        controls[k] = (controls[k] + inc * controls[k].signum()).max(-0.5).min(0.5);
                        // is that okay?
                        let (new_min_ssd, new_ssd_i, new_completion) = integrate_for_collision(&tp, steps_used, rk4_div, &controls, &mut x_end);
                        let okay1 = new_min_ssd >= min_ssd + 0.01;
                        // let check_min_ssd = if okay1 { new_min_ssd } else { integrate_for_collision(&tp, steps_used, rk4_div*2, &controls, &mut x_end_check).0 };
                        if okay1 && new_completion >= 1.0 {
                            changes_made = true;
                            min_ssd = new_min_ssd;
                            let different_ssd_i = min_ssd_i != new_ssd_i;
                            min_ssd_i = new_ssd_i;
                            println!("steering increased for clearance from {:.2} to {:.2}: {:.2}, {}", ref_controls[k], controls[k], min_ssd, min_ssd_i);
                            ref_controls[k] = controls[k];

                            track_i = next_track_idx(track_i, ref_x, ref_y, x_end[0], x_end[2]);
                            // let completion = track_i as f64 / (track_n - 1) as f64 * 100.0;
                            // if completion >= 100.0 {
                            //     println!("track already complete ({:.2}, {:.2}); removing last point(s)!", x_end[0], x_end[2]);
                            //     steps_used -= 1;
                            // }
                            if different_ssd_i {
                                continue_clearing = true;
                                break 'outer;
                            }
                        } else {
                            controls[k] = ref_controls[k];
                            break;
                        }
                    }
                }
            }
        }

        // look for steering pairs (opposite signs close together)
        // if !changes_made || epoch % 4 == 0 {
        //     for diff in 1..80 {
        //         println!("Steering diff: {}", diff);
        //         for k in 0..n_steps-1-diff {
        //             let v1 = controls[k];
        //             let v2 = controls[k + diff];
        //             if v1 != 0.0 && v1 * v2 < 0.0 {
        //                 let remove_amount = v1.abs().min(v2.abs());
        //                 controls[k] -= remove_amount * v1.signum();
        //                 controls[k + diff] -= remove_amount * v2.signum();
        //                 let (new_min_ssd, new_ssd_i) = integrate_for_collision(&tp, steps_used, rk4_div, &controls, &mut x_end);
        //                 let okay1 = new_min_ssd > 0.25;
        //                 let check_min_ssd = if okay1 { new_min_ssd } else { integrate_for_collision(&tp, steps_used, rk4_div*2, &controls, &mut x_end_check).0 };
        //                 if okay1 && check_min_ssd > 0.25 {
        //                     changes_made = true;
        //                     min_ssd = new_min_ssd;
        //                     min_ssd_i = new_ssd_i;
        //                     ref_controls[k] = controls[k];
        //                     ref_controls[k + diff] = controls[k + diff];
        //                     println!("steering pair {} apart at k {} ({:.2}, {:.2}) reduced: {:.2}, {:.2}, {}", diff, k, v1, v2, min_ssd, check_min_ssd, min_ssd_i);
        //                     track_i = next_track_idx(track_i, ref_x, ref_y, x_end[0], x_end[2]);
        //                 } else {
        //                     controls[k] = ref_controls[k];
        //                     controls[k + diff] = ref_controls[k + diff];
        //                 }
        //             }
        //         }
        //     }
        // }
        //
        // // look for accel pairs (opposite signs close together)
        // if !changes_made || epoch % 4 == 0 {
        //     for diff in 1..80 {
        //         println!("Accel diff: {}", diff);
        //         for k in 0..n_steps-1-diff {
        //             let v1 = controls[n_steps + k];
        //             let v2 = controls[n_steps + k + diff];
        //             if v1 != 0.0 && v1 * v2 < 0.0 {
        //                 let remove_amount = v1.abs().min(v2.abs());
        //                 controls[n_steps + k] -= remove_amount * v1.signum();
        //                 controls[n_steps + k + diff] -= remove_amount * v2.signum();
        //                 let (new_min_ssd, new_ssd_i) = integrate_for_collision(&tp, steps_used, rk4_div, &controls, &mut x_end);
        //                 let okay1 = new_min_ssd > 0.25;
        //                 let check_min_ssd = if okay1 { new_min_ssd } else { integrate_for_collision(&tp, steps_used, rk4_div*2, &controls, &mut x_end_check).0 };
        //                 if okay1 && check_min_ssd > 0.25 {
        //                     changes_made = true;
        //                     min_ssd = new_min_ssd;
        //                     min_ssd_i = new_ssd_i;
        //                     ref_controls[n_steps + k] = controls[n_steps + k];
        //                     ref_controls[n_steps + k + diff] = controls[n_steps + k + diff];
        //                     println!("accel pair {} apart at k {} ({:.2}, {:.2}) reduced: {:.2}, {:.2}, {}", diff, k, v1, v2, min_ssd, check_min_ssd, min_ssd_i);
        //                     track_i = next_track_idx(track_i, ref_x, ref_y, x_end[0], x_end[2]);
        //                 } else {
        //                     controls[n_steps + k] = ref_controls[n_steps + k];
        //                     controls[n_steps + k + diff] = ref_controls[n_steps + k + diff];
        //                 }
        //             }
        //         }
        //     }
        // }

        let mut track_i = 0;
        let xs = forward_integrate_bike(rk4_div, &controls);
        // measure time to completion
        for k in 0..steps_used {
            track_i = next_track_idx(track_i, ref_x, ref_y, xs[k], xs[n_steps*2 + k]);
            if track_i >= track_n - 1 {
                steps_used = (k + 4).min(n_steps);
                println!("completed track in {:.2} seconds", k as f64 * dt);
                break;
            }
        }

        epoch += 1;
    }

    let xs = forward_integrate_bike(rk4_div, &controls);

    let mut f = File::create("refined_xs.txt").unwrap();
    write!(f, "[").unwrap();
    for i in 0..steps_used {
        for j in 0..n {
            write!(f, "{:.5}, ", xs[j * n_steps + i]).unwrap();
        }
        write!(f, "; ").unwrap();
    }
    writeln!(f, "];").unwrap();

    let mut f = File::create("refined_us.txt").unwrap();
    write!(f, "[").unwrap();
    for i in 0..steps_used {
        for j in 0..m {
            write!(f, "{:.5}, ", controls[j * n_steps + i]).unwrap();
        }
        write!(f, "; ").unwrap();
    }
    writeln!(f, "];").unwrap();

    // plot_trajectory(&tp, steps_used, &xs);
}

pub fn run_refine_solution() {
    let m = 2;
    let n_steps = 1444;
    let mut controls_trans = vec![0.0; n_steps * m];
    fill_from_csv("us71.txt", &mut controls_trans).unwrap();
    let mut controls = vec![0.0; n_steps * m];
    for k in 0..n_steps {
        for i in 0..m {
            controls[i * n_steps + k] = controls_trans[k * m + i];
        }
    }

    let tp = load_track_problem();
    refine_solution(tp, &controls);
}

pub fn run_problem1() {
    let tp = load_track_problem();
    let mut controls = vec![0.0; 2048 * 16];
    shooting_solve(tp, &mut controls);
}
