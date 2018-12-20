use std::fs::File;
use std::io::Read;
use std::f64;

extern crate rustplotlib;
use rustplotlib::{Axes2D, Line2D, Backend};
use rustplotlib::backend::Matplotlib;

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

fn bike_fun(x: &[f64], delta_f: f64, mut f_x: f64, dx: &mut [f64]) {
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
    let mut f = File::open(filename)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let mut buf_i = 0;
    for i in 0..vals.len() {
        let mut next_nondigit = buf_i;
        while is_float_digit(buf[next_nondigit] as char) {
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
    let mut theta = vec![0.0; track_n];
    fill_from_csv("bl.csv", &mut bl).unwrap();
    fill_from_csv("br.csv", &mut br).unwrap();
    fill_from_csv("cline.csv", &mut cline).unwrap();
    fill_from_csv("theta.csv", &mut theta).unwrap();

    (bl, br, cline, theta)
}

fn trajectory_stays_on_track(x_all: &[f64]) -> bool {
    let n = 6;
    let n_steps = x_all.len() / n;
    let (bls, brs, clines, thetas) = load_test_track();
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
        let l_sideways = sint * bls[track_i] + cost * bls[track_n + track_i];
        let r_sideways = sint * brs[track_i] + cost * brs[track_n + track_i];
        let forward = sint * clines[track_n + track_i] + cost * clines[track_i];
        // and where the next section starts in the forward dimension
        let next_forward = sint * clines[track_n + track_i + 1] + cost * clines[track_i + 1];

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
            .data(&bls[0..track_n], &bls[track_n..track_n*2])
            .color("blue")
            .linestyle("-"))
        .add(Line2D::new("")
            .data(&brs[0..track_n], &brs[track_n..track_n*2])
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
