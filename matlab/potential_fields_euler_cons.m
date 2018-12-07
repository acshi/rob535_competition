function X = potential_fields_euler_cons(x_ic, nSteps, testTrack, XObs)
X = zeros(nSteps + 1, 8);
x = x_ic;

k_psi = 10.;
k_u = 1000.;
u_min = 0.1;
u_target = 15.;
delta_min = -0.5;
delta_max = 0.5;
f_x_min = -5000.;
f_x_max = 2500.;
dt = 0.1;

lookahead = 8;

for i=1:10
    force = electric_force([x(1); x(3)] + lookahead*[cos(x(5)); sin(x(5))], testTrack, XObs);
    force = force/norm(force);
    dir = [cos(x(5)); sin(x(5))];
    cos_d = force'*dir;
    sin_d = force(1)*dir(2) - force(2)*dir(1);
    psi_err = atan2(sin_d, cos_d);
    delta = -k_psi*psi_err;
    if delta < delta_min
        delta = delta_min;
    elseif delta > delta_max
        delta = delta_max;
    end
    
    u_d = u_target*(0.5*(cos_d + 1))^5;
    if u_d < u_min
        u_d = u_min;
    end
    
    f_x = -k_u*(x(2) - u_d);
    if f_x < f_x_min
        f_x = f_x_min;
    elseif f_x > f_x_max
        f_x = f_x_max;
    end
    X(i,:) = [x, delta, f_x];
    x = x + dt*nonl_bike([x';delta;f_x])';
end
X(end,:) = [x, 0, 0];
end

