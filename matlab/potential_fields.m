function [U, X] = potential_fields(testTrack, XObs)
nSteps = 1500;
U = zeros(nSteps, 2);
X = zeros(nSteps, 6);
x = [287,5,-176,0,2,0];

k_psi = 10.;
k_u = 1000.;
u_min = 0.1;
u_target = 15.;
delta_min = -0.5;
delta_max = 0.5;
f_x_min = -5000.;
f_x_max = 2500.;
dt = 0.1;

rem = -1;

lookahead = 8;

i=1;
while true
    i = i + 1;
    disp(i);
    if nargin == 2
        force = electric_force([x(1); x(3)] + lookahead*[cos(x(5)); sin(x(5))], testTrack, XObs);
    else
        force =  electric_force([x(1); x(3)] + lookahead*[cos(x(5)); sin(x(5))], testTrack);
    end
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
    U(i, :) = [delta, f_x];
    X(i,:) = x;
    T = [0 dt];
    [~,Y]=ode45(@(t,x)nonl_bike([x; delta; f_x]),T,x);
    x = Y(end, :);
    
    dist = sum(([x(1); x(3)] - testTrack.cline).^2,1);
    [~,idx] = min(dist);
    if rem < 0 && idx == size(dist,2)
        % simulate a few more steps
        rem = 10;
    end
    rem = rem - 1;
    if rem == 0
        break;
    end
end
X = X(2:i,:);
U = U(2:i,:);
end

