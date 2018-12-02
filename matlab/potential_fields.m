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

for i=1:nSteps
    disp(i);
    force = electric_force([x(1); x(3)], testTrack, XObs);
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
    options = odeset('RelTol',1e-8,'AbsTol',1e-10);
    [~,Y]=ode45(@(t,x)bike(t,x,T,[delta, delta; f_x, f_x]'),T,x, options);
    x = Y(end, :);
end
end

function dzdt=bike(t,x,T,U)
%constants
Nw=2;
f=0.01;
Iz=2667;
a=1.35;
b=1.45;
By=0.27;
Cy=1.2;
Dy=0.7;
Ey=-1.6;
Shy=0;
Svy=0;
m=1400;
g=9.806;


%generate input functions
delta_f=interp1(T,U(:,1),t,'previous','extrap');
F_x=interp1(T,U(:,2),t,'previous','extrap');

%slip angle functions in degrees
a_f=rad2deg(delta_f-atan2(x(4)+a*x(6),x(2)));
a_r=rad2deg(-atan2((x(4)-b*x(6)),x(2)));

%Nonlinear Tire Dynamics
phi_yf=(1-Ey)*(a_f+Shy)+(Ey/By)*atan(By*(a_f+Shy));
phi_yr=(1-Ey)*(a_r+Shy)+(Ey/By)*atan(By*(a_r+Shy));

F_zf=b/(a+b)*m*g;
F_yf=F_zf*Dy*sin(Cy*atan(By*phi_yf))+Svy;

F_zr=a/(a+b)*m*g;
F_yr=F_zr*Dy*sin(Cy*atan(By*phi_yr))+Svy;

F_total=sqrt((Nw*F_x)^2+(F_yr^2));
F_max=0.7*m*g;

if F_total>F_max
    
    F_x=F_max/F_total*F_x;
  
    F_yr=F_max/F_total*F_yr;
end

%vehicle dynamics
dzdt= [x(2)*cos(x(5))-x(4)*sin(x(5));...
          (-f*m*g+Nw*F_x-F_yf*sin(delta_f))/m+x(4)*x(6);...
          x(2)*sin(x(5))+x(4)*cos(x(5));...
          (F_yf*cos(delta_f)+F_yr)/m-x(2)*x(6);...
          x(6);...
          (F_yf*a*cos(delta_f)-F_yr*b)/Iz];
end