function [A, B] = getDerivatives(state, controls)
%Contstants
m = 1400;
Nw = 2;
f = 0.01;
Iz = 2667;
a = 1.35;
b = 1.45;
By = 0.27;
Cy = 1.2;
Dy = 0.7;
Ey = -1.6;
Shy = 0;
Svy = 0;
g = 9.806;

x_ind = 1;
u_ind = 2;
y_ind = 3;
v_ind = 4;
psi_ind = 5;
r_ind = 6;

x = state(x_ind);
u = state(u_ind);
y = state(y_ind);
v = state(v_ind);
psi = state(psi_ind);
r = state(r_ind);

df_ind = 1;
Fx_ind = 2;

df = controls(df_ind);
Fx = controls(Fx_ind);

A = zeros(6,6);
B = zeros(6,2);

A(x_ind, u_ind) = cos(psi);
A(x_ind, v_ind) = -sin(psi);
A(x_ind, psi_ind) = -u*sin(psi) - v*cos(psi);

A(y_ind, u_ind) = sin(psi);
A(y_ind, v_ind) = cos(psi);
A(y_ind, psi_ind) = u*cos(psi) - v*sin(psi);

A(psi_ind, r_ind) = 1;

d_alpha_f_d_df = 1;
d_alpha_f_d_v = -u/(u^2 + (v + a*r)^2);
d_alpha_f_d_u = (v + a*r)/(u^2 + (v + a*r)^2);
d_alpha_f_d_r = -a*u/(u^2 +(v + a*r)^2);

d_alpha_r_d_v = -u/(u^2 + (v - b*r)^2);
d_alpha_r_d_u = (v - b*r)/(u^2 + (v - b*r)^2);
d_alpha_r_d_r = b*u/(u^2 + (v - b*r)^2);

alpha_f = df - atan((v + a*r)/u);
d_phi_d_alpha_f = (1 - Ey) + Ey/(1 + (By*(alpha_f*pi/180 + Shy))^2);
alpha_r = -atan((v - b*r)/u);
d_phi_d_alpha_r = (1 - Ey) + Ey/(1 + (By*(alpha_r*pi/180 + Shy))^2);

Fzr = a*m*g/(a + b);
Fzf = b*m*g/(a + b);
phi_r = (1 - Ey)*(alpha_r + Shy) + (Ey/By)*atan(By*(alpha_r + Shy));
d_Fyr_d_phi = Fzr*Dy*cos(Cy*atan(By*phi_r))*Cy*By/(1 + (By*phi_r)^2);
phi_f = (1 - Ey)*(alpha_f + Shy) + (Ey/By)*atan(By*(alpha_f + Shy));
d_Fyf_d_phi = Fzf*Dy*cos(Cy*atan(By*phi_f))*Cy*By/(1 + (By*phi_f)^2);

Fyr = Fzr*Dy*sin(Cy*atan(By*phi_r)) + Svy;
Fyf = Fzf*Dy*sin(Cy*atan(By*phi_f)) + Svy;

Ftotal = 0.7*m*g;
Fmax = sqrt((Nw*Fx)^2 + (Fyr)^2);

if Ftotal > Fmax
    n = Fmax/Ftotal;
    d_n_d_Fx = -0.5*Fmax*2*Nw*Nw*Fx*((Nw*Fx)^2 + (Fyr)^2)^(-1.5);
    d_n_d_Fyr = -0.5*Fmax*2*Fyr*((Nw*Fx)^2 + (Fyr)^2)^(-1.5);
else
    n = 1;
    d_n_d_Fx = 0;
    d_n_d_Fyr = 0;
end

d_Fsyr_d_Fx = d_n_d_Fx*Fyr;
d_Fsx_d_Fx = d_n_d_Fx*Fx + n;
d_Fsyr_d_Fyr = d_n_d_Fyr*Fyr + n;
d_Fsx_d_Fyr = d_n_d_Fyr*Fx;

d_Fyr_d_r = d_Fyr_d_phi*d_phi_d_alpha_r*d_alpha_r_d_r;
d_Fyr_d_u = d_Fyr_d_phi*d_phi_d_alpha_r*d_alpha_r_d_u;
d_Fyr_d_v = d_Fyr_d_phi*d_phi_d_alpha_r*d_alpha_r_d_v;
d_Fsyr_d_r = d_Fsyr_d_Fyr*d_Fyr_d_r;
d_Fsyr_d_u = d_Fsyr_d_Fyr*d_Fyr_d_u;
d_Fsyr_d_v = d_Fsyr_d_Fyr*d_Fyr_d_v;

d_Fyf_d_r = d_Fyf_d_phi*d_phi_d_alpha_f*d_alpha_f_d_r;
d_Fyf_d_u = d_Fyf_d_phi*d_phi_d_alpha_f*d_alpha_f_d_u;
d_Fyf_d_v = d_Fyf_d_phi*d_phi_d_alpha_f*d_alpha_f_d_v;
d_Fyf_d_df = d_Fyf_d_phi*d_phi_d_alpha_f*d_alpha_f_d_df;

A(r_ind, r_ind) = (1/Iz)*(a*cos(df)*d_Fyf_d_r - b*d_Fsyr_d_r);
A(r_ind, u_ind) = (1/Iz)*(a*cos(df)*d_Fyf_d_u - b*d_Fsyr_d_u);
A(r_ind, v_ind) = (1/Iz)*(a*cos(df)*d_Fyf_d_v - b*d_Fsyr_d_v);
B(r_ind, Fx_ind) = (-b/Iz)*d_Fsyr_d_Fx;
B(r_ind, df_ind) = (1/Iz)*(a*cos(df)*d_Fyf_d_df - a*Fyf*sin(df));

A(u_ind, u_ind) = (1/m)*(Nw*d_Fsx_d_Fyr*d_Fyr_d_u - d_Fyf_d_u*sin(df));
A(u_ind, v_ind) = (1/m)*(Nw*d_Fsx_d_Fyr*d_Fyr_d_v - d_Fyf_d_v*sin(df)) + r;
A(u_ind, r_ind) = (1/m)*(Nw*d_Fsx_d_Fyr*d_Fyr_d_r - d_Fyr_d_r*sin(df)) + v;
B(u_ind, Fx_ind) = (1/m)*(Nw*d_Fsx_d_Fx);
B(u_ind, df_ind) = (1/m)*(-sin(df)*d_Fyf_d_df - a*Fyf*cos(df));

A(v_ind, u_ind) = (1/m)*(cos(df)*d_Fyf_d_u + d_Fsyr_d_u) - r;
A(v_ind, v_ind) = (1/m)*(cos(df)*d_Fyf_d_v + d_Fsyr_d_v);
A(v_ind, r_ind) = (1/m)*(cos(df)*d_Fyf_d_r + d_Fsyr_d_r) - u;
B(v_ind, Fx_ind) = (1/m)*d_Fsyr_d_Fx;
B(v_ind, df_ind) = (1/m)*(cos(df)*d_Fyf_d_df - sin(df)*Fyf);
end