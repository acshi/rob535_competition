function [C, dC] = rk4_cons(X, dt)
Fx_scale = 1000;
%tic;
nSteps = size(X, 1)/8;
C = zeros(6*(nSteps - 1), 1);
dC = spalloc(6*(nSteps - 1), size(X,1), (6 + 36 + 12)*(nSteps - 1));
for i=1:(nSteps-1)
    x = X(8*i-7:8*i-2);
    u = X(8*i-1:8*i);
    u(2) = Fx_scale*u(2);
    x1 = rk4_integrate(x, u, dt, 4);
    C(6*i-5:6*i) = X(8*i+1:8*i+6) - x1;
    
    k1 = nonl_bike([x; u]);
    k2 = nonl_bike([x + 0.5*dt*k1; u]);
    k3 = nonl_bike([x + 0.5*dt*k2; u]);
    k4 = nonl_bike([x + dt*k3; u]);
    x1 = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    
    dC(6*i-5:6*i, 8*i+1:8*i+6) = eye(6);
    [A1, B1] = getDerivatives(x, u);
    [A2, B2] = getDerivatives(x + 0.5*dt*k1, u);
    [A3, B3] = getDerivatives(x + 0.5*dt*k2, u);
    [A4, B4] = getDerivatives(x + dt*k3, u);
    B1(:,2) = Fx_scale*B1(:,2);
    B2(:,2) = Fx_scale*B2(:,2);
    B3(:,2) = Fx_scale*B3(:,2);
    B4(:,2) = Fx_scale*B4(:,2);
    
    dk1dx = A1;
    dk2dx = A2*(eye(6) + 0.5*dt*dk1dx);
    dk3dx = A3*(eye(6) + 0.5*dt*dk2dx);
    dk4dx = A4*(eye(6) + dt*dk3dx);

    dC(6*i-5:6*i, 8*i-7:8*i-2) = -(eye(6) + (dt/6)*(dk1dx + 2*dk2dx + 2*dk3dx + dk4dx));
    
    dk1du = B1;
    dk2du = (dt/2)*A2*dk1du + B2;
    dk3du = (dt/2)*A3*dk2du + B3;
    dk4du = dt*A4*dk3du + B4;
    dC(6*i-5:6*i, 8*i-1:8*i) = -(dt/6)*(dk1du + 2*dk2du + 2*dk3du + dk4du);
end
end