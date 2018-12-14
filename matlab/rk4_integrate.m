function [x1, dx0, du] = rk4_integrate(x0, u, dt, nSteps)
x1 = x0;
h = dt/nSteps;
dx0 = eye(6);
du = zeros(6,2);
for i=1:nSteps
    x = x1;
    k1 = nonl_bike([x; u]);
    k2 = nonl_bike([x + 0.5*h*k1; u]);
    k3 = nonl_bike([x + 0.5*h*k2; u]);
    k4 = nonl_bike([x + h*k3; u]);
    x1 = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);

    [A1, B1] = getDerivatives(x, u);
    [A2, B2] = getDerivatives(x + 0.5*h*k1, u);
    [A3, B3] = getDerivatives(x + 0.5*h*k2, u);
    [A4, B4] = getDerivatives(x + h*k3, u);
    
    dk1dx = A1;
    dk2dx = A2*(eye(6) + 0.5*h*dk1dx);
    dk3dx = A3*(eye(6) + 0.5*h*dk2dx);
    dk4dx = A4*(eye(6) + h*dk3dx);

    dx1dx = eye(6) + (h/6)*(dk1dx + 2*dk2dx + 2*dk3dx + dk4dx);
    dx0 = dx1dx*dx0;
    
    dk1du = B1;
    dk2du = (h/2)*A2*dk1du + B2;
    dk3du = (h/2)*A3*dk2du + B3;
    dk4du = h*A4*dk3du + B4;
    dx1du = (h/6)*(dk1du + 2*dk2du + 2*dk3du + dk4du);
    du = dx1dx*du + dx1du;
end
end