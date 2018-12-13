function [x1] = rk4_integrate(x0, u, dt, nSteps)
x1 = x0;
h = dt/nSteps;
for i=1:nSteps
    x = x1;
    k1 = nonl_bike([x; u]);
    k2 = nonl_bike([x + 0.5*h*k1; u]);
    k3 = nonl_bike([x + 0.5*h*k2; u]);
    k4 = nonl_bike([x + h*k3; u]);
    x1 = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end
end