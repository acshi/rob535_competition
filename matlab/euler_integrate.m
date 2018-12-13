function [x1] = euler_integrate(x0, u, dt, nSteps)
x1 = x0;
h = dt/nSteps;
for i=1:nSteps
    x1 = x1 + h*nonl_bike([x1, u])';
end
end