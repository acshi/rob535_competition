function [C] = euler_cons(X, dt)
nSteps = size(X, 1)/8;
C = zeros(6*(nSteps - 1), 1);
for i=1:(nSteps-1)
    C(6*i-5:6*i) = X(8*i+1:8*i+6) - (X(8*i-7:8*i-2) + dt*nonl_bike(X(8*i-7:8*i)));
end
end