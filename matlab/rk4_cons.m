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
    [x1, dx0, du] = rk4_integrate(x, u, dt, 4);
    C(6*i-5:6*i) = X(8*i+1:8*i+6) - x1;
    dC(6*i-5:6*i, 8*i+1:8*i+6) = eye(6);
    du(:,2) = Fx_scale*du(:,2);
    dC(6*i-5:6*i, 8*i-7:8*i-2) = -dx0;
    dC(6*i-5:6*i, 8*i-1:8*i) = -du;
end
end