function [C, dC] = euler_cons(X, dt)
Fx_scale = 1000;
%tic;
nSteps = size(X, 1)/8;
C = zeros(6*(nSteps - 1), 1);
for i=1:(nSteps-1)
    arg = X(8*i-7:8*i);
    arg(8) = Fx_scale*arg(8);
    C(6*i-5:6*i) = X(8*i+1:8*i+6) - (X(8*i-7:8*i-2) + dt*nonl_bike(arg));
end
%fprintf('euler_cons: %f\n', toc);
if nargout == 1
    return;
end

dC = spalloc(6*(nSteps - 1), size(X,1), (6 + 36 + 12)*(nSteps - 1));
for i=1:nSteps-1
    dC(6*i-5:6*i, 8*i+1:8*i+6) = eye(6);
    [A, B] = getDerivatives(X(8*i-7:8*i-2), X(8*i-1:8*i));
    B(:,2) = Fx_scale*B(:,2);
    dC(6*i-5:6*i, 8*i-7:8*i-2) = -eye(6) - dt*A;
    dC(6*i-5:6*i, 8*i-1:8*i) = -dt*B;
end
end