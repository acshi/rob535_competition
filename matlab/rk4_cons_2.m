function [C, dC, Cineq, dCineq] = rk4_cons_2(X, dt)
Fx_scale = 1000;
%tic;
nSteps = size(X, 1)/8;
C = zeros(6*(nSteps - 1), 1);
dC = spalloc(6*(nSteps - 1), size(X,1), (6 + 36 + 12)*(nSteps - 1));
Cineq = zeros(nSteps-1,1);
dCineq = spalloc(6*(nSteps-1), size(X,1), (48)*(nSteps - 1));
for i=1:(nSteps-1)
    x = X(8*i-7:8*i-2);
    u = X(8*i-1:8*i);
    u(2) = Fx_scale*u(2);
    [x1_4, dx0_4, du_4] = rk4_integrate(x, u, dt, 4);
    du_4(:,2) = Fx_scale*du_4(:,2);
        
    C(6*i-5:6*i) = X(8*i+1:8*i+6) - x1_4;
    dC(6*i-5:6*i, 8*i+1:8*i+6) = eye(6);
    dC(6*i-5:6*i, 8*i-7:8*i-2) = -dx0_4;
    dC(6*i-5:6*i, 8*i-1:8*i) = -du_4;
    
    [x1_2, dx0_2, du_2] = rk4_integrate(x, u, dt, 2);
    du_2(:,2) = Fx_scale*du_2(:,2);

    scale = 1e8;
    Cineq(6*i-5:6*i) = scale*(x1_4 - x1_2).^2 - 1e-8;
    dCineq(6*i-5:6*i, 8*i-7:8*i-2) = scale*2*(x1_4-x1_2).*(dx0_4 - dx0_2);
    dCineq(6*i-5:6*i, 8*i-1:8*i) = scale*2*(x1_4-x1_2).*(du_4 - du_2);
end
end