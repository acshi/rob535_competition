function [sol, exitflag, output] = fmincon_improve_traj(X_init, dt, x_start, x_end, obs, obj_fun)
Fx_scale = 1000;
% TODO provide gradients
problem.solver = 'fmincon';
nSteps = size(X_init, 1);
%problem.Aeq = [eye(6), zeros(6, 8*nSteps - 6)];
%problem.Beq = x_ic;
problem.Aeq = zeros(12, numel(X_init));
problem.Beq = zeros(12, 1);

problem.Aeq(1:6,1:6) = eye(6);
problem.Beq(1:6) = x_start;
problem.Aeq(7:12,end-7:end-2) = eye(6);
problem.Beq(7:12) = x_end;

%problem.Aeq(7,numel(X_init) - 6) = 1;
%problem.Beq(7) = 5;

%problem.objective = @(x) sol_time(x, TestTrack, dt);
problem.objective = obj_fun;
problem.nonlcon = @nonlcon;
problem.x0 = reshape(X_init', [], 1);
problem.lb = -inf(size(problem.x0));
problem.ub = inf(size(problem.x0));
problem.lb(7:8:end) = -.5;
problem.ub(7:8:end) = .5;
problem.lb(8:8:end) = -5000/Fx_scale;
problem.ub(8:8:end) = 2500/Fx_scale;

% % Bound final speed.
%problem.lb(end - 6) = 0;
%problem.lb(end - 4) = -.1;
%problem.ub(2:8:end) = 15;
%problem.ub(end - 6) = 5;
%problem.ub(end - 4) = .1;

problem.options = optimoptions('fmincon',...
    'Display', 'none',...
    'SpecifyObjectiveGradient', true,...
    'SpecifyConstraintGradient', true,...
    'MaxFunctionEvaluations', 6000,...
    'MaxIterations', 300,...
    'CheckGradients', false,...
    'ConstraintTolerance', 1e-8);%    ,    'FiniteDifferenceType', 'central'
[h, g, dh, dg] = problem.nonlcon(problem.x0);
s0 = problem.objective(problem.x0);
[sol, ~, exitflag, output] = fmincon(problem);

    function [C, Ceq,  dC, dCeq] = nonlcon(x)
        [C, dC] = dist_to_obj(x, obs, 0.1);
        C = -C;
        dC = -dC';
        %C = [];
        %dC = [];
        [Ceq, dCeq] = rk4_cons(x, dt);
        dCeq = dCeq';
    end
end