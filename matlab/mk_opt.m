function [states, inputs] = mk_opt(states_ic, inputs_ic, lb, ub, u_step, integrate, objective, constrain)
states = states_ic;
inputs = inputs_ic;

step = 1e-1;

h = [];

plot_u = zeros(1, size(states,2));
plot_v = zeros(1, size(states,2));

plot_x = states(1,:);
plot_y = states(2,:);
plot_x2 = plot_x + plot_u;
plot_y2 = plot_y + plot_v;

rem_err = 0;
    
for i=1:10000
    plot_x_old = plot_x;
    plot_y_old = plot_y;
    plot_x2_old = plot_x2;
    plot_y2_old = plot_y2;
    plot_x = states(1,:);
    plot_y = states(2,:);
    plot_x2 = plot_x + plot_u;
    plot_y2 = plot_y + plot_v;
    if isempty(h)
        figure;
        hold on;
        ylim([-1 4]);
        xlim([-1 4]);
        t = 2*pi*(0:100)/100;
        plot(50*cos(t), 50*sin(t) + 60);
        h = plot(plot_x, plot_y, 'r.');
        h.XDataSource = 'plot_x';
        h.YDataSource = 'plot_y';
        m = plot(plot_x_old, plot_y_old, 'r.');
        m.XDataSource = 'plot_x_old';
        m.YDataSource = 'plot_y_old';
        g = plot(plot_x2, plot_y2, 'b.');
        g.XDataSource = 'plot_x2';
        g.YDataSource = 'plot_y2';
        n = plot(plot_x2_old, plot_y2_old, 'b.');
        n.XDataSource = 'plot_x2_old';
        n.YDataSource = 'plot_y2_old';
    else
        refreshdata(h, 'caller');
        refreshdata(g, 'caller');
        refreshdata(m, 'caller');
        refreshdata(n, 'caller');
    end
    drawnow;
    
    %disp(states);
    dstate = cell(size(states,2) - 1, 1);
    du = cell(size(states,2) - 1, 1);
    step_norm = 0;
    for i=1:size(states,2)-1
        [state, dstate{i}, du{i}] = integrate(states(:,i), inputs(:,i));
        step_norm = max(max(state - states(:,i+1)), step_norm);
        states(:,i+1) = state;
    end
    
   

    
    [obj, dobj] = objective(states);
    fprintf("%.6f %.6f %.6f\n",obj, step_norm, rem_err);
   
    errs = zeros(size(states));
    err = zeros(size(states,1), 1);
    for i=size(states, 2):-1:2
        err = err + step*dobj(:,i);
        
        % Satisfy nonlineq constraint. TODO better solution when objective
        % at this timestep conflicts with constraint. TODO multiple
        % constraints. TODO better solution when already in violation of
        % constraint.
        [C, dC] = constrain(states(:, i));
        
        % Want C(states_i + err_c) <= 0
        % C + dC*err_c <= 0;
        % C + a * dC*err <=0;
        % a <= -C/dC*err;
        if C > 0
            %disp('help we are in constraint violation.');
        end
        if C + dC'*err > 0 % TODO constraint tolerance
            % C + dC'*x = 0
            % Ax = b
            % x = A'(AA')-1 * b
            % dC*dC'*x = -dC*C
            err = err - (dC'*err)*dC/(dC'*dC);
            err = err + 0.1*(dC*(-C)/(dC'*dC));
        end
        
        plot_u(i) = err(1);
        plot_v(i) = err(2);
        errs(:,i) = err;
        
%         % Solve err = b*u_err. This should be a lin prog really... TODO
%         u_err = du{i-1}\err;
%         %inputs(:,i-1) = inputs(:,i-1) + u_err;
%         %inputs(:,i-1)
%         new_inputs = max(lb, min(ub, inputs(:,i-1) + u_err));
%         u_err_actual = new_inputs - inputs(:,i-1);
%         inputs(:,i-1) = new_inputs;
%         
%         % Solve err = A*err0
%         err = dstate{i-1}\(err - du{i-1}*u_err_actual);
%         A = [dstate{i-1}, du{i-1}];
%         z = A'*((A*A')\err);
%         err = z(1:size(state,1), 1);
%         u_err = z(size(state,1)+1:end);
%         inputs(:,i-1) =inputs(:,i-1) + u_err;
        A = dstate{i-1};
        B = du{i-1};
        H = zeros(size(states,1) + size(inputs,1));
        H(1:size(states,1),1:size(states,1)) = eye(size(states,1))*(size(states,2) - i + 1)/(i - 1);
        H = H + [A B]'*[A B];
        f = -[A B]'*err;
        lb_qp = -inf(size(states,1) + size(inputs,1), 1);
        ub_qp = inf(size(states,1) + size(inputs,1), 1);
        lb_qp(end - size(inputs,1) + 1:end) = max(-u_step, lb - inputs(:, i -1));
        ub_qp(end - size(inputs,1) + 1:end) = min(u_step, ub - inputs(:,i-1));
        options = optimoptions('quadprog','Display','none');
        z = quadprog(H, f, [], [], [], [], lb_qp, ub_qp, [], options);
        err = z(1:size(states,1));
        inputs(:,i-1) = inputs(:,i-1) + z(end-size(inputs,1) + 1:end);
    end
    plot_u(1) = err(1);
    plot_v(1) = err(2);
    rem_err = norm(err);
end


end