function X = fmincon_fh(x_ic, dt, TestTrack, XObs, obs)
horizon = 75;
X = zeros(1,8);
% figure;
% hold on;
X_ic = potential_fields_euler_cons(x_ic, horizon, TestTrack, XObs);
obj_fun =  @(x) fmincon_fh_obj(x, TestTrack, obs);

figure;
hold on;
plot(TestTrack.bl(1,:), TestTrack.bl(2,:));
plot(TestTrack.br(1,:), TestTrack.br(2,:));
plot_x = X_ic(:,1);
plot_y = X_ic(:,3);
h = plot(plot_x, plot_y, '.');
h.XDataSource = 'plot_x';
h.YDataSource = 'plot_y';

for i=1:1000
    disp(i);
    X_fh = fmincon_full_state(X_ic, dt, x_ic, TestTrack, obs, obj_fun);
    X(i,:) = X_fh(1:8);
    x_ic = X_fh(9:14)';

    X_ic = zeros(horizon + 1, 8);
    X_ic(1:horizon,:) = reshape(X_fh(9:end), 8, [])';
    end_controls = X_ic(horizon - 1,7:8);
    X_ic(horizon,7:8) = end_controls;
    X_ic(horizon + 1,7:8) = end_controls;
    X_ic(horizon + 1,1:6) = X_ic(horizon,1:6) + dt*nonl_bike(X_ic(horizon,1:8))';
  
%     plot(X_ic(:,1), X_ic(:,3), '.');
%     plot(X_fh(1:8:end), X_fh(3:8:end), '.');
%     plot(TestTrack.bl(1,:), TestTrack.bl(2,:));
%     plot(TestTrack.br(1,:), TestTrack.br(2,:));
    plot_x = X_fh(1:8:end);
    plot_y = X_fh(3:8:end);
    refreshdata(h, 'caller');
    drawnow;
    
    pos = [X_fh(end-7); X_fh(end - 5)];
    dist = norm(pos - TestTrack.cline(:, end));
    if dist < 5
        for j = 1:horizon
            X(i+j,:) = X_fh(8*j + 1:8*j + 8);
        end
        break;
    end
end
end