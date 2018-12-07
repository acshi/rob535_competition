function X = fmincon_fh(x_ic, dt, TestTrack, XObs, obs)
horizon = 10;
X = zeros(1,8);
% figure;
% hold on;
X_ic = potential_fields_euler_cons(x_ic, horizon, TestTrack, XObs);
obj_fun =  @(x) lookahead_dist(x, TestTrack);

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
%     
%     plot(X_ic(:,1), X_ic(:,3), '.');
%     plot(X_fh(1:8:end), X_fh(3:8:end), '.');
%     plot(TestTrack.bl(1,:), TestTrack.bl(2,:));
%     plot(TestTrack.br(1,:), TestTrack.br(2,:));
end
end