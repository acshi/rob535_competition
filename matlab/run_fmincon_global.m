load TestTrack.mat
load fmincon_fh_70s_horison2.mat

x_ic = [287,5,-176,0,2,0];
dt = 0.1;
tic;
obs = obs_seg(TestTrack, []);
obj_fun = @(x) sol_time(x, TestTrack, dt);
X = fmincon_full_state_global(X, dt, x_ic, TestTrack, obs, obj_fun);
fprintf('Opt took %f\n', toc);
figure;
hold on;
plot(X(:,1), X(:,3));
plot(TestTrack.bl(1,:), TestTrack.bl(2,:));
plot(TestTrack.br(1,:), TestTrack.br(2,:));

figure;
hold on;
plot(X(:,2));
plot(X(:,4));
plot(X(:,6)*10);
