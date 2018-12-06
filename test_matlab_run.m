addpath('./SimPkg_F18_V1');

load 'TestTrack.mat';
% Xobs = generateRandomObstacles(10);
% U = ROB599_ControlsProject_part2_Team5(TestTrack, Xobs);
x0 = [287; 5; -176; 0; 2; 0];
[Y, ~] = forwardIntegrateControlInput(U, x0);
info = getTrajectoryInfo(Y, U, Xobs, TestTrack);
close all;
plot(Y(:, 1), Y(:, 3));
hold on;
plot(TestTrack.bl(1, :), TestTrack.bl(2, :), 'b');
plot(TestTrack.br(1, :), TestTrack.br(2, :), 'b');
plot(TestTrack.cline(1, :), TestTrack.cline(2, :), '.');