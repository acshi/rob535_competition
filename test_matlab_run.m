addpath('./SimPkg_F18_V2');
addpath('./rob535submission2');

load 'TestTrack.mat';
Xobs = generateRandomObstacles(10);
tic;
U_short = ROB599_ControlsProject2_part2_Team5(TestTrack, Xobs);
toc
U = U_short;
% U = repelem(U_short, 5, 1);
[Y, ~] = forwardIntegrateControlInput2(U);
info = getTrajectoryInfo(Y, U, Xobs, TestTrack);
close all;
plot(Y(:, 1), Y(:, 3));
hold on;
plot(TestTrack.bl(1, :), TestTrack.bl(2, :), 'b');
plot(TestTrack.br(1, :), TestTrack.br(2, :), 'b');
plot(TestTrack.cline(1, :), TestTrack.cline(2, :), '.');

% [Y, ~] = forwardIntegrateControlInput(repelem(us, 5, 1), x0);
% info = getTrajectoryInfo(Y, repelem(us, 5, 1), Xobs, TestTrack);