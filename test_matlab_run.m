addpath('./SimPkg_F18_V1');

load 'TestTrack.mat';
Xobs = generateRandomObstacles(10);
U = ROB599_ControlProject_part2_Team5(TestTrack, Xobs);
x0 = [287; 5; -176; 0; 2; 0];
[Y, ~] = forwardIntegrateControlInput(U, x0);
info = getTrajectoryInfo(Y, U, Xobs, TestTrack);
plot(Y(:, 1), Y(:, 3));