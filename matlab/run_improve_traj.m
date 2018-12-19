load 65s41.mat
load TestTrack.mat

obs = obs_seg(TestTrack, []);
tic;
X2 = improve_traj(X, TestTrack, 0.1, obs);
toc
