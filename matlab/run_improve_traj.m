load 65s08.mat
load TestTrack.mat

obs = obs_seg(TestTrack, []);
tic;
X_imp = improve_traj(X, TestTrack, 0.1, obs);
toc
