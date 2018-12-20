load 64s60.mat
load TestTrack.mat

obs = obs_seg(TestTrack, []);
tic;
X_imp = improve_traj(X, TestTrack, 0.1, obs,0.1,506);
toc
