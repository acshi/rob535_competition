function controls = ROB599_ControlsProject2_part2_Team5(TestTrack, Xobs)
    obs = cell2mat(Xobs');
    obs_x = obs(:, 1);
    obs_y = obs(:, 2);
    controls = rob535_competition_mex(TestTrack.bl', TestTrack.br', TestTrack.cline', TestTrack.theta, obs_x, obs_y);
%     controls = repelem(controls, 20, 1);
end