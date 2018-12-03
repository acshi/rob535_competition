function force = electric_force(pos, testTrack, XObs)
    force = [0;0];
    force = force + boundary_force(pos, testTrack.bl);
    force = force + boundary_force(pos, testTrack.br);
    force = force + goal_force(pos, testTrack.cline);
    if nargin == 3
        force = force + obstacle_force(pos, XObs);
    end
end