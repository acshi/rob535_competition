function force = obstacle_force(pos, obstacles)
force = [0;0];
for i=1:numel(obstacles)
    obstacle = obstacles{i};
    force = force + boundary_force(pos, [obstacle; obstacle(1,:)]');
end