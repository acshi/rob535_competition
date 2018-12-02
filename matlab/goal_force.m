function force = goal_force(pos, cl)
dist = sum((pos - cl).^2,1);
[~,i] = min(dist);
i = min(i, size(dist,2) - 1);
force = cl(:,i + 1) - cl(:,i);
force = force/norm(force);
end