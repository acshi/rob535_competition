function force = boundary_force(pos, boundary)
force = [0;0];
for i=1:size(boundary, 2)-1
    force = force + electric_field(pos, boundary(:,i), boundary(:,i + 1));
end
end