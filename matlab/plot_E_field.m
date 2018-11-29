xs = -3:0.2:3;
ys = -3:0.2:3;

[p, q] = meshgrid(xs, ys);
coords = [p(:) q(:)];
fields = zeros(size(coords));


p0 = [0;0];
p1 = [1;1];

L = norm(p1 - p0);
u_par = p1 - p0;
u_par = u_par/norm(u_par);
for i = 1:size(coords,1)
    x = coords(i,1);
    y = coords(i, 2);
    if x == y
        continue
    end
    res = [x;y] - p0;
    a = res'*u_par;
    perp = res - a*u_par;
    h = norm(perp);
    u_perp = perp/h;
    b = L - a;
    hyp_a_1 = sqrt(a*a + h*h);
    sin_a_1 = a/hyp_a_1;
    cos_a_1 = h/hyp_a_1;
    
    hyp_a_2 = sqrt(b*b + h*h);
    sin_a_2 = b/hyp_a_2;
    cos_a_2 = h/hyp_a_2;
    
    fields(i,:) = (1/h)*(u_perp*(sin_a_1 + sin_a_2) + u_par*(-cos_a_1 + cos_a_2));
end

quiver(coords(:,1), coords(:,2), fields(:,1), fields(:,2));
    
    
    
    