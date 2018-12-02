function E = electric_field(pos, p0, p1)
L = norm(p1 - p0);
u_par = (p1 - p0)/L;
res = pos - p0;
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

E = (1/h)*(u_perp*(sin_a_1 + sin_a_2) + u_par*(-cos_a_1 + cos_a_2));
end