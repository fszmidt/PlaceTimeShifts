function [t_disparo, p_disparo, z_t_disparo, z_p_disparo, v, v_n]=compute_field_centres(n_target, eta, v_min, v_max, paso)

f_aux=@(x)log(eta*x+1)-log(1);
f=@(x)(f_aux(x)-f_aux(0))/(f_aux(v_max)-f_aux(0))*v_max;


v=fliplr(linspace(v_min,v_max,paso));   %cm/seg
v_n = f(v);
t_disparo = n_target ./ f(v);
p_disparo = v * n_target ./ f(v);

media_t_disparo = mean(t_disparo);
desv_t_disparo = std(t_disparo);
media_p_disparo = mean(p_disparo);
desv_p_disparo = std(p_disparo);

z_t_disparo = media_t_disparo/desv_t_disparo;
z_p_disparo = media_p_disparo/desv_p_disparo;

