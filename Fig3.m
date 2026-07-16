% This script takes a point in the line attractor (variable n_target), a
% range of velocities, an eta, and returns p_n and t_n. It also takes the length of the track, 
% and duration of a lap to show results in terms of range distance and
% range duration.
% Employed to construct Figure 3.

%% range duration and range duration for a range of velocities, for a given 
% neuron in the attractor, with a given eta. Employed to construct Figure S1
clearvars
%close all
v_min=0;    %lowest velocity in the range (cm/seg)
v_max=28;   %highest velocity in the range (cm/seg)
v_pasos=100;%steps from lowest to highest velocity in the range
n_target=50;%variable n (the chosen neuron, a point in the line attractor)
eta=0.05;
l_total=150; %the length of the track (cm)
t_total=10;  %the duration of the lap (seg)

ancho=0.15;
alto=0.2;

graph_mode = 2; %1: plot as a function of ranges; 2: plot as a function of speed
[t_n, p_n, z_t_disparo, z_p_disparo, v, v_n]=compute_field_centres(n_target, eta, v_min, v_max, v_pasos);
switch graph_mode
    case 1
        In ranges
        r_duration = l_total./v;
        r_distance = t_total*v;
    case 2
        % velocity
        r_duration = v;
        r_distance = v;
end

fig1 = figure;
sp1=subplot(1,3,1);
plot(v, v_n)
xlabel('velocity (cm/s)')
ylabel("attractor's velocity (a.u.)")



sp2=subplot(1,3,2);
plot(p_n, r_duration)
xlabel('position (cm)')
switch graph_mode
    case 1
        ylabel('range duration (s)')
    case 2
        ylabel('velocity (cm/s)')
end
xlim([0, l_total])

sp3=subplot(1,3,3);
plot(t_n, r_distance)
xlabel('time (s)')
switch graph_mode
    case 1
        ylabel('range duration (s)')
    case 2
        ylabel('velocity (cm/s)')
end
xlim([0, t_total])

sp1.Position=[sp1.Position(1),sp1.Position(2), ancho, alto];
sp2.Position=[sp2.Position(1),sp2.Position(2), ancho, alto];
sp3.Position=[sp3.Position(1),sp3.Position(2), ancho, alto];


