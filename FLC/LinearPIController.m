%Clear workspace , command window and close all windows
clear
clc
close all 

%Tune and configure the linear PI Controller
%This represents our given system
Gp = zpk([], [-1, -9], 10);

%With the aid of the Control System Designer toolbox we tuned our PI
%controller in our stored session

%tuning process
%We choose a zero close to our system strong pole -1
%Gc = zpk(-1.3, 0, 1);
%controlSystemDesigner(Gp, Gc);

%from our session load the tuning parameters
load("ControlSystemDesignerSession.mat");

design_data = ControlSystemDesignerSession.DesignerData.Designs.Data;
Gc = design_data.C; %tuned PI Controller

sys_open_loop = Gp * Gc;

figure
rlocus(sys_open_loop);

sys_closed_loop = feedback(sys_open_loop, 1, -1);

figure
step(sys_closed_loop);

Kp = sys_closed_loop.K/10;
Ki = -sys_closed_loop.Z{1,1} * Kp;
