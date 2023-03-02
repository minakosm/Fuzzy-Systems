%% Clear Workspace, command window and close all windows
clc
clear 
close all 

%% Create Initial Fuzzy Variables and their Membership Functions
var_names = ["dv" "dh" "theta" "dtheta"];
mf_names = ["S" "M" "L" ; "N" "ZE" "P"];
boundaries = [0 0 0.5 1 1 ; 0 0 0.5 1 1 ; -180 -180 0 180 180 ; -130 -130 0 130 130];

carFis = mamfis;
carFis = addOutput(carFis, [boundaries(end,1) boundaries(end,5)], "Name", var_names(end));
for i=1:length(var_names)-1
    carFis = addInput(carFis, [boundaries(i,1), boundaries(i,5)], "Name", var_names(i));
end

for i=1:length(var_names)
    for j=1:length(mf_names(1,:))
        carFis = addMF(carFis, var_names(i), "trimf", [boundaries(i,j) boundaries(i,j+1) boundaries(i,j+2)], "Name", mf_names(ceil(i/2),j));
    end
end

%% Load RuleBase

%carFis = readfis("carFis_init.fis");
carFis = readfis("carFis_opt.fis");
%% Initialize enviroment 
x_bounds = [0 10.2];
y_bounds = [0 4.2];
obstacle = [5 0 ; 5 1 ; 6 1 ; 6 2 ; 7 2 ; 7 3 ; 10 3]; 
init_theta = [0 -45 -90];
u = 0.05;
goal = [10 3.2];
epsilon = 0.075;

%% Start Car's Obstacle Avoidance Controler
for i=1:3
    x = 4.1;
    y = 0.3;
    theta = init_theta(i);
    dtheta = [];
    while(1)
        [dh, dv] = getDistances(x(end),y(end));
        if(isnan(dh*dv))
            fprintf("\nHit Wall at (%f, %f) at step %d\n\n",x(end),y(end));
            break;
        end

        dtheta = evalfis(carFis, [dv, dh, theta(end)]);
        theta = [theta theta(end)+dtheta];
        ang = theta(end)*pi/180;
        x = [x x(end)+cos(ang)*u];
        y = [y y(end)+sin(ang)*u];

        reach_goal = pdist([goal ; [x(end) y(end)]]) < epsilon;

        out_of_bounds = (x(end) < x_bounds(1) || x(end)> x_bounds(2))... 
                        || (y(end) < y_bounds(1) || y(end) > y_bounds(2));

        if(reach_goal)
            fprintf("REACHED GOAL at (%d,%d)\n", x(end), y(end));
            break;
        elseif(out_of_bounds)
            fprintf("OUT OF BOUNDS\n");
            break;
        end
    end
    figure(i)
    hold on;
    plot(x,y, "r", "LineWidth",2);
    plot(goal(1), goal(2), "b*");
    a = area(obstacle(:,1),obstacle(:,2), 'DisplayName','Obstacles');
    set(a, 'FaceColor', [0.2 0.2 0.2]);
end
