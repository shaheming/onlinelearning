% G is  positive retional number begin with  1
G = (rand(100,1) + 0.1)' ;
% your decision domain used in projection
Xb = 0;
Xe = 100;
% D and eta are used in reward function U
D = 1;
eta = 1;
% output variable
outx = [];
outy = [];
outr = [];

% the initial y
y = 50.5;
disp('Begin Loop');
for i = 0 : 999
    Z = D * rand(99,1);
    x_t = project(y,Xb,Xe);
    y = y + (1 / (i+1))*((Ut_z(x_t,Z,G,eta)));
    outx = [outx,x_t];
    outy = [outy,Ut(x_t,Z,G,eta) - Ut(30,Z,G,eta)];
    outr = [outr,sum(outr)+outy(i+1)];
end
disp('End Loop');

figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
plot(outx);
figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
plot(outr);
figure('name','The distance between my prediction and expert prediction in every turn','NumberTitle','off','Position',[700,1000,700,500]);
plot(outy);

% the difference of reward function U
function [u] = Ut_z(x_t,Z,G,eta)
gz = G(2:end) * Z;
u = - G(1)*(G(1)*x_t - gz - eta);
end
% the reward function U
function [u] = Ut(x_t,Z,G,eta)
gz = G(2:end) * Z;
u = -0.5 * (G(1)*x_t - gz - eta)^2;
end
% the projection funciton
function x_t = project(y_t,Xb,Xe)
if Xb <= y_t && y_t <= Xe
    x_t = y_t;
else
    if y_t < Xb
        x_t = Xb;
    else
        x_t = Xe;
    end
end
end


