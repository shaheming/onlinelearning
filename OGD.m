G = (rand(100,1) + 0.1)' ;
% G is  positive retional number begin with  1
R = 10;
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

%
y = 50.5;
disp('Begin Loop');
for i = 0 : 999
    Z = D * rand(99,1);
    x_t = project(y,Xb,Xe);
    y = y + (1 / (i+1))*((Ut_z(x_t,Z,G,eta)));
    outx = [outx,x_t];
    outy = [outy,y];
    outr = [outr,Ut(x_t,Z,G,eta) - Ut(50,Z,G,eta)];
end
disp('End Loop');

figure('name','The output of Xt','NumberTitle','off','Position',[0,500,700,500]);
plot(outx);
figure('name','The output of regret','NumberTitle','off','Position',[700,500,700,500]);
plot(outr);


function [u] = Ut_z(x_t,Z,G,eta)
gz = G(2:end) * Z;
u = -G(1)*(G(1)*x_t - gz - eta);
end

function [u] = Ut(x_t,Z,G,eta)
gz = G(2:end) * Z;
u = -0.5 * (G(1)*x_t - gz - eta)^2;
end

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


