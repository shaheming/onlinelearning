% Set turns
T = 2000;
% G is  positive retional number begin with 1
N = 100; % N is used to set G and Z
G = ones(1,N);
% your decision domain used in projection
Xb = 0;
Xe = 100;
Ex_n = 1000;
% D and eta are used in reward function U
D = 1;
eta = 1;
gzs  = zeros(1,T); % <G , Z>
% output variable
myChoices = zeros(1,T);
outy = [];
regrets = zeros(1,T);
regrets_div_t = zeros(1,T);
experts = zeros(1,T);
expertsRewards = zeros(1,T);
myRewards = zeros(1,T);
myReward = 0;

% the initial y
y = 0.5;
u = 0;
disp('Begin Loop');
fprintf('Iterate %d turns',T);

for t = 1 : T  
    Z = D * rand(N - 1,1);
    gzs(t) = G(2:end) * Z;
    % caculate u
    if t == 1
      u =  (gzs(t) + eta) / G(1);
    else
      u = t/(t+1) * experts(t - 1) + 1/(t+1)* 1 /G(1) * (gzs(t) + eta);
    end
    u = project(u,Xb,Xe); %?
    experts(t) = u;
    x_t = project(y,Xb,Xe);
    y = y + (1 / (t+1))*(gradient(x_t,gzs(t),eta,G));
    myChoices(t) = x_t;
    myRewards(t)  = Ut(x_t,gzs(t),eta,G);
%   myReward = Ut_new(x_t,gzs,eta,t,G);
%   outy = [outy,myRewards(t+1) - mtn(Ut(experts,Z,G,eta))];
    expertsRewards(t) = Ut_new(experts(t),gzs,eta,t,G);
    regrets(t) = sum(myRewards(t)) - expertsRewards(t); 
    regrets_div_t(t) = regrets(t) / t; 
%     if t >= T - 9
%     disp([myReward,expertsRewards(t)])
%     end
end

disp('End Loop');

figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
plot(experts,'DisplayName','experts');
hold on;
plot(myChoices,'DisplayName','mychoice');
legend('experts','mychoice');
hold off;
figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
plot(regrets);
figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
plot(regrets_div_t);

% the difference of reward function U
function uout = gradient(x_t,gz,eta,G)
    uout = - G(1)*(G(1)*x_t - gz - eta);
end
% the reward function U
function uout = Ut(x_t,gz,eta,G)
    uout = -0.5 * (G(1).*x_t - gz - eta).^2;
end

function uout = Ut_new(u,gzs,eta,t,G)
  uout = -0.5 * (t * ((G(1)* u - eta)^2 + 2 * eta)+sum(-2*(G(1)*u -eta) * gzs(1:t) + gzs(1:t).^2));
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


