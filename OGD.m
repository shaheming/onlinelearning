% Set turns
T = 2000;
% G is  positive retional number begin with 1
N = 100; % N is used to set G and Z
global G;
G = ones(1,N);
global Z;
Z = zeros(N-1,1)
% your decision domain used in projection
global x_bound;
x_bound = [0,100];
% D and eta are used in reward function U
global D;
D = 1;
global eta;
eta = 1;
global gzs;
gzs  = zeros(1,T); % <G , Z>
% output variable
global regrets;
regrets = zeros(1,T);
global regrets_div_t;
regrets_div_t = zeros(1,T);
global experts;
experts = zeros(1,T);
global expertsRewards;
expertsRewards = zeros(1,T);
global myChoices;
myChoices = zeros(1,T);
global myRewards;
myRewards = zeros(1,T);

% the initial y
global y;
y = 0.5;

OGD_Primary(T);


function out = OGD_Primary(T)
  global regrets_div_t;
  global experts;
  global myChoices;
  global regrets;
  disp('Begin Loop');
  fprintf('Iterate %d turns',T);
  iteration(1,T);
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
end
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
function x_t = project(y_t,x_bound)
  if x_bound(1) <= y_t && y_t <= x_bound(2)
    x_t = y_t;
  else
    if y_t < x_bound(1)
      x_t = x_bound(1);
    else
      x_t = x_bound(2);
    end
  end
end

function [x_t,expert,regret] = iteration(t_b,t_e)
  global G;
  global Z;
  global x_bound;
  global D;
  global eta;
  global gzs;
  global regrets;
  global regrets_div_t;
  global experts;
  global expertsRewards;
  global myChoices;
  global myRewards;
  global y;
  u = 0
  slot_end = t_e - t_b + 1;

  for t = 1 : slot_end
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(t) = G(2:end) * Z;
    % caculate u
    if t == 1
      u =  (gzs(t) + eta) / G(1);
    else
      u = t/(t+1) * experts(t - 1) + 1/(t+1)* 1 /G(1) * (gzs(t) + eta);
    end
    u = project(u,x_bound);
    experts(t) = u;
    x_t = project(y,x_bound);
    y = y + (1 / (t+1))*(gradient(x_t,gzs(t),eta,G));
    myChoices(t) = x_t;
    myRewards(t)  = Ut(x_t,gzs(t),eta,G);
    %   myReward = Ut_new(x_t,gzs,eta,t,G);
    %   outy = [outy,myRewards(t+1) - mtn(Ut(experts,Z,G,eta))];
    expertsRewards(t) = Ut_new(experts(t),gzs,eta,t,G);
    regrets(t) = sum(myRewards(1:t)) - expertsRewards(t);
    regrets_div_t(t) = regrets(t) / t;
    %   if t >= T - 9
    %   disp([myReward,expertsRewards(t)])
    %   end
  end
  x_t =  myChoices(slot_end)
  expert = experts(slot_end)
  regret = regrets(slot_end)
end

