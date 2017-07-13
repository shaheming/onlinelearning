function OMG_DELAY()
import MinHeap
%use doubling tricking to iterate
M = 15; % 2 ^ 15 = 32768
% the maxiums turn will iterate T times;
T = 2^(M)-1; % avoid the last value to 0
% T = 50000;
% G is  positive retional number begin with 1
N = 100; % N is used to set G and Z
global G;
G = ones(1,N);
global Z;
Z = zeros(N-1,1);
% your decision domain used in projection
global x_bound;
x_bound = [0,1000];
% D and eta are used in reward function U
global D;
D = 1;
global eta;
eta = 1;
global gzs;
gzs  = zeros(1,T+1); % <G , Z>
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
y = 8;

global feedbackHeap;
feedbackHeap = MinHeap(T);
%%%%%%%%%%%%%%%%%%
% main function  %
%%%%%%%%%%%%%%%%%%
out=doubling(M);
regrets = zeros(1,T);
out1 = OGD_Primary(T);

figure('name','Regrets','NumberTitle','off','Position',[0,500,700,500]);
plot(out,'DisplayName','doubling');
hold on;
plot(out1,'DisplayName','omd');
hold off;
%%%%%%%end%%%%%%%%


end

function out = doubling(M)
  global regrets_div_t;
  global experts;
  global myChoices;
  global regrets;
  global myRewards;
  global expertsRewards;
  rng(1);
  %rng('shuffle');
  for m = 1 : M
%    [myChoices(m),experts(m), regrets(m)]=
    iteration(2^(m-1),2^(m)-1,true);
%    regrets_div_t(m) = regrets(m)/2^m;
  end
  
   rng(1);
% regret_s=ogddoublingtrick(M-1);
  
  figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
  plot(experts,'DisplayName','experts');
  hold on;
  plot(myChoices,'DisplayName','mychoice');
  legend('experts','mychoice');
  hold off;
  figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
  hold on;
%   plot(regret_s'+ones(1,size(regret_s,1))*89);
  plot(regrets);
  hold off;

%   diff(1:end) = regrets -(regret_s'-ones(1,size(regret_s,1))*89);
  figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
  plot(regrets_div_t);
  out = regrets;
end

function out = OGD_Primary(T)
  global regrets_div_t;
  global experts;
  global myChoices;
  global regrets;
  
   
  disp('Begin Loop');
  fprintf('Iterate %d turns',T);
  iteration(1,T,false);
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
  out = regrets;
end


function iteration(t_b,t_e,doubling_flag)
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
  global feedbackHeap;
  
  u = 0;
  eta1 = 0;
  feedbackCount = 0;
  
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(1) = G(2:end) * Z;
    x_t = project(y,x_bound);
  end

%   disp([t_b,t_e]);
  for t = t_b : t_e 
    %Z(1:end) = D * rand(size(Z,1),1);
    % gzs(t) = G(2:end) * Z;
    % my choice
    if doubling_flag 
      eta1 = t_b+1; 
    else
      eta1 = t+1;
    end
    x_t = project(y,x_bound);
    y = y + (1 / eta1)*(gradient(x_t,gzs(t),eta,G));
    
    myChoices(t) = x_t;
    % my rewards
    if t ~=1
      myRewards(t)  = myRewards(t-1) + Ut(x_t,gzs(t),eta,G);
    else
      myRewards(t)  = Ut(x_t,gzs(t),eta,G);
    end

    
    % caculate expert choice
    if t ~= 1
      u = t/(t+1) * experts(t - 1) + 1/(t+1)* 1 /G(1) * (gzs(t) + eta);
    else
      u =  (gzs(t) + eta) / G(1);
    end
    
    u = project(u,x_bound);
    experts(t) = u;
    
    % expert rewards
    expertsRewards(t) = Ut_expert(experts(t),gzs,eta,t,G);
    
    regrets(t) = myRewards(t) - expertsRewards(t);
    regrets_div_t(t) = regrets(t) / t;
    %%%
    
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(t+1) = G(2:end) * Z;
  end
  
  
end


% the difference of reward function U
function uout = gradient(x_t,gz,eta,G)
  uout = - G(1)*(G(1)*x_t - gz - eta);  
end
% the reward function U
function uout = Ut(x_t,gz,eta,G)
  uout = -0.5 * (G(1).*x_t - gz - eta).^2;
end

function uout = Ut_expert(u,gzs,eta,t,G)
  uout = -0.5 * (t * ((G(1)* u - eta)^2 )+sum(-2*(G(1)*u -eta) * gzs(1:t) + gzs(1:t).^2));
end
%the projection funciton
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


%%%%%%%
%     %
%%%%%%%

function [feedbackTime,gz] = boundDelay(t,B)
  global G;
  global Z;
  Z(1:end) = D * rand(size(Z,1),1);
  feedbackTime = t + rand(1) * B;
  gz =  G(2:end) * Z;
end

function [feedbackTime,gz] = linearDelay(t)
  global G;
  global Z;
  Z(1:end) = D * rand(size(Z,1),1);
  feedbackTime = t * 2;
  gz =  G(2:end) * Z;
end

function [feedbackTime,gz] = logDelay(t)
  global G;
  global Z;
  Z(1:end) = D * rand(size(Z,1),1);
  feedbackTime = t + ceil(log2(t));
  gz =  G(2:end) * Z;
end

function [feedbackTime,gz] = squareDelay(t)
  global G;
  global Z;
  Z(1:end) = D * rand(size(Z,1),1);
  feedbackTime = t^T;
  gz =  G(2:end) * Z;
end
