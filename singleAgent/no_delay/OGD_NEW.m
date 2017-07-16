function out = OMG_NEW()
%use doubling tricking to iterate
M = 10; % 2 ^ 15 = 32768
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

y1 = 8;

%%%%%%%%%%%%%%%%%%
% main function  %
%%%%%%%%%%%%%%%%%%
  isDraw =true;
%   rng(1);
%   out_d=OGD_doubling(M,y1,isDraw);
%   
%   figure('name','Regrets Compare','NumberTitle','off','Position',[100,0,700,500]);
%   plot(out_d,'DisplayName','doubling');
%   hold on;
  
   rng(1);
   out = OGD_Primary(T,y1,isDraw);
%    plot(out,'DisplayName','omd');
%    legend('doubling','omd');
%    
%    hold off;
%   
%   rng(1);
%   regret_s=ogdfix(8,x_bound);
%   figure('name','RG','NumberTitle','off','Position',[0,500,700,500]);
%   plot(out,'DisplayName','out_s');
%   hold on;
%   plot(regret_s,'DisplayName','regret_s');
%   legend('out_s','regret_s');
%   hold off;
%%%%%%%end%%%%%%%%


end

function out = OGD_doubling(M,y1,isDraw)
  global regrets_div_t;
  global experts;
  global myChoices;
%   global myRewards;
%   global expertsRewards;
%   global gzs;
   yout = y1;
  %rng('shuffle');
  for m = 1 : M

    [yout,regretsOut] = iteration(2^(m-1),2^(m)-1,yout,true);
%    [myChoices(m),experts(m), regrets(m)]=
%    regrets_div_t(m) = regrets(m)/2^m;
  end
 
  if isDraw
    figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
    plot(experts,'DisplayName','experts');
    hold on;
    plot(myChoices,'DisplayName','mychoice');
    legend('experts','mychoice');
    hold off;
    figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
    hold on;
    %plot(regret_s'+ones(1,size(regret_s,1))*89);
    
    plot(regretsOut);
    hold off;
    
%     figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
%     plot(regrets_div_t);
  end
%   diff(1:end) = regrets -(regret_s'-ones(1,size(regret_s,1))*89);

  out = regretsOut;
end

function out = OGD_Primary(T,y1,isDraw)
  global regrets_div_t;
  global experts;
  global myChoices;

%   output variable
%   regrets = zeros(1,T);
%   regrets_div_t = zeros(1,T);
%   experts = zeros(1,T);
%   expertsRewards = zeros(1,T);
%   myChoices = zeros(1,T);
%   myRewards = zeros(1,T);
   
  disp('Begin Loop');
  fprintf('Iterate %d turns',T);
  
   [~,regretsOut ]= iteration(1,T,y1,false);
  
  disp('End Loop');
  
  if isDraw
    figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
    plot(experts,'DisplayName','experts');
    hold on;
    plot(myChoices,'DisplayName','mychoice');
    legend('experts','mychoice');
    hold off;
    figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
    plot(regretsOut);
%     figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
%     plot(regrets_div_t);
  end
  out = regretsOut;
end


function [yout,regretsOut ]=iteration(t_b,t_e,y,doubling_flag)
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
  


  
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1
    % I think there is problem
    % If we choose iterate from 1
    % The X_1 is decided by which parameter
    % This simulation is a little bit different from the normal algorithm
    % in the paper. 
    % In my code we get x at first, then we gerate y
    % get x0
    x_t = project(y,x_bound);
    % x0 feedback
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(1) = G(2:end) * Z;    
    y = y - gradient(x_t,gzs(1));
  end
   
  for t = t_b : t_e
    %Z(1:end) = D * rand(size(Z,1),1);
    % gzs(t) = G(2:end) * Z;
    % my choice

    if doubling_flag 
      eta1 = t_b+1; 
    else
      eta1 = t + 1;
    end
    
    % get x_t
    myChoices(t) = project(y,x_bound);
  
    % feedback
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(t) = G(2:end) * Z;
    
    % get y t + 1
    y = y - (1 / eta1)*gradient(myChoices(t),gzs(t));
    
    % my rewards
    if t == 1
       myRewards(1)  = userLoss(myChoices(1),gzs(1));
    else
       myRewards(t)  = myRewards(t-1) + userLoss(myChoices(t),gzs(t));
    end
    % caculate expert choice

    u = updateExpert(experts,t,t,gzs);
    u = project(u,x_bound);
    
    % record expert's choice
    experts(t) = u;
    %expert's rewards
    
    expertsRewards(t) = expertLoss(experts(t),gzs,t);
    
    % regret
    regrets(t) = myRewards(t) - expertsRewards(t);
    
    % test the big O property
    regrets_div_t(t) = regrets(t) / t;
    %%%
  end
  
  yout = y;
  regretsOut = regrets;
end

function u = updateExpert(experts,t,feedBackTimes,gzs)
  % note this with change with loss function
  if t ~= 1
    u = (feedBackTimes-1)/feedBackTimes * experts(t-1) + 1/feedBackTimes* gzs(feedBackTimes);
  else
    u =  gzs(1);
  end
end
% the difference of reward function U
function uout = gradient(x_t,gz)
  uout = x_t - gz;
end
% the reward function U
function uout = userLoss(x_t,gz)
  uout = 0.5 * (x_t  - gz)^2;
end

function uout = expertLoss(u,gzs,t)
  uout = 0.5 * (t * ( u^2 )+sum(-2* u * gzs(1:t) + gzs(1:t).^2));
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


function regret=ogdfix(y1,x_bound)
%x belongs to [0,1000]
% global myRewards;
% global myChoices;
%y0=8;
  T=2^11-1;
  eta=1;
  n=100;
  s=zeros(T+1,1);

  y=zeros(T,1);
  % this y has some problem!!!
  % I fix it
  %calculate y(1)
  expert_reward=zeros(T,1);
  users_reward=zeros(T,1);
  regret=zeros(T,1);
  u=zeros(T,1);
  user=zeros(T,1);
  z=zeros(n-1,1);
 
  x=zeros(T,1);
  
  % init 
  % we should say given y1,insteading given y0
  % get x0
  %  y(1) = y1;
  x = project(y1,x_bound);
  % feedback t = 0
  z(:)=rand(n-1,1);
  s =sum(z);
  % y = 1
  y(1)= y1 -  (x-sum(z)-eta);
  % feedback function
  % t = 1
  x(1) = project( y(1),x_bound);
  % feedback t = 1
  z(:)=rand(n-1,1);
  s(1) =sum(z);
  user(1) = -0.5*(x(1)-s(1)-eta)^2; 
  u(1)=s(1)+eta;
  expert_reward(1) = -0.5*(u(1)-s(1)-eta)^2;
  regret(1)=user(1)-expert_reward(1);
  % y 2 
   y(2)= y(1) - 1/2 * (x(1)-sum(z)-eta);
  for t=2:T
      %user's choice
      
      % get x and save user choice
      x(t)=project(y(t),x_bound);

      % get feedback
      z=rand(n-1,1);
      s(t)=sum(z);

      % update y
      y(t+1)=y(t)-(x(t)-s(t)-eta)/(t+1);
      %user performance
      users_reward(t)=-0.5*(x(t)-s(t)-eta)^2; 
 
      user(t)=user(t-1)+users_reward(t);
      %best expert performance
     

      u(t)=u(t-1)*(t-1)/t+(s(t)+eta)/t; 

      for j=1:t
          expert_reward(t)=expert_reward(t)-0.5*(u(t)-s(j)-eta)^2;
       end
      %calculate regret
      regret(t)=user(t)-expert_reward(t);
   end
end




