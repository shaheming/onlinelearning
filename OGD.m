function out = OMG()
%use doubling tricking to iterate
M = 11; % 2 ^ 15 = 32768
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
feedbackHeap = MinHeap(T+1,ones(1,4)* inf);
%%%%%%%%%%%%%%%%%%
% main function  %
%%%%%%%%%%%%%%%%%%
% out=doubling(M);
% regrets = zeros(1,T);
  rng(1);
  out_s = OGD_Primary(T);
  rng(1);
  regret_s=ogdfix(8,x_bound);
  figure('name','RG','NumberTitle','off','Position',[0,500,700,500]);
  plot(out_s,'DisplayName','out_s');
  hold on;
  plot(regret_s,'DisplayName','regret_s');
  legend('out_s','regret_s');
  hold off;
%  figure('name','Regrets','NumberTitle','off','Position',[100,0,700,500]);
%  plot(out,'DisplayName','doubling');
%  hold on;
%  plot(out_1,'DisplayName','omd');
%  legend('doubling','omd');
%  hold off;
%%%%%%%end%%%%%%%%


end

function out = doubling(M)
  global regrets_div_t;
  global experts;
  global myChoices;
  global regrets;
  global myRewards;
  global expertsRewards;
  global gzs;
  
  T = 2^(M)-1; % avoid the last value to 0
  gzs  = zeros(1,T+1); % <G , Z>
  %output variable
  regrets = zeros(1,T);
  regrets_div_t = zeros(1,T);
  experts = zeros(1,T);
  expertsRewards = zeros(1,T);
  myChoices = zeros(1,T);
  myRewards = zeros(1,T);
  

  %rng('shuffle');
  for m = 1 : M
%    [myChoices(m),experts(m), regrets(m)]=
    iteration(2^(m-1),2^(m)-1,true);
%    regrets_div_t(m) = regrets(m)/2^m;
  end

  figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
  plot(experts,'DisplayName','experts');
  hold on;
  plot(myChoices,'DisplayName','mychoice');
  legend('experts','mychoice');
  hold off;
  figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
  hold on;
%    plot(regret_s'+ones(1,size(regret_s,1))*89);
  plot(regret_s);
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
  global myRewards;
  global expertsRewards;
  global gzs;
  gzs  = zeros(1,T+1); % <G , Z>
  %output variable
  regrets = zeros(1,T);
  regrets_div_t = zeros(1,T);
  experts = zeros(1,T);
  expertsRewards = zeros(1,T);
  myChoices = zeros(1,T);
  myRewards = zeros(1,T);
   
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



  
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1
    % I think there is problem
    % If we choose iterate from 1
    % The X_1 is decided by which parameter
    % This simulation is a little bit different from the normal algorithm
    % in the paper. 
    % In my code we get x at first, then we gerate y
    
    % get x1
    myChoices(1) = project(y,x_bound);

    % get feedback
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(1) = G(2:end) * Z;
    
    % find the best expert's choice
    u =  (gzs(1) + eta) / G(1);
    u = project(u,x_bound);
    experts(1) = u;
    
    expertsRewards(1) = expertReward(experts(1),gzs,eta,1,G);
%   expertsRewards(1) = -0.5*(u-gzs(1)-eta)^2;
    myRewards(1)  = userLoss(myChoices(1),gzs(1),eta,G);
    regrets(1) = myRewards(1) - expertsRewards(1);
    regrets_div_t(1) = regrets(1);
    
    % get y + 1
    y = y + (1 / 2)*(gradient(myChoices(1),gzs(1),eta,G));
    experts(1) = u;
  end

   
 %disp([t_b,t_e]);
  for t = t_b + 1 : t_e
    %Z(1:end) = D * rand(size(Z,1),1);
    % gzs(t) = G(2:end) * Z;
    % my choice

    if doubling_flag 
      eta1 = t_b+1; 
    else
      eta1 = t+1;
    end
    
    % get x_t
    myChoices(t) = project(y,x_bound);
  
    % feedback
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(t) = G(2:end) * Z;
    
    % get y t + 1
    y = y + (1 / eta1)*(gradient(myChoices(t) ,gzs(t),eta,G));
    
    % my rewards
    myRewards(t)  = myRewards(t-1) + userLoss(myChoices(t),gzs(t),eta,G);
   
    % caculate expert choice
    u = (t-1)/ t* experts(t - 1) + 1/t * 1 /G(1) * (gzs(t) + eta);
    u = project(u,x_bound);
    
    % record expert's choice
    experts(t) = u;
    %expert's rewards
    
    expertsRewards(t) = expertReward(experts(t),gzs,eta,t,G);
    
    % regret
    regrets(t) = myRewards(t) - expertsRewards(t);
    
    % test the big O property
    regrets_div_t(t) = regrets(t) / t;
    %%%
  end
  
  
end


% the difference of reward function U
function uout = gradient(x_t,gz,eta,G)
  uout = - G(1)*(G(1)*x_t - gz - eta);  
end
% the reward function U
function uout = userLoss(x_t,gz,eta,G)
  uout = -0.5 * (G(1).*x_t - gz - eta).^2;
end

function uout = expertReward(u,gzs,eta,t,G)
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
  y(1) = y1;
  x(1) = project(y1,x_bound);
  % feedback
  z(:)=rand(n-1,1);
  s(1)=sum(z);
  user(1) = -0.5*(x(1)-s(1)-eta)^2; 
  % feedback function
  y(2)= y(1) - 1/2 * (x(1)-sum(z)-eta);
  u(1)=s(1)+eta;
  expert_reward(1) = -0.5*(u(1)-s(1)-eta)^2;
  regret(1)=user(1)-expert_reward(1);

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





