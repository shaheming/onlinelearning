function LOGD(M)
  mkdir img;
  
  T = 2^(M)-1; % avoid the last value to 0
  N = 100; % N is used to set G and Z
  global G;
  G = ones(1,N);
  global Z;
  Z = zeros(N-1,1);
  % your decision domain used in projection
  global X_BOUND;
  X_BOUND = [0,1000];
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
  global experts;
  experts = zeros(1,T);
  global expertLosses;
  expertLosses = zeros(1,T);
  global myChoices;
  myChoices = zeros(1,T);
  global userLosses;
  userLosses = zeros(1,T);
  % the initial y
  
  y0 = 8;
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  isDraw =true;
  
  OGD_Primary(T,y0,isDraw);
  % Run doubling trick algorithm
  OGD_doubling(M,y0,isDraw);
  
end

function out = OGD_doubling(M,y0,isDraw)
  global experts;
  global myChoices;
  yout = y0;
  for m = 1 : M
    [yout,regretsOut] = iteration(2^(m-1),2^(m)-1,yout,true);
  end
  
  if isDraw
    fig = figure('name','X','NumberTitle','off','Position',[0,500,700,500]);
    plot(experts,'DisplayName','experts');
    hold on;
    plot(myChoices,'DisplayName','mychoice');
    legend('experts','mychoice');
    hold off;
    fig_1=figure('name','Regret',NumberTitle','off','Position',[700,500,700,500]);
    hold on;
    
    plot(regretsOut);
    hold off;
    print(fig,strcat('img/','LOGD-Doubling-Trick-X'),'-dpng','-r500');
    print(fig_1,strcat('img/','LOGD-Doubling-Trick-Regret'),'-dpng','-r500');
  end
  
  
  out = regretsOut;
end

function out = OGD_Primary(T,y0,isDraw)
  global experts;
  global myChoices;
  
  disp('Begin Loop');
  fprintf('Iterate %d turns',T);
  
  [~,regretsOut ]= iteration(1,T,y0,false);
  
  disp('End Loop');
  
  if isDraw
    fig = figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
    plot(experts,'DisplayName','experts');
    hold on;
    plot(myChoices,'DisplayName','mychoice');
    legend('experts','mychoice');
    hold off;
    fig_1=figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
    plot(regretsOut);
  end
  out = regretsOut;
  print(fig,strcat('img/','LOGD-X'),'-dpng','-r500');
  print(fig_1,strcat('img/','LOGD-Regret'),'-dpng','-r500');
end


function [yout,regretsOut]=iteration(t_b,t_e,y,doubling_flag)
  global G;
  global Z;
  global X_BOUND;
  global D;
  global gzs;
  global regrets;
  global experts;
  global expertLosses;
  global myChoices;
  global userLosses;
  
  
  if t_b == 1
    x0 = project(y,X_BOUND);
    % x0 feedback
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(1) = G(2:end) * Z;
    y = y - gradient(x0,gzs(1));
  end
  
  for t = t_b : t_e
    if doubling_flag
      eta1 = t_b+1;
    else
      eta1 = t + 1;
    end
    
    % get x_t
    myChoices(t) = project(y,X_BOUND);
    
    % feedback
    Z(1:end) = D * rand(size(Z,1),1);
    gzs(t) = G(2:end) * Z;
    
    % get y t + 1
    y = y - (1 / eta1)*gradient(myChoices(t),gzs(t));
    
    % my regrets
    if t == 1
      userLosses(1)  = userLoss(myChoices(1),gzs(1));
    else
      userLosses(t)  = userLosses(t-1) + userLoss(myChoices(t),gzs(t));
    end
    % caculate expert choice
    
    u = updateExpert(experts,t,t,gzs);
    u = project(u,X_BOUND);
    
    % record expert's choice
    experts(t) = u;
    %expert's regrets
    
    expertLosses(t) = expertLoss(experts(t),gzs,t);
    
    % regret
    regrets(t) = userLosses(t) - expertLosses(t);

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
function x_t = project(y_t,X_BOUND)
  if X_BOUND(1) <= y_t && y_t <= X_BOUND(2)
    x_t = y_t;
  else
    if y_t < X_BOUND(1)
      x_t = X_BOUND(1);
    else
      x_t = X_BOUND(2);
    end
  end
end







