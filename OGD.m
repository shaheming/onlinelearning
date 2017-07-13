%use doubling tricking to iterate
M = 12; % 2 ^ 15 = 32768
% the maxiums turn will iterate T times;
T = 2^(M) - 1;
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
global diff;
diff = zeros(1,T);
% the initial y
global y;
y = 0.5;

% SET SEEDS


%%%%%%%%%%%%%%%%%%
% main function  %
%%%%%%%%%%%%%%%%%%
doubling(M);
%OGD_Primary(T);
%%%%%%%end%%%%%%%%


function doubling(M)
  global regrets_div_t;
  global experts;
  global myChoices;
  global regrets;
  global diff;
%   experts=zeros(1,M);
%   myChoices=zeros(1,M);
%   regrets=zeros(1,M);
%   regrets_div_t=zeros(1,M);
  rng(1);
  for m = 1 : M
%    [myChoices(m),experts(m), regrets(m)]=
    iteration(2^(m-1),2^(m)-1,true);
%    regrets_div_t(m) = regrets(m)/2^m;
  end
  
   rng(1);
  regret_s=ogddoublingtrick(M-1);
  
  figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
  plot(experts,'DisplayName','experts');
  hold on;
  plot(myChoices,'DisplayName','mychoice');
  legend('experts','mychoice');
  hold off;
  figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
  plot(regrets);
  hold on;
%   plot(regret_s'+ones(1,size(regret_s,1))*89);
  plot(regret_s);
  hold off;

%   diff(1:end) = regrets -(regret_s'-ones(1,size(regret_s,1))*89);
  figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
  plot(regrets_div_t);
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
  uout = -0.5 * (t * ((G(1)* u - eta)^2 )+sum(-2*(G(1)*u -eta) * gzs(1:t) + gzs(1:t).^2));
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
  u = 0;
  eta1 = 0;
%   slot_end = t_e - t_b + 1;

  for t = t_b : t_e
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
    if doubling_flag 
      eta1 = t_b ;
    else
      eta1 = t+1;
    end
    
    y = y + (1 / eta1)*(gradient(x_t,gzs(t),eta,G));
    
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
end



function regret=ogddoublingtrick(period)
%x belongs to [0,1000]
y0=0.5;
eta=1;
n=100;
round=2^(period+1)-1;
s=zeros(round,1);
%calculate y(1)
z=zeros(n-1,1);
for i=2:n
    z(i)=rand(1);
end
x0=project(y0);
sum0=sum(z);
y(1)=y0-(x0-sum0-eta);

% regret
x=zeros(round,1);
y=zeros(round,1);
u=zeros(round,1);
user=zeros(round,1);
expert_reward=zeros(round,1);
users_reward=zeros(round,1);
regret=zeros(round,1);

for m=[0:period]
    for t=(2^m):(2^(m+1)-1)
         %user's choice
        for i=2:n
            z(i)=rand(1);
        end
        x(t)=project(y(t));
        s(t)=sum(z);
        y(t+1)=y(t)-(x(t)-s(t)-eta)/((2^m)+1);
        %user performance
        users_reward(t)=-0.5*(x(t)-s(t)-eta)^2; 
        if t==1
            user(t)=users_reward(t);
        else
            user(t)=user(t-1)+users_reward(t);
        end
        %best expert performance
        if t==1
            u(t)=s(t)+eta;
        else
%           u(t)=u(t-1)*t/(t+1)+(s(t)+eta)/(t+1); 
            u(t)=u(t-1)*(t-1)/t+(s(t)+eta)/t; 
        end
        for j=1:t
            expert_reward(t)=expert_reward(t)-0.5*(u(t)-s(j)-eta)^2;
        end
        %calculate regret
        regret(t)=user(t)-expert_reward(t);
    end
end

function x=project(y)
if y>=0 && y<=1000
    x=y;
elseif y>1000
    x=1000;
else
    x=0;
end
end

end

   
