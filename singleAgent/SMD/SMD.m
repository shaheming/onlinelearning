function out = SMD()
%use doubling tricking to iterate
M = 6; % 2 ^ 15 = 32768
% the maxiums turn will iterate T times;
T = 2^(M)-1; % avoid the last value to 0
% T = 50000;

global r_bound;
r_bound = [0,1];
y0 = 2;
% D and eta are used in reward function U
global D;
D = 2*pi;



global eta;
eta = 1;

global thetas;
thetas  = zeros(1,T); % <G , Z>
% output variable
global regrets;
regrets = zeros(1,T);

global experts;
experts = zeros(1,T);
global expertsRewards;
expertsRewards = zeros(1,T);
global myChoices;
myChoices = zeros(1,T);
global myRewards;
myRewards = zeros(1,T);

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
  
%    rng(2);
   out = OGD_Primary(T,y0,isDraw);
%    plot(out,'DisplayName','omd');
%    legend('doubling','omd');
%    
%    hold off;
%   
%   rng(1);
%   regret_s=ogdfix(8,r_bound);
%   figure('name','RG','NumberTitle','off','Position',[0,500,700,500]);
%   plot(out,'DisplayName','out_s');
%   hold on;
%   plot(regret_s,'DisplayName','regret_s');
%   legend('out_s','regret_s');
%   hold off;
%%%%%%%end%%%%%%%%


end



function out = OGD_Primary(T,y0,isDraw)

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
  
   [~,regretsOut ]= iteration(1,T,y0,false);
  

   disp('End Loop');
  
   
   
   step = 100;
   theta = 0:2*pi/step:2*pi;
   r =0:1/step:1;
   [THETA,R]=meshgrid(theta,r);
   G = (3 + sin(5*THETA)+ cos(3*THETA)).*(R.^2).*(5/3-R);
   figure;
   hold on;
   mesh(R.*cos(THETA), R.*sin(THETA), G);
   hold on;
   shading interp
   global thetas;
   Z=zeros(1,T);
   for i = 1:T
     Z(i) = userLoss(myChoices(i),thetas(i));
   end
   Z = ones(1,T)*4;
   plot3(myChoices.*cos(thetas),myChoices.*sin(thetas),Z,'Color','r');
   hold off;
   figure;
   plot(myChoices);
  if isDraw
%     figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
%     plot(experts,'DisplayName','experts');
%     hold on;
%     plot(myChoices,'DisplayName','mychoice');
%     legend('experts','mychoice');
%     hold off;
%     figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
%     plot(regretsOut);
%     figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
%     plot(regrets_div_t);
  end
  out = myChoices';
end


function [rOut,regretsOut ]=iteration(t_b,t_e,y0,doubling_flag)

  global r_bound;
  global D;
  global thetas;
  global regrets;
  global experts;
  global expertsRewards;
  global myChoices;
  global myRewards;
  
  %theta

  
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
    r = project(y0,r_bound);
    %x0 feedback
    thetas(1) = rand(1) * D;    
    y = y0 - 1*gradient(r,thetas(1));
  end
   
  for t = t_b : t_e
    %Z(1:end) = D * rand(size(Z,1),1);
    % thetas(t) = G(2:end) * Z;
    % my choice

    if doubling_flag 
      eta1 = t_b+1; 
    else
      eta1 = 2*t + 1;
    end
    
    % get x_t
    myChoices(t) = project(y,r_bound);
    thetas(t) = rand(1) * D;
    % get theta_t + 1
    y = y - (1 / eta1)*gradient(myChoices(t),thetas(t));
    
    % my rewards
    if t == 1
       myRewards(1)  = userLoss(myChoices(1),thetas(1));
    else
       myRewards(t)  = myRewards(t-1) + userLoss(myChoices(t),thetas(t));
    end
    
    % caculate expert choice

    u = updateExpert(experts,t,thetas);
    u = project(u,r_bound);
    
    % record expert's choice
    experts(t) = u;
    %expert's rewards
    
    expertsRewards(t) = expertLoss(experts(t),thetas,t);
   
    %regret
    regrets(t) = myRewards(t) - expertsRewards(t);
    
  end
  
  rOut = r;
  regretsOut = regrets;
end

function u = updateExpert(experts,t,thetas)
  % note this with change with loss function
  u = 0;
end
% the difference of reward function U
function uout = gradient(r,theta)
  uout = (3+ sin(5*theta) + cos(3*theta))*(10/3 * r - 3 * r^2);
end

% the reward function U
function uout = userLoss(r,theta)
  uout = (3+ sin(5*theta) + cos(3*theta))* r^2 *(5/3-r);
end

function uout = expertLoss(r,thetas,t)
  uout = 0;
end
%the projection funciton
function x_t = project(y_t,r_bound)
  if r_bound(1) <= y_t && y_t <= r_bound(2)
    x_t = y_t;
  else
    if y_t < r_bound(1)
      x_t = r_bound(1);
    else
      x_t = r_bound(2);
    end
  end
end




function out = OGD_doubling(M,y1,isDraw)

  global experts;
  global myChoices;
%   global myRewards;
%   global expertsRewards;
%   global thetas;
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

