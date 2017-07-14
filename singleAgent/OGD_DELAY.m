function OMG_DELAY(type)
import MinHeap
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

global delaytimes;
delaytimes = zeros(T,1);

global feedbackHeap;
feedbackHeap = MinHeap(T+1,ones(1,4)* inf);
feedbackHeap.ExtractMin();
global feedBackCount;
feedBackCount = 0;
%%%%%%%%%%%%%%%%%%
% main function  %
%%%%%%%%%%%%%%%%%%
% out=doubling(M);
% regrets = zeros(1,T);


% Delay bound
% if the B == 1 there is no bound
global B;
B = 1;
type = 'bound';  
% bound
% linear
% log
% square
rng(1);
OGD_Primary(T,type);
%types = {'square','bound','linear','log','square'};
% for i = types
% OGD_DELAY(char(i))
% end

% figure('name','Regrets','NumberTitle','off','Position',[0,500,700,500]);
% plot(out,'DisplayName','doubling');
% hold on;
% plot(out1,'DisplayName','omd');
% hold off;
%%%%%%%end%%%%%%%%


end



function out = OGD_Primary(T,type)
%   global regrets_div_t;
  global experts;
  global myChoices;
  global regrets;
  

  disp('Begin Loop');
  fprintf('Iterate %d turns',T);
  
  [myRewards,expertsRewards]= iteration(0,T,false,type);
  
  disp('End Loop');
  imgXCompare = figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
  plot(experts,'DisplayName','experts');
  hold on;
  plot(myChoices,'DisplayName','mychoice');
  legend('experts','mychoice');
  title('My x and expert '' u','FontSize',20,'FontWeight','normal');
  hold off;
  
  saveas(imgXCompare,strcat('img/',type,'_xcompare'),'png');
  
  imgRegret = figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
  plot(regrets);
  title('Regret','FontSize',20,'FontWeight','normal');
  saveas(imgRegret,strcat('img/',type,'_regret'),'png');
  
%   figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
%   plot(regrets_div_t);

  imgRewardCompare = figure('name','ExpertsRewards and myRewards','NumberTitle','off','Position',[100,500,700,500]);

  plot(myRewards,'DisplayName','myRewards');
  hold on;
  title('Rewards Compare','FontSize',20,'FontWeight','normal');
  plot(expertsRewards,'DisplayName','expertsRewards');
  legend('myRewards','expertsRewards');
  hold off;
  saveas(imgRewardCompare,strcat('img/','type','_reward'),'png');
  
  out = regrets;
  close all;
end


function[outMyRewards,outExpertsRewards]=iteration(t_b,t_e,doubling_flag,type)
  global G;
  global D;
  global Z;
  global B;
  global y;
  global eta;
  global gzs;  
  global x_bound;
  global regrets;
  global regrets_div_t;
  global experts;
  global myChoices;
  global myRewards;
  global expertsRewards;
  global feedbackHeap;
  global delaytimes;
  global feedBackCount;

  u = 10;
  x_t = 0;
%   gz = 0;

  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 0
%   Z(1:end) = D * rand(size(Z,1),1);
%   gzs(1)=G(2:end) * Z;
    x_t = project(y,x_bound);
%   y = y + (gradients(x_t,gzs(1),eta,G));
    
   %gDelayedFeedBack(B,D,G,Z,0,x_t,eta,feedbackHeap,type);
  end

  for t = t_b : t_e 
    isFeedBack = false;
    % generate feedback delay
    Z(1:end) = D * rand(size(Z,1),1);
    gDelayedFeedBack(B,D,G,Z,t,x_t,eta,feedbackHeap,type);
    
    if doubling_flag 
      eta1 = t_b+1; 
    else
      eta1 = t+1;
    end
    
    if feedbackHeap.Count() > 0 
  
      out = num2cell(feedbackHeap.ReturnMin());
      [feedBackTime,gz,gradient,reward] = out{:};
      if feedBackTime - 1  == t
        isFeedBack = true;
         tmp = 0;
        while feedbackHeap.Count() > 0  
           out = num2cell(feedbackHeap.ReturnMin());
           [feedBackTime,gz,gradient,reward] = out{:};
           
           if feedBackTime - 1 > t
             break;
           else
             feedBackCount = feedBackCount + 1;
             tmp = tmp +1;
              out = num2cell(feedbackHeap.ExtractMin());
             [feedBackTime,gz,gradient,reward] = out{:};   
             
             myRewards(t) = myRewards(t) + reward;% x_s rewards
             % update y
             y = y + (1 / eta1) * gradient;
             % update x
             x_t = project(y,x_bound);
             gzs(feedBackCount) = gz;
             
             % update u
             if feedBackCount ~= 1
               % the t-1 reprent the u(t-1) when there is no feedback u(t) will
               % not be changed
               u = feedBackCount/(feedBackCount+1) * experts(t - 1) + 1/(feedBackCount+1)* 1 /G(1) * (gzs(feedBackCount) + eta);
             else
               u =  (gzs(1) + eta) / G(1);
             end
             u = project(u,x_bound);
           end
           delaytimes(t) = tmp;
        end
        
        %x_t = project(y,x_bound);
        %y = y + (1 / eta1)*(gradients(x_t,gzs(nonzeroIndex:tmp - 1 ),eta,G));
        myChoices(t) = x_t;
        experts(t) = u;
         % caculate expert choice
         % the feedBackCount = |Fs|

        expertsRewards(t) = Ut_expert(experts(t),gzs,eta,t,G);
      end
    end
     
%     if ~isFeedBack
%       if t ==1
%         myRewards(t)  = 0;
%         expertsRewards(t) = 0;          
% 
%       else
%          myRewards(t)  = myRewards(t-1);
%         
%       end
% 
%     end
   
     if t > 1
       myRewards(t) =  myRewards(t-1);
       expertsRewards(t) = expertsRewards(t-1);    % expert rewards
     elseif  ~isFeedBack
        % is feedback the rewards will be update
        myRewards(t)  = 0;
        expertsRewards(t) = 0;      
     end
  
    myChoices(t) = x_t;
    experts(t) = u;
         
    regrets(t) = myRewards(t) - expertsRewards(t);
    regrets_div_t(t) = regrets(t) / t;

  end
  outMyRewards = myRewards;
  outExpertsRewards = expertsRewards;
%   delayCompare=[myRewards',expertsRewards',delaytimes];

end


% the difference of reward function U
function uout = gradients(x_t,gzs,eta,G)
  uout = sum(- G(1)*((G(1) * x_t- eta) * ones(size(gzs , 1)) - gzs ));  
end
% the reward function U
function uout = Ut(x_t,gzs,eta,G)
  uout = sum(-0.5 .* ((G(1)*x_t - eta ) * ones(size(gzs , 1)) - gzs).^2);
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


%%%%%%%%%%%%%%%%%%%%%
%  delay function   %
%%%%%%%%%%%%%%%%%%%%%

function [feedBackTime,gz] = boundDelay(t,B,D,G,Z)  
  Z(1:end) = D * rand(size(Z,1),1);
  feedBackTime = randi([1,B])+t;
  gz =  G(2:end) * Z;
end

function [feedBackTime,gz] = linearDelay(t,D,slop,G,Z)
  Z(1:end) = D * rand(size(Z,1),1);
  if t == 0
    t = randi([1,size(Z,1)]);
  end
  feedBackTime = t * slop;
  gz =  G(2:end) * Z;
end

function [feedBackTime,gz] = logDelay(t,D,G,Z)
  Z(1:end) = D * rand(size(Z,1),1);
  if t < 2
    t = randi([1,size(Z,1)]);
  end
  feedBackTime = t + ceil(t * log2(t));
  gz =  G(2:end) * Z;
end

function [feedBackTime,gz] = squareDelay(t,D,G,Z)
  Z(1:end) = D * rand(size(Z,1),1);
  if t == 0
    t = randi([1,size(Z,1)]);
  end
  feedBackTime = t^2 + t;
  gz =  G(2:end) * Z;
end

function gDelayedFeedBack(B,D,G,Z,t,x_t,eta,feedbackHeap,type)
 feedBackTime = 0;
 gz =0;
  switch lower(type)
    case 'bound'
      [feedBackTime,gz] = boundDelay(t,B,D,G,Z);  
    case 'linear'
      [feedBackTime,gz] =  linearDelay(t,D,2,G,Z);
    case 'log'
      [feedBackTime,gz] = logDelay(t,D,G,Z);
    case 'square'
      [feedBackTime,gz] = squareDelay(t,D,G,Z);
    otherwise
       error('Delay type err');
  end
   
    gradient = gradients(x_t,gz,eta,G);
    reward = Ut(x_t,gz,eta,G);
    feedbackHeap.InsertKey([feedBackTime,gz,gradient,reward]);
end

function out = doubling(M)
  global regrets_div_t;
  global experts;
  global myChoices;
  global regrets;

  %rng('shuffle');
   rng(1);
  for m = 1 : M
%    [myChoices(m),experts(m), regrets(m)]=
    iteration(2^(m-1),2^(m)-1,true);
%    regrets_div_t(m) = regrets(m)/2^m;
  end
  
% regret_s=ogddoublingtrick(M-1);
  
  figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
  plot(experts,'DisplayName','experts');
  hold on;
  plot(myChoices,'DisplayName','mychoice');
  legend('experts','mychoice');
  hold off;
  figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
  hold on;
% plot(regret_s'+ones(1,size(regret_s,1))*89);
  plot(regrets);
  hold off;

% diff(1:end) = regrets -(regret_s'-ones(1,size(regret_s,1))*89);
  figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
  plot(regrets_div_t);
  out = regrets;
end