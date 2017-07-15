
function  OGD_DELAY_NEW_ETA( M )
  %   M = 15;
  mkdir img OGD_DELAY_NEW_ETA;
  global img_path;
  img_path ='OGD_DELAY_NEW_ETA/';
  types = {'bound','linear','log','square'};
  regrets = {};
  for i = types
    rng(1);
    [outRegrets,outMyChoices] = OGD_DELAY_IN(char(i),M,false);
    regrets{end+1} = {outRegrets,outMyChoices,char(i)};
  end
  regFig = figure('name','Regrets');
  set(regFig,'position',get(0,'screensize'));
  
  for i = regrets
    plot(i{1}{1},'DisplayName',char(i{1}{3}));
    hold on;
  end
  lgd =legend(types);
  lgd.FontSize = 16;
  hold off;
  
  cFig = figure('name','Choices');
  set(cFig,'position',get(0,'screensize'));
  
  for i = regrets
    plot(i{1}{2},'DisplayName',char(i{1}{3}));
    hold on;
  end
  lgd =legend(types);
  lgd.FontSize = 16;
  hold off;
  
  saveas(regFig,strcat('img/',img_path,'regretsCompare'),'png');
  saveas(cFig,strcat('img/',img_path,'choicesCompare'),'png');
end



function [outRegrets,outMyChoices]= OGD_DELAY_IN(type,M,isDraw)
  import MinHeap
  %use doubling tricking to iterate
  % 2 ^ 15 = 32768
  % the maxiums turn will iterate T times;
  T = 2^(M)-1; % avoid the last value to 0
  % T = 50000;
  % your decision domain used in projection
  global x_bound;
  x_bound = [0,100];
  % D and eta are used in reward function U
  global D;
  D = 100;
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
  B = 5;
  % type = 'bound';
  y1 = 8;
  
  [outRegrets,outMyChoices] = OGD_Primary(T,y1,type,isDraw);
  %%%%%%%end%%%%%%%%
  
  
end



function [outRegrets,outMyChoices ]= OGD_Primary(T,y1,type,isDraw)
  % global regrets_div_t;
  global experts;
  global myChoices;
  
  if ~isDraw
    figConfig = 'off';
  else
    figConfig = 'on';
  end
  
  fprintf('Begin Loop with %s form delay\n',type);
  fprintf('Iterate %d turns\n',T);
  
  [myRewards,expertsRewards,outRegrets]= iteration(1,T,y1,false,type);
  
  
  fprintf('End Loop\n');
  headline = sprintf('LOGD %s Delay Choice',type);
  imgXCompare = figure('name',headline,'NumberTitle','off','Position',[0,500,700,500],'visible',figConfig);
  
  plot(experts,'DisplayName','experts');
  hold on;
  plot(myChoices,'DisplayName','mychoice');
  lg = legend('experts','mychoice');
  lg.FontSize = 16;
  title(headline,'FontSize',20,'FontWeight','normal');
  hold off;
  
  headline = sprintf('LOGD %s Delay Regret',type);
  imgRegret = figure('name',headline,'NumberTitle','off','Position',[700,500,700,500],'visible',figConfig);
  plot(outRegrets,'DisplayName','regrets');
  title(headline,'FontSize',20,'FontWeight','normal');
  
  
  %   figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
  %   plot(regrets_div_t);
  
  %   imgRewardCompare = figure('name','ExpertsRewards and myRewards','NumberTitle','off','Position',[100,500,700,500],'visible',figConfig);
  %   plot(myRewards,'DisplayName','myRewards');
  %   hold on;
  %   title('Rewards Compare','FontSize',20,'FontWeight','normal');
  %   plot(expertsRewards,'DisplayName','expertsRewards');
  %   legend('myRewards','expertsRewards');
  %   hold off;
  global img_path;
  saveas(imgXCompare,strcat('img/',img_path,type,'_xcompare'),'png');
  saveas(imgRegret,strcat('img/',img_path,type,'_regret'),'png');
  outMyChoices =myChoices;
end


function[outMyRewards,outExpertsRewards,outRegrets]=iteration(t_b,t_e,y1,doubling_flag,type)
  
  global B;
  %global y;
  global gzs;
  global x_bound;
  global regrets;
  global regrets_div_t;
  global experts;
  global myChoices;
  global myRewards;
  global expertsRewards;
  global feedbackHeap;
  global feedBackCount;
  global D;
  y = y1;
  lastUpdateTime = 0;
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1
    % t = 0
    z_t = project(y1,x_bound);
    gz =rand(1)*D;
    y = y - gradients(z_t,gz);
  end
  
  % from 1
  for t = t_b : t_e
    % generate delayed feedback
    %  generate feedback dela
    
    
    
    if t == 1
      myChoices(1) = project(y,x_bound);
      myRewards(t) = 0;
      expertsRewards(t)=0;
      experts(t) = 0;
      
    else
      myChoices(t) = project(y,x_bound);
      myRewards(t) = myRewards(t-1) ;
      expertsRewards(t)=expertsRewards(t-1);
      experts(t) = experts(t-1);
    end
    gDelayedFeedBack(B,t, myChoices(t),D,feedbackHeap,type);
    
    if feedbackHeap.Count()
      % check delay
      out = num2cell(feedbackHeap.ReturnMin());
      [feedBackTime,gz,gradient,reward] = out{:};
      if feedBackTime - 1  == t
        if doubling_flag
          eta1 = t_b+1;
        else
          eta1 = lastUpdateTime+1;
        end
        lastUpdateTime = t;
        
        
        % get all feedbacks
        while feedbackHeap.Count() > 0
          out = num2cell(feedbackHeap.ReturnMin());
          [feedBackTime,gz,gradient,reward] = out{:};
          
          if feedBackTime - 1 > t
            break;
          else
            
            feedBackCount = feedBackCount + 1;
            
            out = num2cell(feedbackHeap.ExtractMin());
            [feedBackTime,gz,gradient,reward] = out{:};
            
            % count feedback loss function
            gzs(feedBackCount) = gz;
            
            % update y + 1
            y = y - (1 / eta1) * gradient;
            myRewards(t) = myRewards(t) + reward;
            
            % update u
            
            u= updateExpert(experts,t,feedBackCount,gzs);
            u = project(u,x_bound);
          end
          
        end
        
        experts(t) = u;
        % note the t = feedbackcount
        expertsRewards(t) = expertLoss(experts(t),gzs,feedBackCount);
        
      end
    end
    
    regrets(t) = myRewards(t) - expertsRewards(t);
    regrets_div_t(t) = regrets(t) / t;
  end
  outMyRewards = myRewards;
  outExpertsRewards = expertsRewards;
  outRegrets = regrets;
  %   delayCompare=[myRewards',expertsRewards',delaytimes];
  
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
function uout = gradients(x_t,gz)
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


%%%%%%%%%%%%%%%%%%%%%
%  delay function   %
%%%%%%%%%%%%%%%%%%%%%

function [feedBackTime,gz] = boundDelay(t,B,D)
  
  feedBackTime = randi([1,B])+t;
  gz =  rand(1)*D;
end

function [feedBackTime,gz] = linearDelay(t,slop,D)
  feedBackTime = t * slop;
  gz =  rand(1)*D;
end

function [feedBackTime,gz] = logDelay(t,D)
  if t < 2
    feedBackTime = t+1;
  else
    feedBackTime = t + ceil(t * log2(t));
  end
  gz =  rand(1)*D;
end

function [feedBackTime,gz] = squareDelay(t,D)
  feedBackTime = t^2 + t;
  gz =  rand(1)*D;
end

function gDelayedFeedBack(B,t,x_t,D,feedbackHeap,type)
  switch lower(type)
    case 'bound'
      [feedBackTime,gz] = boundDelay(t,B,D);
    case 'linear'
      [feedBackTime,gz] =  linearDelay(t,2,D);
    case 'log'
      [feedBackTime,gz] = logDelay(t,D);
    case 'square'
      [feedBackTime,gz] = squareDelay(t,D);
    otherwise
      error('Delay type err');
  end
  
  gradient = gradients(x_t,gz);
  reward = userLoss(x_t,gz);
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
    iteration(2^(m-1),2^(m)-1,true);
  end
  

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