
function  LOGD( M )

  mkdir img;
  algorithmName = 'LOGD-DELAY';
  
  regretsFigName = sprintf('%s-%s',algorithmName,'Regrets');
  xFigName = sprintf('%s-%s',algorithmName,'X');
  
  global B;
  B = 5;
  global step;
  step = 5;
  global y0;
  y0 = 8;
  
  global X_BOUND;
  X_BOUND = [0,1000];
  % D and  are used to generate Z
  global D;
  D = 100;
  
  types = {'nodelay','bound','linear','log','square','exp','step'};
  regrets = {size(types,2)};
  index = 0;
  for i = types
    rng(2);
    index = index+ 1;
    [outRegrets,outMyChoices] = OGD_DELAY_IN(char(i),M);
    regrets{index} = {outRegrets,outMyChoices,char(i)};
  end
  
  regFig = figure('name',regretsFigName,'NumberTitle','off');
  set(regFig,'position',get(0,'screensize'));
  
  for i = regrets
    plot(i{1}{1},'DisplayName',char(i{1}{3}),'LineWidth',1.5);
    hold on;
  end
  legh  =legend(types,'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  
  hold off;
  
  cFig = figure('name',xFigName,'NumberTitle','off');
  set(cFig,'position',get(0,'screensize'));
  
  for i = regrets
    plot(i{1}{2},'DisplayName',char(i{1}{3}),'LineWidth',1.5);
    hold on;
  end
  legh  =legend(types,'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  
  saveas(regFig,strcat('img/',regretsFigName),'png');
  saveas(cFig,strcat('img/',xFigName),'png');
  
  close all;
end



function [outRegrets,outMyChoices]= OGD_DELAY_IN(type,M)
  import MinHeap
  T = 2^(M)-1; % avoid the last value to 0
  % your decision domain used in projection
  global gzs;
  gzs  = zeros(1,T); % <G , Z>
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
  global y0;
  %generate delay
  global feedbackHeap;
  global B;
  global step;
  % generate delayed feedback
  % generate feedback dela
  feedbackHeap=gDelayedFeedBack(B,step,T,type);

  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%

  
  fprintf('Begin Loop with %s form delay\n',type);
  fprintf('Iterate %d turns\n',T);
  
  [outRegrets]= iteration(1,T,y0);
  

  outMyChoices =myChoices;
  
  %%%%%%%end%%%%%%%%
  
end


function[outRegrets]=iteration(t_b,t_e,y0)
  
  global D;
  global gzs;
  global X_BOUND;
  global regrets;
  global experts;
  global myChoices;
  global userLosses;
  global expertLosses;
  global feedbackHeap;
  

  y = y0;
  feedBackCount = 0;
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  if t_b == 1
    % t = 0
    % z_0
    x0 = project(y0,X_BOUND);
    gz =rand(1)*D;
    % y_1
    y = y - gradients(x0,gz);
    gzs(1:end) = rand(1,t_e)* D;
  end
  
  % from 1
  for t = t_b : t_e

    % update x
    myChoices(t) = project(y,X_BOUND);
    u=updateExpert(experts,t,gzs);
    experts(t)= project(u,X_BOUND);
    expertLosses(t)=expertLoss(experts(t),gzs,t);
    
    if t == 1
      userLosses(t) = userLoss(myChoices(t),gzs(t));
    else
      userLosses(t) = userLosses(t-1) + userLoss(myChoices(t),gzs(t));
    end
    
    if feedbackHeap.Count()
      % check delay
      out = num2cell(feedbackHeap.ReturnMin());
      [feedBackTime,~] = out{:};
      if feedBackTime - 1  == t
          eta1 = t+1;
        % get all feedbacks
        while feedbackHeap.Count() > 0
          out = num2cell(feedbackHeap.ReturnMin());
          [feedBackTime,~] = out{:};
          
          if feedBackTime - 1 > t
            break;
          else
            
            feedBackCount = feedBackCount + 1;
            out = num2cell(feedbackHeap.ExtractMin());
            [~,originTime] = out{:};
            
            y = y - (1 / eta1) * gradients(myChoices(originTime),gzs(originTime));
            
          end
        end
      end
    end
    
    regrets(t) = userLosses(t) - expertLosses(t);
    
  end
  
  outRegrets = regrets;
  
end

function u = updateExpert(experts,t,gzs)
  % note this with change with loss function
  if t ~= 1
    u = (t-1)/t * experts(t-1) + 1/t* gzs(t);
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
  uout = 0.5 * (x_t - gz)^2;
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


%%%%%%%%%%%%%%%%%%%%%
%  delay function   %
%%%%%%%%%%%%%%%%%%%%%


function feedbackHeap=gDelayedFeedBack(B,step,T,type)
 
  switch lower(type)
    case 'nodelay'
      [delayData] = boundDelay(T,1);
    case 'bound'
      [delayData] = boundDelay(T,B);
    case 'linear'
      [delayData] =  linearDelay(T,1);
    case 'log'
      [delayData] = logDelay(T);
    case 'square'
      [delayData] = squareDelay(T);
    case 'exp'
      [delayData] = expDelay(T);
    case 'step'
      [delayData] = stepDelay(T,step);
    otherwise
      error('Delay type err');
  end
  
  feedbackHeap = MinHeap(T,delayData);
end



function [delayData] = boundDelay(T,B)
  iterations =  (1:T)';
  delayData = [randi([1,B],T,1)+iterations,iterations];
end

function [delayData] = linearDelay(T,slop)
  iterations =  (1:T)';
  delayData = [iterations.*(slop+1),iterations];
end

function [delayData] = logDelay(T)
  iterations =  (1:T)';
  delayData = [ceil(ceil(log2(iterations).*(iterations)./20)+iterations),iterations];
  delayData(1,1) = 1 + delayData(1,2);
end

function [delayData] = squareDelay(T)
  iterations =  (1:T)';
  delayData = [iterations.^2 + iterations,iterations];
end

function [delayData] = expDelay(T)
  iterations = (1:T)';
  delayData = [2.^iterations + iterations,iterations];  
end

function [delayData] = stepDelay(T,step)
  delayData = zeros(T,2);
  for t = 1:T
  remainder = mod(t,step);
  if remainder ~=0
    delayData(t,1) = (step-remainder) + t+1;
  else
    delayData(t,1) = t+1;
  end
  delayData(t,2) = t;
  end
end
