
function  StepsizeLOGD( M )
  % M = 15;
  mkdir img StepsizeLOGD;
  global img_path;
  img_path ='StepsizeLOGD/';

  algorithmName = 'StepsizeLOGD';
  regretsFigName = sprintf('%s-%s',algorithmName,'Regrets');
  xFigName = sprintf('%s-%s',algorithmName,'X');

  global B;
  B = 5;
  global step;
  step = 5;
  global y0;
  y0 = 8;
  
  global x_bound;
  x_bound = [0,100];
  % D and  are used to generate Z
  global D;
  D = 100;
  

  isDraw = false;
  types = {'nodelay','bound','linear','log','square','exp','step'};
  regrets = {size(types,2)};
  index = 0;
  
  for i = types
    rng(1);
    index = index+ 1;
    [outRegrets,outMyChoices] = OGD_DELAY_IN(char(i),M,isDraw);
    regrets{index} = {outRegrets,outMyChoices,char(i)};
  end
  regFig = figure('name',regretsFigName,'NumberTitle','off');
  set(regFig,'position',get(0,'screensize'));
  
  for i = regrets
    plot(i{1}{1},'DisplayName',char(i{1}{3}),'LineWidth',1.5);
    hold on;
  end
  legh =legend(cellstr(types));
  legh.LineWidth = 2;
  legh.FontSize = 20;
  
  hold off;

  cFig = figure('name',xFigName,'NumberTitle','off');
  set(cFig,'position',get(0,'screensize'));
  
  for i = regrets
    plot(i{1}{2},'DisplayName',char(i{1}{3}),'LineWidth',1.5);
    hold on;
  end
  legh  =legend(types);
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  
  saveas(regFig,strcat('img/',regretsFigName),'png');
  saveas(cFig,strcat('img/',xFigName),'png');
end



function [outRegrets,outMyChoices]= OGD_DELAY_IN(type,M,isDraw)
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
  global expertsRewards;
  expertsRewards = zeros(1,T);
  global myChoices;
  myChoices = zeros(1,T);
  global myRewards;
  myRewards = zeros(1,T);
  
  % the initial y
  global y0;
  y1 = y0;

  
  global feedbackHeap;
  feedbackHeap = MinHeap(T+1,ones(1,4)* inf);
  feedbackHeap.ExtractMin();

  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  
  if ~isDraw
    figConfig = 'off';
  else
    figConfig = 'on';
  end
  
  fprintf('Begin Loop with %s form delay\n',type);
  fprintf('Iterate %d turns\n',T);
  %%%% iteration 
  iteration(1,T,y1,false,type);
  
  fprintf('End Loop\n');
  headline = sprintf('LOGD %s Delay Choice',type);
  imgXCompare = figure('name',headline,'NumberTitle','off','Position',[0,500,700,500],'visible',figConfig);
  
  plot(experts,'DisplayName','experts','LineWidth',1);
  hold on;
  plot(myChoices,'DisplayName','mychoice','LineWidth',1);
  legh  = legend('experts','mychoice');
  legh.FontSize = 16;
  title(headline,'FontSize',20,'FontWeight','normal');
  hold off;
  
  headline = sprintf('LOGD %s Delay Regret',type);
  imgRegret = figure('name',headline,'NumberTitle','off','Position',[700,500,700,500],'visible',figConfig);
  plot(regrets,'DisplayName','regrets','LineWidth',1);
  title(headline,'FontSize',20,'FontWeight','normal');

 
  
  %   imgRewardCompare = figure('name','ExpertsRewards and myRewards','NumberTitle','off','Position',[100,500,700,500],'visible',figConfig);
  %   plot(myRewards,'DisplayName','myRewards');
  %   hold on;
  %   title('Rewards Compare','FontSize',20,'FontWeight','normal');
  %   plot(expertsRewards,'DisplayName','expertsRewards');
  %   legend('myRewards','expertsRewards');
  %   hold off;
  global img_path;
  
  %saveas(imgXCompare,strcat('img/',img_path,type,'_xcompare'),'png');
  %saveas(imgRegret,strcat('img/',img_path,type,'_regret'),'png');
  % saveas(imgRewardCompare,strcat('img/','type','_reward'),'png');
  outMyChoices =myChoices;
  outRegrets =regrets;
  %%%%%%%end%%%%%%%%
  
end



function[outY]=iteration(t_b,t_e,y1,doubling_flag,type)
  
  global B;
  global D;
  global step;
  global gzs;
  global x_bound;
  global regrets;
  global experts;
  global myChoices;
  global myRewards;
  global expertsRewards;
  global feedbackHeap;

  y = y1;
 lastUpdateTime = 0; 
 feedBackCount = 0;
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1
    % t = 0
    z_t = project(y1,x_bound);
    gz =rand(1)*D;
    y = y - gradients(z_t,gz);
    gzs(1:end) = rand(1,t_e)*D;
  end
  
  % from 1
  for t = t_b : t_e
    % generate delayed feedback
    %  generate feedback dela
    
    % update x 

    
    gDelayedFeedBack(B,step,t,feedbackHeap,type);
    
    myChoices(t) = project(y,x_bound);
    u=updateExpert(experts,t,gzs);
    experts(t)= project(u,x_bound);
    expertsRewards(t)=expertLoss(experts(t),gzs,t);
    
    if t == 1
      myRewards(t) = userLoss(myChoices(t),gzs(t));
    else
      myRewards(t) = myRewards(t-1) + userLoss(myChoices(t),gzs(t));
    end
    
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
          [feedBackTime,~,~,~] = out{:};
          
          if feedBackTime - 1 > t
            break;
          else
            
            feedBackCount = feedBackCount + 1;
            
            out = num2cell(feedbackHeap.ExtractMin());
            [~,choiceTime,~,~] = out{:};
            
            % count feedback loss function
            %gzs(feedBackCount) = gz;
            % update y + 1
            y = y - (1 / eta1) * gradients(myChoices(choiceTime),gzs(choiceTime));
            %myRewards(t) = myRewards(t) + reward;
          end
        end
      end
    end
    
    regrets(t) = myRewards(t) - expertsRewards(t);

  end

  % delayCompare=[myRewards',expertsRewards',delaytimes];
  outY = y;
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

function [feedBackTime] = boundDelay(t,B)
  feedBackTime = randi([1,B])+t;
end

function [feedBackTime] = linearDelay(t,slop)
  feedBackTime = t+t * slop;
end

function [feedBackTime] = logDelay(t)
  d = ceil( t * log2(t));
  if d  <= 1
    d = 1;
  end
  feedBackTime = t + d;
end

function [feedBackTime] = squareDelay(t)
  feedBackTime = t^2 + t;
end

function [feedBackTime] = expDelay(t)
  feedBackTime = 2^t + t;
end

function [feedBackTime] = stepDelay(t,step)
    remainder = mod(t,step);
    if remainder ~=0
    feedBackTime = (step-remainder) + t+1;
    else
      feedBackTime = t+1;
    end
end

function gDelayedFeedBack(B,step,t,feedbackHeap,type)
  switch lower(type)
    case 'nodelay'
      [feedBackTime] = boundDelay(t,1);
    case 'bound'
      [feedBackTime] = boundDelay(t,B);
    case 'linear'
      [feedBackTime] =  linearDelay(t,1);
    case 'log'
      [feedBackTime] = logDelay(t);
    case 'square'
      [feedBackTime] = squareDelay(t);
    case 'exp'
      [feedBackTime] = expDelay(t);
    case 'step'
      [feedBackTime] = stepDelay(t,step);      
    otherwise
      error('Delay type err');
  end
  
  gradient = nan;
  reward = nan;
  feedbackHeap.InsertKey([feedBackTime,t,gradient,reward]);
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