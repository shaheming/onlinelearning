
function  InjectionLOGD( M )
  % M = 15;
  
  mkdir img injectionLOGD;
  global img_path;
  img_path ='injectionLOGD';
  algorithmName = 'injectionLOGD';
  global y0;
  y0 =48;
  global theta;
  theta = 100;
  regretsFigName = sprintf('%s-%s',algorithmName,'Regrets');
  xFigName =sprintf('%s eta=t+%d y0=%d iteration=%d',algorithmName,theta,y0,2^M-1);
  
  global B;
  B = 5;
  global step;
  step = 5;
  
  
  global x_bound;
  x_bound = [0,1000];
  % D and  are used to generate Z
  global D;
  D = 10;
  
  isDraw = false;
  %   types = {'nodelay','bound','linear','log','square','exp','step'};
   types = {'log','square','linear'};
%   types = {'log'};
  %   regrets = {size(types,2)};
  cFig = figure('name',xFigName,'NumberTitle','off');
  set(cFig,'position',get(0,'screensize'));
  index = 0;
  for i = types
    index = index+ 1;
    [outMyChoices] = OGD_DELAY_IN(char(i),M,isDraw);
    plot(outMyChoices,'DisplayName',char(i),'LineWidth',1.5);
    clear outMyChoices;
    hold on;
  end
  
  title(xFigName,'FontSize',20,'FontWeight','normal');
  hold on;
  
  legh  =legend(types,'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  saveas(cFig,strcat('img/',xFigName),'png');
end



function [outMyChoices]= OGD_DELAY_IN(type,M,isDraw)
  import MinHeap
  T = 2^(M)-1; % avoid the last value to 0
  global log_bound;
  log_bound = T;
  % your decision domain used in projection
  global gzs;
  gzs  = zeros(1,T); % <G , Z>
  
  global myChoices;
  myChoices = zeros(1,T);
  %   global myRewards;
  %   myRewards = zeros(1,T);
  
  % the initial y
  global y0;
  y1 = y0;
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  
  
  fprintf('Begin Loop with %s form delay\n',type);
  fprintf('Iterate %d turns\n',T);
  
  [~]= iteration(1,T,y1,false,type);
  
  fprintf('End Loop\n');
  outMyChoices =myChoices;
  
  %%%%%%%end%%%%%%%%
  
end




function [outY] = iteration(t_b,t_e,y1,doubling_flag,type)
  global gzs;
  global x_bound;
  global myChoices;
  global theta;
  y = y1;
  
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1
    % t = 0
    z_t = project(y1,x_bound);
    
    
    gz =50;
    % y_1
    y = y - 1/theta*gradients(z_t,gz);
    % gzs(1:end) = rand(1,t_e)* D;
    gzs(1:end) = ones(1,t_e)*50;
    eta1 = theta;
    feedBackSum = gradients(z_t,gz);
    feedBacks= generateFeedBacks(t_e,type);
    originTime = 1;
  end
  
  % from 1
  for t = t_b : t_e
    % generate delayed feedback
    %  generate feedback dela
    
    % update x
    % gDelayedFeedBack(B,step,t,feedbackHeap,type);
    myChoices(t) = project(y,x_bound);
    
    %     [feedBackTime,originTime] = getFeedBack(t,type);
    feedBackTime = feedBacks(originTime);
    
    if feedBackTime - 1  == t
      % clean
      
      feedBackSum = 0;
      % get all feedbacks
      % count feedback loss function
      % originTime is the time the delay genrate
      feedBackSum = feedBackSum + gradients(myChoices(originTime),gz);
      originTime = originTime + 1;
    end
    eta1 = eta1+1;
    
    y = y - (1 / eta1) * feedBackSum;
    
  end
  outY = y;
end


% the difference of reward function U
function uout = gradients(x_t,gz)
  uout = x_t - gz;
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




function [feedBackTime] = getFeedBack(t,type)

  switch lower(type)
    case 'linear'
      [feedBackTime] =  t*50+t;
    case 'log'
       if t == 1
         feedBackTime  = 2;
       else
        feedBackTime = t*ceil(log2(t)) + t ;
       end
    case 'square'
      [feedBackTime] = t*t;
    case 'exp'
      [feedBackTime] = 2^t;
    otherwise
      error('Delay type err');
  end

end

function feedBacks = generateFeedBacks(T,type)
  feedBacks = zeros(T,1);
  for t = 1:T
    switch lower(type)
      case 'linear'
        feedBacks(t,1) =  t*50+t;
      case 'log'
        if t ==1
          feedBacks(t,1) = 2;
        else
          feedBacks(t,1)= t*ceil(log2(t)) + t;
        end
      case 'square'
        feedBacks(t,1) = t*t+t;
      case 'exp'
        feedBacks(t,1) = 2^t +t;
      otherwise
        error('Delay type err');
    end
  end
end
