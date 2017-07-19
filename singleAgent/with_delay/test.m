function test()
  
  
  % step = 100;
  % theta = 0:2*pi/step:2*pi;
  % % r =1./(N:-0.1:1);
  % r =0:1/step:1;
  %
  % [THETA,R]=meshgrid(theta,r);
  % G = (3 + sin(5*THETA)+ cos(3*THETA)).*(R.^2).*(5/3-R);
  % %
  % figure;
  % hold on;
  %  mesh(R.*cos(THETA), R.*sin(THETA), G);
  %  hold on;
  %  shading interp
  %  for i = 0:0.1:4
  % plot3(0, 0, i,'ko');
  % hold on;
  %  end
  global T;
  T = 2^15-1;
  global log_bound;
  log_bound =T ;
  types = {'log'};%'linear',,'square','exp'
  feedBacks = zeros(2^15,2);
  feedBacks= zeros(T,1);
  feedBacks= generateFeedBacks(T,'log');
  for i = 1:T
    tic;
    [a,b]= getFeedBack(i,'log');
    toc;
    feedBacks(i,:)=[a,b];
  end
  %   for i = types
  %     fprintf('%s',char(i));
  %     for t = 1:T
  %       [feedBackTime,originTime]=getFeedBack(t,char(i));
  %       if feedBackTime -1 ==t
  %         fprintf('feedBackTime: %d t: %d originTime: %d \n',feedBackTime,t,originTime);
  %       end
  %     end
  %   end
  
end


function [feedBackTime,originTime] = getFeedBack(t,type)
  global log_bound;
  switch lower(type)
    case 'linear'
      [originTime] =  t/50;
    case 'log'
      
      for i = floor(log2(t)):log_bound
        if t ==1
          originTime = t;
          break;
        elseif i*ceil(log2(i)) + i== t
          originTime = i;
          break;
        else
          originTime = 0.1;
        end
      end
    case 'square'
      [originTime] = sqrt(t);
    case 'exp'
      [originTime] = log2(t);
    otherwise
      error('Delay type err');
  end
  if ceil(originTime) == originTime
    feedBackTime = t + 1;
  else
    feedBackTime = t - 1;
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
  
