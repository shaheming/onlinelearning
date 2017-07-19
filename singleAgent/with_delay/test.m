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
  T = 2^10 -1;
  types = {'log'};%'linear',,'square','exp'
  for i = types
    fprintf('%s',char(i));
    for t = 1:T
      [feedBackTime,originTime]=getFeedBack(t,char(i));
      if feedBackTime -1 ==t
        fprintf('feedBackTime: %d t: %d originTime: %d \n',feedBackTime,t,originTime);
      end
    end
  end
  
end


function [feedBackTime,originTime] = getFeedBack(t,type)
  switch lower(type)
    case 'linear'
      [originTime] =  t/5;
    case 'log'
      global T;
      for i = 1:log2(T+1)
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