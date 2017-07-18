


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
  delayData = [ceil(log2(iterations).*(iterations)+iterations),iterations];
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