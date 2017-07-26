
function  MULTI_MACHINE( M )
  % M = 15;
  
  mkdir img;

  global algorithmName;
  algorithmName = 'MULI-MACHINE';
  
  
  global X_BOUND;
  X_BOUND = [0,1000];  
  
  E_A = 5;
  E_B = 50;
  
  y0 =8;
  OGD_DELAY_IN(M,y0,E_A,E_B);
  
  close all;
end



function OGD_DELAY_IN(M,y0,E_A,E_B)
  T = 2^(M)-1; % avoid the last value to
  
  global algorithmName;
  
  
  fprintf('Begin Loop\n');
  fprintf('Iterate %d turns\n',T);
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  [outMyChoice]= iteration(1,T,y0,E_A,E_B);
  
  fprintf('End Loop\n');
  
  xFigName =sprintf('%s E[a]=%d E[b]=%d',algorithmName,E_A,E_B);
  cFig = figure('name',xFigName,'NumberTitle','off','Visible','on');
  set(cFig,'position',get(0,'screensize'));
  
  plot(outMyChoice,'DisplayName','X','LineWidth',1.5);
  hold on;
  title(xFigName,'FontSize',20,'FontWeight','normal');
  hold on;
  
  legh  =legend('X','Location','best');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  
  
  print(cFig,strcat('img/',xFigName),'-dpng','-r500');
  %%%%%%%end%%%%%%%%
  
end




function [outMyChoice] = iteration(t_b,t_e,y0,E_A,E_B)
  global X_BOUND;
  
  myChoices = zeros(1,t_e);
  ETA = 1;
  y = y0;
  T = zeros(1,10);
  A = zeros(1,10);
  B = zeros(1,10);
  Gradients = zeros(1,10);
  index = 1;
  
  pos = randperm(10);
  
% umcomment this part and comment 85 - 88 line , you can set different machine with different random distribution  
%   bernoulli_pos = pos(1:2);
%   uniform_pos = pos(3:5);
%   normal_pos = pos(6:8);
%   logNormal_pos = pos(9:end);
  
  
  bernoulli_pos = [];
  uniform_pos = pos;
  normal_pos = [];
  logNormal_pos = [];
  
  
  if t_b == 1
    % t = 0
    x0 = project(y0,X_BOUND);
    T(1:5) = rand(1,5)*10;
    T(6:end) = rand(1,5)*100;
    
    for i = bernoulli_pos
      B(i) = bernoulli(1/2,E_B);
      A(i) =  bernoulli(1/2,E_A);
    end
    
    for i = uniform_pos
      B(i) = uniform(E_B);
      A(i) =  uniform(E_A);
    end
    
    for i = logNormal_pos
      B(i) = logNormal(E_B);
      A(i) =  logNormal(E_A);
    end
    for i = normal_pos
      B(i) = normal(E_B);
      A(i) =  normal(E_A);
    end
    % Given x0, all machines will calculate gradient
    
    Gradients = gradients(x0,A,B);
 
    myChoices(index) = x0;
  end
  
  
  for t = 2 : t_e
    [M,I] = min(T);
   
    index= index + 1;
    ETA = ETA + 1;
    
    y = y - 1/ETA*Gradients(I);
    myChoices(index) = project(y,X_BOUND);
    T = T - M;
    T(I) = 0;
    
    
    if I > 5
      T(I) = rand(1)*100;
    else
      T(I) = rand(1)*10;
    end
    
    b = rand(1)*2*E_B;
    a = rand(1)*2*E_A;
    
    if find(bernoulli_pos == I)
      b = bernoulli(1/2,E_B);
      a =  bernoulli(1/2,E_A);
    elseif find(uniform_pos == I)
      b = uniform(E_B);
      a =  uniform(E_A);
    elseif  find(normal_pos == I)
      b = normal(E_B);
      a =  normal(E_A);
    elseif  find(logNormal_pos == I)
      b = uniform(E_B);
      a =  uniform(E_A);
    end
    Gradients(I) = gradients(myChoices(index),a,b);
    
  end
  outMyChoice = myChoices;
end



function GS = gradients(X,A,B)
  GS = A.*X - B;
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

function [outPut] = bernoulli(p,E)
  outPut =2*E*binornd(1,p);
end

function [outPut]= logNormal(E)
  mu=log(E)-1/2*ones(size(E));
  outPut=lognrnd(mu,ones(size(E)));
end

function [outPut]= normal(E)
  outPut=normrnd(E,1);
end



function [outPut]= uniform(E)
  outPut=rand(1)*2*E;
end

