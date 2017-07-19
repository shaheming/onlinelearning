function out = LOGD_M(M)
  %use doubling tricking to iterate
  % M = 18; % 2 ^ 15 = 32768
  % the maxiums turn will iterate T times;
  T = 2^(M)-1; % avoid the last value to 0
  % T = 50000;
  
  global x_bound;
  x_bound = [0,1;0,2*pi];
  y0 = [1.5,3*pi];
  % D and eta are used in reward function U
  global eta;
  eta = 1;
  N = 5;

  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  
  
  %    rng(2);
  OGD_Primary(T,y0,N);
  
  %%%%%%%end%%%%%%%%
  
  
end



function  OGD_Primary(T,y0,N)
  
  mkdir img;
  
  disp('Begin Loop');
  fprintf('Iterate %d turns',T);
  %%%iteration begin
  choices=iteration(1,T,y0,N);
  %%%end
  disp('End Loop');
 
  xFig = figure('name','LOGD-MULTIAGENT','NumberTitle','off');
  set(xFig,'position',get(0,'screensize'));
  hold on;
  
  for i = 1:N
    lineName{i} =sprintf('Agent: %d',i);
  end
  
  plot(choices,'DisplayName',char(lineName),'LineWidth',1.5);
  hold on;
  
  legh  =legend(lineName,'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  xFigName = sprintf('%s-%s','LOGD_M','X');
  saveas(xFig,strcat('img/',xFigName),'png');
  
end


function outChoices=iteration(t_b,t_e,y0,N)
  
  global x_bound;
  
  G = rand(N,N);
  eta =rand(N,1);
  r_start = rand(N,1);
  for i = 1:N
    x_bound(i,1) = 0;
    x_bound(i,2) = 5;
  end
  
  choices=zeros(t_e,N);
  y0 = 1:N;
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1
    x_0 = project(y0,x_bound,N);
    %x0 feedback
    y = y0 - 1*gradient(x_0,G,r_start,eta);
  end
  
  for t = t_b : t_e
    %Z(1:end) = D * rand(size(Z,1),1);
    % thetas(t) = G(2:end) * Z;
    % my choice
    eta1 = t + 1;
    %x
    choices(t,:) = project(y,x_bound,N);
    %y
    y = y - (1 / eta1)*gradient(choices(t,:),G,r_start,eta);
  end
  outChoices = choices;
  
end

% the difference of reward function U
function outX = gradient(x,G,r_start,eta)
  outX = (diag(G).*x'-r_start.*(G*x' - x'.*diag(G)+ eta))';
end

% the reward function U
function uout = userCost(x,G,r_start,eta)
  uout = (1./(2*diag(G)).*(diag(G).*x'-r_start.*(G*x' - x'.*diag(G)+ eta)).^2)';
end


%the projection funciton
function x_t = project(y_t,x_bound,N)
  for i = 1:N
    if x_bound(i,1) <= y_t(i) && y_t(i) <=x_bound(i,2)
      x_t(i) = y_t(i);
    else
      if y_t(i) < x_bound(i,1)
        x_t(i) = x_bound(i,1);
      else
        x_t(i) = x_bound(i,2);
      end
    end
  end
end



