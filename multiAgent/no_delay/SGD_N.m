function out = SGD_M(M)
  %use doubling tricking to iterate
  % M = 18; % 2 ^ 15 = 32768
  % the maxiums turn will iterate T times;
  T = 2^(M)-1; % avoid the last value to 0
  % T = 50000;
  mkdir img;
  global x_bound;
  x_bound = [0,1;0,2*pi];
  y0 = [0,0,0,0];
  % D and eta are used in reward function U
  global eta;
  eta = 1;
  N = 4;
  
  algorithmName = 'SGD_M-MULTIAGENT';
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  %rng(2);
  OGD_Primary(T,y0,N,algorithmName);
  
  %%%%%%%end%%%%%%%%
  
  
end



function  OGD_Primary(T,y0,N,algorithmName)
  
  fprintf('Begin Loop\n');
  fprintf('Iterate %d turns\n',T);
  
  y0 = [0,0,0,0];
  %%%iteration begin
  choices_1=iteration(1,T,y0,N);
  
  %%best choice
  y0_1=[0.016815,0.023363,0.031101, 0.016220];
  choices_2=iteration(1,T,y0_1,N);
  
  fprintf('End Loop');
 

  
  for i = 1:2:N*2
    lineName{i} =sprintf('p:%d',(i+1)/2);
    lineName{i+1} =sprintf('p%d*',(i+1)/2);
  end
  
  xFig = figure('name',algorithmName,'NumberTitle','off');
  set(xFig,'position',get(0,'screensize'));
  hold on;
  matrix1=[y0(1),y0_1(1),y0(2),y0_1(2);choices_1(:,1),choices_2(:,1),choices_1(:,2),choices_2(:,2)];
  
  plot(matrix1(:,1),'-o','DisplayName',char(lineName{1}),'LineWidth',1.5,'color','b');
  hold on;
  plot(matrix1(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5,'color','r');
  hold on;
  plot(matrix1(:,3),'-o','DisplayName',char(lineName{3}),'LineWidth',1.5,'color','c');
  hold on;
  plot(matrix1(:,4),'-.','DisplayName',char(lineName{4}),'LineWidth',1.5,'color','m');
  
  hold on;
  legh  =legend(lineName{1:N},'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  
  matrix2=[y0(3),y0_1(3),y0(4),y0_1(4);choices_1(:,3),choices_2(:,3),choices_1(:,4),choices_2(:,4)];

  xFig_1 = figure('name','LOGD-MULTIAGENT','NumberTitle','off');
  set(xFig_1,'position',get(0,'screensize'));
  hold on;
% plot(matrix2,'DisplayName',char(lineName{N+1:N}),'LineWidth',1.5);
  plot(matrix2(:,1),'-o','DisplayName',char(lineName{1}),'LineWidth',1.5,'color','b');
  hold on;
  plot(matrix2(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5,'color','r');
  hold on;
  plot(matrix2(:,3),'-o','DisplayName',char(lineName{3}),'LineWidth',1.5,'color','c');
  hold on;
  plot(matrix2(:,4),'-.','DisplayName',char(lineName{4}),'LineWidth',1.5,'color','m');
  hold on;
  legh  =legend(lineName{N+1:end},'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
    
  
  xFigName = sprintf('%s-%s','SGD_M-Link-1-2','X');
  saveas(xFig,strcat('img/',xFigName),'png');
  xFigName = sprintf('%s-%s','SGD_M-Link-3-4','X');
  saveas(xFig_1,strcat('img/',xFigName),'png');
  
end


function outChoices=iteration(t_b,t_e,y0,N)
  
  global x_bound;
  
  
  G = [6,1,2,1,;1,6,1,2;2,1,6,1;1,2,1,6];
  eta = [0.1;0.2;0.3;0.1];

  r_start = [0.5,0.5,0.5,0.5];
  for i = 1:N
    x_bound(i,1) = 0;
    x_bound(i,2) = 0.04;
  end
  
  choices=zeros(t_e,N);
 
% 
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
    eta1 = t +1;
    %x
    choices(t,:) = project(y,x_bound,N);
    %y
    y = y - (1 / eta1)*gradient(choices(t,:),G,r_start,eta);
  end
  outChoices = choices;
  
end

% the difference of reward function U
function outX = gradient(x,G,r_start,eta)
  outX = (diag(G).*x'-r_start'.*(G*x' - x'.*diag(G)+ eta))';
end

% the reward function U
function uout = userCost(x,G,r_start,eta)
  uout = (1./(2*diag(G)).*(diag(G).*x'-r_start'.*(G*x' - x'.*diag(G)+ eta)).^2)';
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



