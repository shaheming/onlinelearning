function out = LOGD_UNEQUATION(M)
  %use doubling tricking to iterate
  % M = 18; % 2 ^ 15 = 32768
  mkdir img LOGD_UNEQUATION;
  
  global img_path;
  img_path ='LOGD_UNEQUATION/';
  % the maxiums turn will iterate T times;
  T = 2^(M)-1; % avoid the last value to 0
  
  N = 4;
  global x_bound;
  x_bound = ones(N,2).*[0,10^4];
  y0 = [0,0,0,0];
  
  global updateP;
  %  updateP = [1/10 ,1/10,1/10,1/10];
  %   updateP = [1,1,1,1];
  %   updateP = ones(1,N)*1/8;
  %  updateP = [1/10,1/100,1/10,5/10];
  updateP = [ 0.4259 ,0.8384 ,0.7423 ,0.0005];
  
  isUseP = true;
  
  global G;
  global eta;
  global r_start;
  G = [6,1,2,1,;1,6,1,2;2,1,6,1;1,2,1,6];
  eta = [0.1;0.2;0.3;0.1];
  r_start = [0.5,0.5,0.5,0.5];
  
  global xFigName_1;
  global xFigName_2;
  xFigName_1 = sprintf('%s-%s-p1-%.3f-p2-%.3f','LOGD-M-Link-1-2','X',updateP(1),updateP(2));
  xFigName_2 = sprintf('%s-%s-p1-%.3f-p2-%.3f','LOGD-M-Link-3-4','X',updateP(3),updateP(4));
  
  %   q=testUnequation(G,r_start,eta);
  %   outP = findUnconvergeP(G,r_start,eta);
  %   if size(outP) ~= size([])
  %     updateP = outP(1,:);
  %   end
  
  
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  rng(2);
  
  
  OGD_Primary(T,y0,N,isUseP);
  
  %%%%%%%end%%%%%%%%
  
  
end



function  OGD_Primary(T,y0,N,isUseP)
  
  
  markersize = 1;
  disp('Begin Loop\n');
  fprintf('Iterate %d turns',T);
  
  %%%iteration begin
  choices_1=iteration(1,T,y0,N,isUseP);
  %%%end
  y0_1=[0.016815,0.023363,0.031101, 0.016220];
  %   choices_2=iteration(1,T,y0_1,N);
  choices_2=ones(T,N).*y0_1;
  disp('End Loop');
  
  global xFigName_1;
  global xFigName_2;
  
  
  for i = 1:2:N*2
    lineName{i} =sprintf('p:%d',(i+1)/2);
    lineName{i+1} =sprintf('p%d*',(i+1)/2);
  end
  
  xFig = figure('name',xFigName_1,'NumberTitle','off');
  set(xFig,'position',get(0,'screensize'));
  hold on;
  matrix1=[y0(1),y0_1(1),y0(2),y0_1(2);choices_1(:,1),choices_2(:,1),choices_1(:,2),choices_2(:,2)];
  
  plot(matrix1(:,1),'-o','MarkerSize',markersize,'DisplayName',char(lineName{1}),'LineWidth',1.5,'color','b');
  hold on;
  plot(matrix1(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5,'color','r');
  hold on;
  plot(matrix1(:,3),'-o','MarkerSize',markersize,'DisplayName',char(lineName{3}),'LineWidth',1.5,'color','c');
  hold on;
  plot(matrix1(:,4),'-.','DisplayName',char(lineName{4}),'LineWidth',1.5,'color','m');
  
  hold on;
  legh  =legend(lineName(1:N),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  title(xFigName_1,'FontSize',20,'FontWeight','normal');
  hold off;
  
  matrix2=[y0(3),y0_1(3),y0(4),y0_1(4);choices_1(:,3),choices_2(:,3),choices_1(:,4),choices_2(:,4)];
  
  xFig_1 = figure('name',xFigName_2,'NumberTitle','off');
  set(xFig_1,'position',get(0,'screensize'));
  hold on;
  %   plot(matrix2,'DisplayName',char(lineName{N+1:N}),'LineWidth',1.5);
  plot(matrix2(:,1),'-o','MarkerSize',markersize,'DisplayName',char(lineName{1}),'LineWidth',1.5,'color','b');
  hold on;
  plot(matrix2(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5,'color','r');
  hold on;
  plot(matrix2(:,3),'-o','MarkerSize',markersize,'DisplayName',char(lineName{3}),'LineWidth',1.5,'color','c');
  hold on;
  plot(matrix2(:,4),'-.','DisplayName',char(lineName{4}),'LineWidth',1.5,'color','m');
  hold on;
  legh  =legend(lineName(N+1:end),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  title(xFigName_2,'FontSize',20,'FontWeight','normal');
  hold off;
  global img_path;
  
  xFigName_1 = sprintf('%s%s-%s-%s',img_path,'LOGD_M-Link-1-2','X',datestr(now, 'dd-mm-yy-HH-MM-SS'));
  xFigName_2 = sprintf('%s%s-%s-%s',img_path,'LOGD_M-Link-3-4','X',datestr(now, 'dd-mm-yy-HH-MM-SS'));
  saveas(xFig,strcat('img/',xFigName_1),'png');
  saveas(xFig_1,strcat('img/',xFigName_2),'png');
  
end


function outChoices=iteration(t_b,t_e,y0,N,isUseP)
  
  global x_bound;
  global updateP;
  global G;
  global eta;
  global r_start;
  
  
  
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
    gradientTmp = binornd(1,updateP).*gradient(choices(t,:),G,r_start,eta);
    if isUseP
      y = y - (1 / eta1)*gradientTmp./updateP;
    else
      y = y - (1 / eta1)*gradientTmp;
    end
    
  end
  outChoices = choices;
  
end

% the difference of reward function U
function outX = gradient(x,G,r_start,eta)
  outX = (diag(G).*x'-r_start'.*(G*x' - x'.*diag(G)+ eta))';
  %*(size(G,1)-1)
end

% the reward function U
function uout = userCost(x,G,r_start,eta)
  uout = (1./(2*diag(G)).*(diag(G).*x'-r_start'.*(G*x' - x'.*diag(G) + eta)).^2)';
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


function outX = gradientTest(x,G,r_start,eta)
  x_best = [0.016815,0.023363,0.031101, 0.016220];
  outX = sum( (diag(G).*x'-r_start'.*(G*x' - x'.*diag(G)+ eta))./diag(G).*(x-x_best)' );
  
  %*(size(G,1)-1)
end

function outXP = gradientTestP(x,G,r_start,eta,p)
  x_best = [0.016815,0.023363,0.031101, 0.016220];
  outXP = sum( (diag(G).*x'-r_start'.*(G*x' - x'.*diag(G)+ eta)).*(p'./diag(G)).*(x-x_best)' );
end

function outP=findUnconvergeP(G,r_start,eta)
  x = rand(2^20,4)*100;
  %or
  %x = rand(1,4)*100.*ones(2^20,4);
  p = rand(2^20,4);
  testParameter = gradientTestP(x,G,r_start,eta,p);
  pos= find(testParameter < 0);
  outP = p(pos',:);
end


function outP=testUnequation(G,r_start,eta)
  x = rand(2^25,4)*100;
  testParameter = gradientTest(x,G,r_start,eta);
  pos= find(testParameter < 0);
  outP = x(pos',:);
end

% x = rand(N,4)*100;
% outX = sum( (diag(G).*x'-r_start'.*(G*x' - x'.*diag(G)+ eta))./diag(G).*(x-x_best)' );
% outX = outX'.*ones(N,4);
% xa = 1:N;
% xa =xa'.*ones(N,4);
% stem3(xa,x,outX)
% figure
% stem3(xa,x,outX)
%
% p = [1/10,1/100,1/10,5/10];
% outXP =  (diag(G).*x'-r_start'.*(G*x' - x'.*diag(G)+ eta)).*(p'./diag(G)).*(x-x_best)' ;
% outXP = outXP'.*ones(N,4);
% outXP=sum(outXP);
% outXP = outXP'.*ones(N,4);
% stem3(xa,x,outXP)


% x = rand(2^25,4)*100;
% outXP =  (diag(G).*x'-r_start'.*(G*x' - x'.*diag(G)+ eta)).*(p'./diag(G)).*(x-x_best)' ;
% outXP=sum(outXP);
% find(outXP < 0)


