function out = LOGD_N(M)
  %use doubling tricking to iterate
  % M = 18; % 2 ^ 15 = 32768
  mkdir img LOGD_N;
  
  global img_path;
  img_path ='LOGD_N/';
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
%   updateP = [ 0.4259 ,0.8384 ,0.7423 ,0.0005];
   % updateP = [ 1/2 1/2 ,1/2 ,1/2];
  updateP = [1,1,1,1];
  isUseP = false;
  
  global G;
  global ETA;
  global r_start;
  G = [6,1,2,1,;1,6,1,2;2,1,6,1;1,2,1,6];
  ETA = [0.1;0.2;0.3;0.1];
  r_start = [0.5,0.5,0.5,0.5];
  
  global xFigName_1;
  global xFigName_2;
  xFigName_1 = sprintf('%s-%s-p1-%.3f-p2-%.3f-p3-%.3f-p4-%.3f','LOGD-M-Link-1-2-3-4','X',updateP);
  xFigName_2 = sprintf('%s-%s-p1-%.3f-p2-%.3f','LOGD-M-Link-3-4','X',updateP(3),updateP(4));
  
  %   q=testUnequation(G,r_start,eta);
  %   outP = findUnconvergeP(G,r_start,eta);
  %   if size(outP) ~= size([])
  %     updateP = outP(1,:);
  %   end
  
  
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  
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
  matrix2=[y0(3),y0_1(3),y0(4),y0_1(4);choices_1(:,3),choices_2(:,3),choices_1(:,4),choices_2(:,4)];
  
  plot(matrix1(:,1),'-o','MarkerSize',markersize,'DisplayName',char(lineName{1}),'LineWidth',1.5);
  hold on;
  plot(matrix1(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5);
  hold on;
  plot(matrix1(:,3),'-o','MarkerSize',markersize,'DisplayName',char(lineName{3}),'LineWidth',1.5);
  hold on;
  plot(matrix1(:,4),'-.','DisplayName',char(lineName{4}),'LineWidth',1.5);
  hold on;
  plot(matrix2(:,1),'-o','MarkerSize',markersize,'DisplayName',char(lineName{1}),'LineWidth',1.5);
  hold on;
  plot(matrix2(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5);
  hold on;
  plot(matrix2(:,3),'-o','MarkerSize',markersize,'DisplayName',char(lineName{3}),'LineWidth',1.5);
  hold on;
  plot(matrix2(:,4),'-.','DisplayName',char(lineName{4}),'LineWidth',1.5);
  hold on;
 
  legh  =legend(lineName(1:end),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  title(xFigName_1,'FontSize',20,'FontWeight','normal');
  hold off;
  
  global img_path;
  
  xFigName_1 = sprintf('%s%s-%s-%s',img_path,'LOGD_M-Link-1-2','X',datestr(now, 'dd-mm-yy-HH-MM-SS'));
  saveas(xFig,strcat('img/',xFigName_1),'png');
end


function outChoices=iteration(t_b,t_e,y0,N,isUseP)
  
  global x_bound;
  global updateP;
  global G;
  global ETA;
  global r_start;
  
  p=[1/4,3/4];
    
  G1 = ...
    [
       9 ,0.25,5 ,1;
    0.25 ,   9,1 ,5;
    5 ,1 ,9 ,0.25;
    1 ,5 ,0.25 ,9;
    ];
  
  G2 = ...
    [
    5,  1.25,  1,  1;
    1.25,  5,  1,  1;
    1,  1,  5,  1.25;
    1,  1,  1.25,  5;
    ];
  
  ETA1=[0.07;0.14;0.21;0.07];
  ETA2=[0.11;0.22;0.33;0.11];

  PT=[2/5,3/5;1/5,4/5];
    g = zeros(4,4);
  choices=zeros(t_e,N);
  state = [0,0];
   type = 'Log-normal';
%type = 'Bernoulli';
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1
    x_0 = project(y0,x_bound,N);
    %x0 feedback
    [G,ETA,state]  =  stochasticFunct(G1,G2,ETA1,ETA2,state,p,PT,type);
    y = y0 - 1*gradient(x_0,G,r_start,ETA);
   g = g +G;
  end
  
  for t = t_b : t_e
    %Z(1:end) = D * rand(size(Z,1),1);
    % thetas(t) = G(2:end) * Z;
    % my choice
    eta1 = t +1;
    %x
    choices(t,:) = project(y,x_bound,N);
    %y
    [G,ETA,state] =   stochasticFunct(G1,G2,ETA1,ETA2,state,p,PT,type);
    gradientTmp = binornd(1,updateP).*gradient(choices(t,:),G,r_start,ETA);
    g = g +G;
 
    if isUseP
      y = y - (1 / eta1)*gradientTmp./updateP;
    else
      y = y - (1 / eta1)*gradientTmp;
    end
    
  end
  outChoices = choices;
  disp(g/t_e);
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

% types = {'Bernoulli','Log-normal','Markovian'};
function  [G,ETA,outState] = stochasticFunct(G1,G2,ETA1,ETA2,lastState,p,PT,type)
  switch type
    case 'Bernoulli'
    [G,ETA,outState] = bernoulli(G1,G2,ETA1,ETA2,p);
    case 'Log-normal'
     [G,ETA,outState] = logNormal(G1,G2,ETA1,ETA2,p);
    case 'Markovian'
     [G,ETA,outState] =  markovian(G1,G2,ETA1,ETA2,lastState,p,PT);
    otherwise
      error('Delay type err');
  end
  
end

function [G,ETA,outState] = bernoulli(G1,G2,ETA1,ETA2,p)

  output = binornd(1,p(1));
  outputs =[output,1-output];
  G =  G1*outputs(1)+G2*outputs(2);
% output = binornd(1,0.25);
% outputs =[output,1-output];
  ETA = ETA1*outputs(1)+ETA2*outputs(2);
 
  outState = [-1,-1];
end

function [G,ETA,outState]= logNormal(G1,G2,ETA1,ETA2,p)
  ETA_E = p(1) * ETA1 + p(2) *ETA2;
  G_E = p(1)  * G1+p(2)*G2;
  %log normal G
  mu=log(G_E)-1/2*ones(size(G_E));
  G=lognrnd(mu,ones(size(G1)));
  %log normal eta
  mu=log(ETA_E)-1/2*ones(size(ETA_E));
  ETA = lognrnd(mu,ones(size(ETA1)));
  outState = [-1,-1];
end

function [G,ETA,outState] =  markovian(G1,G2,ETA1,ETA2,lastState,p,PT)
 s1={G1,ETA1};
 s2={G2,ETA2};
 S = {s1,s2};
 if lastState ==[0,0]
   output = binornd(1,p(1));
   outState =[output,1-output];
 elseif lastState == [1,0]
   output = binornd(1,PT(1,1));
   outState =[output,1-output];
 elseif lastState ==[0,1]
   output = binornd(1,PT(2,1));
   outState =[output,1-output];
 end
 outputs =outState;
 %begain
 %output=binornd(1,p(1));
 %outputs =[output,1-output];
 if outputs(1) == 1
   s= S{1};
 else
   s=S{2};
 end
 G = s{1};
 ETA = s{2};
 %update probility
end

