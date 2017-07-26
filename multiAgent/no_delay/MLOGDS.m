function out = MLOGDS(M)
  % the maxiums turn will iterate T times;
  T = 2^(M)-1; % avoid the last value to 0
  mkdir img MLOGDS;
  algorithmName = 'Multi-agnet-LOGD-NOIOSE';
  global img_path;
  img_path ='MLOGDS/';
  y0 = [0,0];
  
  N = size(y0,2);
  global X_BOUND;
  X_BOUND = ones(N,2).*[0,10^4];
  
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  %rng(2);
 types = {'Bernoulli','Log-normal','Markovian'};
%  types = {'Bernoulli'};
  for i = types
    rng(1);
    OGD_Primary(T,y0,N,algorithmName,i);
  end
  
  %%%%%%%end%%%%%%%%
  
  pause(5);
  close all;
end



function  OGD_Primary(T,y0,N,algorithmName,type)
  
  fprintf('Begin Loop of %s dsitribution\n',char(type));
  fprintf('Iterate %d turns\n',T);
  
  %%%iteration begin
  choices_1=iteration(1,T,y0,N,char(type));
  
  %%best choice
  y0_1=[0.06125,0.05125];
%  choices_2=iteration(1,T,y0_1,N,char(type));
  choices_2 = ones(T,2).*y0_1;
  fprintf('End Loop\n');
  
  
  
  for i = 1:2:N*2
    lineName{i} =sprintf('p:%d',(i+1)/2);
    lineName{i+1} =sprintf('p:%d*',(i+1)/2);
  end
  
  xFigName = sprintf('%s-%s-%s-%s',algorithmName,char(type),'Link-1','X');

  xFig = figure('name',xFigName,'NumberTitle','off');
  set(xFig,'position',get(0,'screensize'));
  hold on;
  matrix1=[y0(1),y0_1(1);choices_1(:,1),choices_2(:,1)];
  
  plot(matrix1(:,1),'-.','DisplayName',char(lineName{1}),'LineWidth',1.5,'color','b');
  hold on;
  plot(matrix1(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5,'color','r');
  hold on;
  

  title(xFigName,'FontSize',20,'FontWeight','normal');
  hold on;
  legh  =legend(lineName(1:N),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
    global img_path;
  saveas(xFig,strcat('img/',img_path,xFigName),'png');
  
  
  matrix2=[y0(2),y0_1(2);choices_1(:,2),choices_2(:,2)];
  
  xFigName = sprintf('%s-%s-%s-%s',algorithmName,char(type),'Link-2','X');

  xFig_1 = figure('name','LOGD-MULTIAGENT','NumberTitle','off');
  set(xFig_1,'position',get(0,'screensize'));
  hold on;
  plot(matrix2(:,1),'-.','DisplayName',char(lineName{1}),'LineWidth',1.5,'color','b');
  hold on;
  plot(matrix2(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5,'color','r');
  hold on;
  title(xFigName,'FontSize',20,'FontWeight','normal');
  hold on;
  legh  =legend(lineName(N+1:end),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  

  xFigName = sprintf('%s%s-%s-%s-%s',img_path,algorithmName,char(type),'Link-2','X');
  saveas(xFig_1,strcat('img/',xFigName),'png');
  
end


function outChoices=iteration(t_b,t_e,y0,N,type)
  
  global X_BOUND;
  
  p=[1/4,3/4];
  
  
  G1=[1,3;3,1];
  G2=[2,1;1,2];
%   G1=[0.55,0.45;0.45,0.55];
%   G2=[2.15,1.85;1.85,2.15];
  ETA1=[0.1;0.2];
  ETA2=[0.15;0.05];

  PT=[2/5,3/5;1/5,4/5];
  P = zeros(t_e+1,1);
  R_START = ones(1,N)*0.5;
 
  STATE = [0,0];
  choices=zeros(t_e,N);
  
  if t_b == 1
    x_0 = project(y0,X_BOUND,N);
    %x0 feedback
    
    [G,ETA,STATE]  =  stochasticFunct(G1,G2,ETA1,ETA2,STATE,p,PT,type);
     P(1) = STATE(1);
    y = y0 - 1*gradient(x_0,G,R_START,ETA);
  end
  
  for t = t_b : t_e
    eta1 = t + 1;
    %x
    choices(t,:) = project(y,X_BOUND,N);
    %y
    [G,ETA,STATE] =   stochasticFunct(G1,G2,ETA1,ETA2,STATE,p,PT,type);
    P(1+t) = STATE(1);
    y = y - (1 / eta1)*gradient(choices(t,:),G,R_START,ETA);
  end
  outChoices = choices;
  
end

% the difference of reward function U
function outX = gradient(x,G,R_START,eta)
  outX = (diag(G).*x'-R_START'.*(G*x' - x'.*diag(G)+ eta))';
end

% the reward function U
function uout = userCost(x,G,R_START,eta)
  uout = (1./(2*diag(G)).*(diag(G).*x'-R_START'.*(G*x' - x'.*diag(G)+ eta)).^2)';
end


%the projection funciton
function x_t = project(y_t,X_BOUND,N)
  for i = 1:N
    if X_BOUND(i,1) <= y_t(i) && y_t(i) <=X_BOUND(i,2)
      x_t(i) = y_t(i);
    else
      if y_t(i) < X_BOUND(i,1)
        x_t(i) = X_BOUND(i,1);
      else
        x_t(i) = X_BOUND(i,2);
      end
    end
  end
end

% types = {'Bernoulli','Log-normal','Markovian'};
function  [G,ETA,outSTATE] = stochasticFunct(G1,G2,ETA1,ETA2,lastSTATE,p,PT,type)
  switch type
    case 'Bernoulli'
    [G,ETA,outSTATE] = bernoulli(G1,G2,ETA1,ETA2,p);
    case 'Log-normal'
     [G,ETA,outSTATE] = logNormal(G1,G2,ETA1,ETA2,p);
    case 'Markovian'
     [G,ETA,outSTATE] =  markovian(G1,G2,ETA1,ETA2,lastSTATE,p,PT);
    otherwise
      error('Delay type err');
  end
  
end

function [G,ETA,outSTATE] = bernoulli(G1,G2,ETA1,ETA2,p)
  
  output = binornd(1,p(1));
  outputs =[output,1-output];
  G =  G1*outputs(1)+G2*outputs(2);
  ETA = ETA1*outputs(1)+ETA2*outputs(2);
  outSTATE = outputs;
end

function [G,ETA,outSTATE]= logNormal(G1,G2,ETA1,ETA2,p)
  ETA_E = p(1) * ETA1 + p(2) *ETA2;
  G_E = p(1)  * G1+p(2)*G2;
  %log normal G
  mu=log(G_E)-1/2*ones(size(G_E));
  G=lognrnd(mu,[1,1;1,1]);
  %log normal eta
  mu=log(ETA_E)-1/2*ones(size(ETA_E));
  ETA = lognrnd(mu,[1;1]);
  outSTATE = [-1,-1];
end

function [G,ETA,outSTATE] =  markovian(G1,G2,ETA1,ETA2,lastSTATE,p,PT)
 s1={G1,ETA1};
 s2={G2,ETA2};
 S = {s1,s2};
 if lastSTATE ==[0,0]
   output = binornd(1,p(1));
   outSTATE =[output,1-output];
 elseif lastSTATE == [1,0]
   output = binornd(1,PT(1,1));
   outSTATE =[output,1-output];
 elseif lastSTATE ==[0,1]
   output = binornd(1,PT(2,1));
   outSTATE =[output,1-output];
 end
 outputs =outSTATE;
 if outputs(1) == 1
   s= S{1};
 else
   s=S{2};
 end
 G = s{1};
 ETA = s{2};
 %update probility
end

