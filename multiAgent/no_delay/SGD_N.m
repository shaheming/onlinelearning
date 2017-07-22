function out = SGD_N(M)
  % the maxiums turn will iterate T times;
  T = 2^(M)-1; % avoid the last value to 0
  mkdir img SGD_N;
  algorithmName = 'SGD-MULTIAGENT-NOIOSE';
  global img_path;
  img_path ='SGD_N/';
  y0 = [0,0];
  
  N = size(y0,2);
  global x_bound;
  x_bound = ones(N,2).*[0,10^4];
  
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  %rng(2);
   types = {'Bernoulli','Log-normal','Markovian'};
% types = {'Markovian'};
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
  
  y0 = [0,0];
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
  %   hold on;
  %   plot(matrix1(:,3),'-o','DisplayName',char(lineName{3}),'LineWidth',1.5,'color','c');
  %   hold on;
  %   plot(matrix1(:,4),'-.','DisplayName',char(lineName{4}),'LineWidth',1.5,'color','m');
  
  hold on;
  

  title(xFigName,'FontSize',20,'FontWeight','normal');
  hold on;
  legh  =legend(lineName(1:N),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  saveas(xFig,strcat('img/',xFigName),'png');
  
  
  matrix2=[y0(2),y0_1(2);choices_1(:,2),choices_2(:,2)];
  
  xFigName = sprintf('%s-%s-%s-%s',algorithmName,char(type),'Link-2','X');

  xFig_1 = figure('name','LOGD-MULTIAGENT','NumberTitle','off');
  set(xFig_1,'position',get(0,'screensize'));
  hold on;
  % plot(matrix2,'DisplayName',char(lineName{N+1:N}),'LineWidth',1.5);
  plot(matrix2(:,1),'-.','DisplayName',char(lineName{1}),'LineWidth',1.5,'color','b');
  hold on;
  plot(matrix2(:,2),'--','DisplayName',char(lineName{2}),'LineWidth',1.5,'color','r');
  hold on;
  %   plot(matrix2(:,3),'-o','DisplayName',char(lineName{3}),'LineWidth',1.5,'color','c');
  %   hold on;
  %   plot(matrix2(:,4),'-.','DisplayName',char(lineName{4}),'LineWidth',1.5,'color','m');
  %   hold on;
  title(xFigName,'FontSize',20,'FontWeight','normal');
  hold on;
  legh  =legend(lineName(N+1:end),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  hold off;
  
  global img_path;
  xFigName = sprintf('%s%s-%s-%s-%s',img_path,algorithmName,char(type),'Link-2','X');
  saveas(xFig_1,strcat('img/',xFigName),'png');
  
end


function outChoices=iteration(t_b,t_e,y0,N,type)
  
  global x_bound;
  
  %probability of ETA and G
  p=[1/4,3/4];
  
  
  G1=[1,3;3,1];
  G2=[2,1;1,2];
  
  ETA1=[0.1;0.2];
  ETA2=[0.15;0.05];

  PT=[2/5,3/5;1/5,4/5];
  
  r_start = ones(1,N)*0.5;
 
  
  choices=zeros(t_e,N);
  
  if t_b == 1
    x_0 = project(y0,x_bound,N);
    %x0 feedback
    
    [G,ETA,p] = stochasticFunct(G1,G2,ETA1,ETA2,[0,0],PT,type);
     
    y = y0 - 1*gradient(x_0,G,r_start,ETA);
  end
  
  for t = t_b : t_e
    %Z(1:end) = D * rand(size(Z,1),1);
    % thetas(t) = G(2:end) * Z;
    % my choice
    eta1 = t + 1;
    %x
    choices(t,:) = project(y,x_bound,N);
    %y
   [G,ETA,p] = stochasticFunct(G1,G2,ETA1,ETA2,p,PT,type);
    y = y - (1 / eta1)*gradient(choices(t,:),G,r_start,ETA);
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

% types = {'Bernoulli','Log-normal','Markovian'};
function   [G,ETA,p] = stochasticFunct(G1,G2,ETA1,ETA2,p,PT,type)
  switch type
    case 'Bernoulli'
      [G,ETA,p] = bernoulli(G1,G2,ETA1,ETA2,p);
    case 'Log-normal'
      [G,ETA,p] = logNormal(G1,G2,ETA1,ETA2,p);
    case 'Markovian'
      [G,ETA,p] =  markovian(G1,G2,ETA1,ETA2,p,PT);
    otherwise
      error('Delay type err');
  end
  
end

function  [G,ETA,outP] = bernoulli(G1,G2,ETA1,ETA2,p)
  
  output = binornd(1,0.25);
  outputs =[output,1-output];
  G =  G1*outputs(1)+G2*outputs(2);
% output = binornd(1,0.25);
% outputs =[output,1-output];
  ETA = ETA1*outputs(1)+ETA2*outputs(2);
  outP = p;
  
end

function  [G,ETA,outP] = logNormal(G1,G2,ETA1,ETA2,p)
  ETA_E = 0.25 * ETA1 + 0.75*ETA2;
  G_E = 0.25 * G1 + 0.75*G2;
  %log normal G
  mu=log(G_E)-1/2*ones(size(G_E));
  G=lognrnd(mu,[1,1;1,1]);
  %log normal eta
  mu=log(ETA_E)-1/2*ones(size(ETA_E));
  ETA = lognrnd(mu,[1;1]);
  outP = p;
end

function [G,ETA,outState] =  markovian(G1,G2,ETA1,ETA2,lastState,PT)
 s1={G1,ETA1};
 s2={G2,ETA2};
 S = {s1,s2};
 if lastState ==[0,0]
   output = binornd(1,0.25);
   outState =[output,1-output];
 elseif lastState ==[1,0]
   output = binornd(1,2/5);
   outState =[output,1-output];
 elseif lastState ==[0,1]
   output = binornd(1,1/5);
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

