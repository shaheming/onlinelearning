function out = MLOGD_U_N(varargin)
  %use doubling tricking to iterat
  
  mkdir img MLOGD_U_N;
  global img_path;
  img_path ='MLOGD_U_N/';
  global algorithmName;
  algorithmName = 'MLOGD_U_N';
  
  global X_BOUND;
  global G1;
  global G2;
  global ETA1;
  global ETA2;
  global NOISE_P;
  global PT;
  global G0;
  global ETA0;
  global R_STAR;
  global updateP;
  global oP;
  
  M = varargin{1};
  % the maxiums turn will iterate T times;
  T = 2^(M)-1; % avoid the last value to 0
  N = 4;
  X_BOUND = ones(N,2).*[0,10^4];
  Y0 = [0,0,0,0];
  G0 = [6,1,2,1,;1,6,1,2;2,1,6,1;1,2,1,6];
  ETA0 = [0.1;0.2;0.3;0.1];
  R_STAR = [0.5,0.5,0.5,0.5];
  % optimum p
  oP=[0.016815,0.023363,0.031101, 0.016220];
  
  PT=[2/5,3/5;1/5,4/5]; %MKjTRAN
  NOISE_P=[1/4,3/4];
  
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
  
  %updateP = [ 1/2 1/2 ,1/2 ,1/2];
  %sha
  %updateP=[ 0.4259 ,0.8384 ,0.7423 ,0.0005];
  %sun
  %updateP = [0.9977 ,0.8468 ,0.0713 ,0.0049];
  
  types = {'No'};
  if size(varargin,2) == 1
    isUseP = false;
    updateP=[1,1,1,1];
  elseif size(varargin,2) >= 2
    if size(varargin{2},2) < N
      error('P dimensions is not match')
    else
      isUseP = true;
      updateP = varargin{2};
    end
    if size(varargin,2) == 3
      types = {'No','Log-normal','Bernoulli','Markovian'};
    end
  end
  
  if isUseP
    xFigTitle = sprintf('%s-[p1-p4]%.3f-%.3f-%.3f-%.3f','MLOGD-UpdateP-Link[1-4]',updateP);
    xFigName = sprintf('%s','MLOGD-UpdateP-Link[1-4]');
  else
    xFigTitle = sprintf('%s','MLOGD-Link[1-4]');
    xFigName =  sprintf('%s','MLOGD-Link[1-4]');
  end
  
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  
  
  isNormalize = true;
  
  fprintf('Begin Loop\n');
  fprintf('Iterate %d turns\n',T);
  tic
  for i = types
    if isUseP
      OGD_Primary(T,Y0,N,char(i),isUseP,~isNormalize,xFigTitle,xFigName);
      OGD_Primary(T,Y0,N,char(i),isUseP,isNormalize,xFigTitle,xFigName);
    else
      OGD_Primary(T,Y0,N,char(i),isUseP,~isNormalize,xFigTitle,xFigName);
    end
    toc
  end
  fprintf('End Loop\n');
  %%%%%%%end%%%%%%%%
  pause(3);
  close all;
end



function  OGD_Primary(T,Y0,N,noiseType,isUseP,isNormalize,xFigTitle,xFigName)
  
  %%%%%%%%%%%%%%%%%%
  %   SET TITLE  %
  %%%%%%%%%%%%%%%%%%
  lineWidth = 1;
  global algorithmName;
  global img_path;
  global updateP;
  imgName = sprintf('%s-%s',algorithmName);
  titleName = sprintf('%s-%s-P[p_1-p_4]-[%.3f,%.3f,%.3f,%.3f]',algorithmName,'Link[1-4]',updateP);
  
  if isUseP && isNormalize
    titleName = sprintf('%s-%s',xFigTitle,'Normalize');
    imgName = sprintf('%s-%s',xFigName,'Normalize');
  elseif isUseP && ~isNormalize
    titleName = sprintf('%s-%s',xFigTitle,'No-Normalize');
    imgName = sprintf('%s-%s',xFigName,'No-Normalize');
  end
  
  
  imgName = sprintf('%s-Noise:%s',imgName,noiseType);
  titleName = sprintf('%s-Noise:%s',titleName,noiseType);
  
  fileName = sprintf('%s%s',img_path,imgName);
  fprintf('%s\n',titleName);
  % Main iteration
  
  choices_1=iteration(1,T,Y0,N,isNormalize,noiseType);
  
  
  %draw and save img
  
  global oP;
  choices_2=ones(T+1,N).*oP;
  
  for i = 1:2:N*2
    lineName{i} =sprintf('p:%d',(i+1)/2);
    lineName{i+1} =sprintf('p%d*',(i+1)/2);
  end
  
  xFig = figure('name',imgName,'NumberTitle','off');
  set(xFig,'position',get(0,'screensize'));
  hold on;
  
  colorset =...
    [
    0.4588 ,0.6784,0.2314;
    0.3294 ,0.7412,0.9255;
    0.8353 ,0.3098,0.1137;
    0.9216 ,0.6941,0.1882;
    ];
  
  for i = 1:N
    plot([Y0(i);choices_1(:,i)],'DisplayName',char(lineName{i}),'LineWidth',lineWidth,'color',colorset(i,:));
    hold on;
    plot(choices_2(:,i),'-.','DisplayName',char(lineName{2*i}),'LineWidth',lineWidth,'color',colorset(i,:));
    hold on;
  end
  
  legh  =legend(lineName(1:end),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  title(titleName,'FontSize',20,'FontWeight','normal');
  hold off;
  
  saveas(xFig,strcat('img/',fileName,datestr(now,'HH-MM-SS')),'png');
  
end


function outChoices=iteration(t_b,t_e,Y0,N,isNormalize,noiseType)
  
  global X_BOUND;
  global updateP;
  global G0;
  global ETA0;
  global R_STAR;
  global NOISE_P;
  global G1;
  global G2;
  global ETA1;
  global ETA2;
  global PT;
  
  choices=zeros(t_e,N);
  STATE = [0,0];
  
  if t_b == 1
    x_0 = project(Y0,X_BOUND,N);
    %x0 feedback
    [G,ETA,STATE]  =  stochasticFunct(G0,G1,G2,ETA0,ETA1,ETA2,STATE,NOISE_P,PT,noiseType);
    gradientTmp = gradient(x_0,G,R_STAR,ETA);
    y = Y0 - 1*gradientTmp;
  end
  
  for t = t_b : t_e
    
    eta1 = t +1;
    % my choice
    choices(t,:) = project(y,X_BOUND,N);
    %y
    [G,ETA,STATE] =  stochasticFunct(G0,G1,G2,ETA0,ETA1,ETA2,STATE,NOISE_P,PT,noiseType);
    gradientTmp = binornd(1,updateP).*gradient(choices(t,:),G,R_STAR,ETA);
    
    if isNormalize
      y = y - (1 / eta1)*gradientTmp./updateP;
    else
      y = y - (1 / eta1)*gradientTmp;
    end
    
  end
  outChoices = choices;
end

% the difference of reward function U
function outX = gradient(x,G,R_STAR,eta)
  outX = (diag(G).*x'-R_STAR'.*(G*x' - x'.*diag(G)+ eta))';
end

% the reward function U
function uout = userCost(x,G,R_STAR,eta)
  uout = (1./(2*diag(G)).*(diag(G).*x'-R_STAR'.*(G*x' - x'.*diag(G) + eta)).^2)';
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

% types = {'Bernoulli','Log-normal','Markovian','No'};
function  [G,ETA,outState] = stochasticFunct(G0,G1,G2,ETA0,ETA1,ETA2,STATE,NOISE_P,PT,noiseType)
  switch noiseType
    case 'Bernoulli'
      [G,ETA,outState] = bernoulli(G1,G2,ETA1,ETA2,NOISE_P);
    case 'Log-normal'
      [G,ETA,outState] = logNormal(G1,G2,ETA1,ETA2,NOISE_P);
    case 'Markovian'
      [G,ETA,outState] =  markovian(G1,G2,ETA1,ETA2,STATE,NOISE_P,PT);
    case 'No'
      G = G0;
      ETA =ETA0;
      outState = [-1,-1];
    otherwise
      error('Noise type err');
  end
  
end

function [G,ETA,outState] = bernoulli(G1,G2,ETA1,ETA2,p)
  
  output = binornd(1,p(1));
  outputs =[output,1-output];
  G =  G1*outputs(1)+G2*outputs(2);
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
  if outputs(1) == 1
    s= S{1};
  else
    s=S{2};
  end
  G = s{1};
  ETA = s{2};
  %update probility
end

