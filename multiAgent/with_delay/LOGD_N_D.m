function out = LOGD_N_D(M)
  %use doubling tricking to iterat
  
  mkdir img LOGD_N_D;
  global img_path;
  img_path ='LOGD_N_D/';
  global algorithmName;
  algorithmName = 'SGD-N-U-D';
  
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
  global B;
  B = 100; % BoundDelay
  % the maxiums turn will iterate T times;
  T = 2^(M)-1; % avoid the last value to 0
  N = 4;
  X_BOUND = ones(N,2).*[-400,400];
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
  
  updateP = [1,1,1,1];
  %updateP = [ 0.4259 ,0.8384 ,0.7423 ,0.0005];
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  types = {'Bernoulli','Log-normal','Markovian','No'};
  types = {'No'};
  isRegular = false;
  feedBackTypes = {'Injection','LOGD'};
  fprintf('Begin Loop\n');
  fprintf('Iterate %d turns\n',T);
  for i = types
    tic
    rng(5);
    for j = feedBackTypes
      OGD_Primary(T,Y0,N,char(i),char(j),isRegular);
    end
  end
  fprintf('End Loop\n');
  %%%%%%%end%%%%%%%%
  toc;
  pause(3);
  close all;
end



function  OGD_Primary(T,Y0,N,noiseType,feedBackType,isRegular)
  
  %%%%%%%%%%%%%%%%%%
  %   SET TITLE  %
  %%%%%%%%%%%%%%%%%%
  lineWidth = 1;
  global algorithmName;
  global img_path;
  global updateP;
  imgName = sprintf('%s-%s',algorithmName,datestr(now, 'HH-MM-SS'));
  titleName = sprintf('%s-%s-P[p_1-p_4]-[%.3f,%.3f,%.3f,%.3f]',algorithmName,'Link[1-4]',updateP);
  imgName = sprintf('%s-Noise:%s',imgName,noiseType);
  titleName = sprintf('%s-Noise:%s',titleName,noiseType);
  
  if isRegular
    imgName = sprintf('%s %s',imgName,'Normalize');
    titleName = sprintf('%s %s',titleName,'Normalize');
  else
    imgName = sprintf('%s %s',imgName,'No-Normalize');
    titleName = sprintf('%s %s',titleName,'No-Normalize');
  end
  
  imgName = sprintf('%s %s',imgName,feedBackType);
  titleName = sprintf('%s %s',titleName,feedBackType);
  
  fileName = sprintf('%s%s',img_path,imgName);
  fprintf('%s\n',titleName);
  % Main iteration
  
  
  choices_1=iteration(1,T,Y0,N,isRegular,noiseType,feedBackType);
  
  
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
  
  for i = 1:N
    plot([Y0(i);choices_1(:,i)],'DisplayName',char(lineName{i}),'LineWidth',lineWidth);
    hold on;
    plot(choices_2(:,i),'-.','DisplayName',char(lineName{2*i}),'LineWidth',lineWidth);
    hold on;
  end
  
  legh  =legend(lineName(1:end),'Location','best','EdgeColor','w');
  legh.LineWidth = 2;
  legh.FontSize = 20;
  title(titleName,'FontSize',20,'FontWeight','normal');
  hold off;
  
  saveas(xFig,strcat('img/',fileName),'png');
  
end


function outChoices=iteration(t_b,t_e,Y0,N,isRegular,noiseType,feedBackType)
  
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
  global B;
  
  %   feedBackImg = zeros(4,t_e);
  agentIndex = ones(1,N);
  originTimesForNoOverlapDelay = ones(1,N);
  %   delayType = 'bound';
%   delayType = 'log';
  delayType ='linear';
  for i=1:N
    heapCells{i}=MinHeap(B+1,ones(1,4)* inf);
    heapCells{i}.ExtractMin();
  end
  
  feedBackSums =zeros(1,N);
  choices=zeros(t_e,N);
  STATE = [0,0];
  if t_b == 1
    x_0 = project(Y0,X_BOUND,N);
    %x0 feedback
    [G,ETA,STATE]  =  stochasticFunct(G0,G1,G2,ETA0,ETA1,ETA2,STATE,NOISE_P,PT,noiseType);
    y = Y0 - 1*gradient(x_0,G,R_STAR,ETA);
    
  end
  
  for t = t_b : t_e
    gDelayedFeedBack(B,t,heapCells,N,delayType);
    
    feedBackTimes = getFeedBackTime(heapCells,N,delayType,originTimesForNoOverlapDelay);
    % check if agent get feedback
    
    eta1 = t +100;
    % my choice
    choices(t,:) = project(y,X_BOUND,N);
    
    checkFeedBack = feedBackTimes==agentIndex.*(t+1);
    % feedBackImg(:,t) = checkFeedBack';
    updateAgentPos = find(checkFeedBack == 1);
    if updateAgentPos
      % count grediant of every user
      [feedBackSums]=getFeedBackSum(t,heapCells,checkFeedBack,choices,G,R_STAR,ETA,...
        feedBackSums,feedBackType,delayType,originTimesForNoOverlapDelay);
      
       originTimesForNoOverlapDelay(updateAgentPos) = originTimesForNoOverlapDelay(updateAgentPos)+1;
    end
    
    %y
    [G,ETA,STATE] =  stochasticFunct(G0,G1,G2,ETA0,ETA1,ETA2,STATE,NOISE_P,PT,noiseType);
    %gradientTmp = binornd(1,updateP).*gradient(choices(t,:),G,R_STAR,ETA);
    
    y = y - (1 / eta1)*feedBackSums;
    
    switch  feedBackType
      case  'Injection'
      otherwise
        feedBackSums = zeros(1,N);        
    end
  
  end
  outChoices = choices;
  %   feedBackImg = feedBackImg.*(1:4)';
  %   figure;
  %   for i = 1:N
  %
  %   scatter(1:t_e,feedBackImg(i,:),0.5);
  %   hold on;
  %   end
end

% the difference of reward function U
function outX = gradient(x,G,R_STAR,eta)
  outX = (diag(G).*x'-R_STAR'.*(G*x' - x'.*diag(G)+ eta))';
end

% the reward function U
function uout = userCost(x,G,R_STAR,eta)
  uout = (1./(2*diag(G)).*(diag(G).*x'-R_STAR'.*(G*x' - x'.*diag(G) + eta)).^2)';
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
  outState = outputs;
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



function gDelayedFeedBack(B,t,heapCells,N,feedBackType)
  switch lower(feedBackType)
    case 'bound'
      [feedBackTimes] = randi([1,B],1,N)+t;
      for i = 1:N
        heapCells{i}.InsertKey([feedBackTimes(i),t,-1,-1]);
      end
  end
end

function [feedBackTimes]=getFeedBackTime(heapCells,N,delayType,originTimes)
  feedBackTimes = zeros(1,N);
  switch delayType
    case 'bound'
      for i = 1:N
        out = num2cell( heapCells{i}.ReturnMin());
        [feedBackTime,~,~,~] = out{:};
        feedBackTimes(i) = feedBackTime;
      end
    case 'linear'
      feedBackTimes =  originTimes +originTimes ;
    case 'log'
      if originTimes(1)~= 1
        feedBackTimes = originTimes.*ceil(log2(originTimes)) + originTimes;
      else
        feedBackTimes = originTimes * 2;
      end
    case 'square'
      feedBackTimes = originTimes.^2 + originTimes;
  end
end

function [feedBackSums]=getFeedBackSum(t,heapCells,checkFeedBack,choices,G,R_STAR,ETA,lastFeedBackSums,feedBackType,delayType,originTimes)
  feedBackSums = zeros(size(checkFeedBack));
  feedBackCounts = zeros(size(checkFeedBack));
  switch delayType
    case 'bound'
      for i= find(checkFeedBack==1)
        while heapCells{i}.Count() > 0
          out = num2cell(heapCells{i}.ReturnMin());
          [feedBackTime,~,~,~] = out{:};
          if feedBackTime - 1 > t
            break;
          else
            out = num2cell(heapCells{i}.ExtractMin());
            [~,originTime,~,~] = out{:};
            gradients = gradient(choices(originTime,:),G,R_STAR,ETA);
            feedBackSums(i)=feedBackSums(i)+ gradients(i);
            feedBackCounts(i) = feedBackCounts(i) + 1;
          end
        end
      end
    otherwise % linear log squr
      if find(checkFeedBack==1)
        gradients = gradient(choices(originTimes(1),:),G,R_STAR,ETA);
        feedBackSums = gradients;
        feedBackCounts = feedBackCounts + 1;
      end
  end
  
  switch feedBackType
    case 'Injection'
      % if there is no update will follow the update of the last iteration
      pos = find(checkFeedBack==0);
      posUpdate = find(checkFeedBack==1);
      feedBackSums(posUpdate) = feedBackSums(posUpdate)./feedBackCounts(posUpdate);
      feedBackSums(pos)=lastFeedBackSums(pos);
  end
end


%the projection funciton
function x_t = project(y_t,X_BOUND,N)
  x_t = zeros(1,N);
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

