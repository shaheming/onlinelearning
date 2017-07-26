function out = MLOGD_N_D_U(varargin)
  mkdir img MLOGD_N_D_U;
  global img_path;
  img_path ='MLOGD_N_D_U/';
  global algorithmName;
  algorithmName = 'MLOGD-N-D-U';
  
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
  B = 1; % BoundDelay
  
  M = varargin{1};
  T = 2^(M)-1; % avoid the last value to 0
  N = 4;
  X_BOUND = ones(N,2).*[0,100];
  Y0 = [0,0,0,0];
  %    Y0 = rand(1,N)*0.4;
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
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%
  % deal with parameter  %
  %%%%%%%%%%%%%%%%%%%%%%%%
  % noiseTypes = {'Bernoulli','Log-normal','Markovian','No'};
  updateP = [1,1,1,1];
  %updateP = [ 0.4259 ,0.8384 ,0.7423 ,0.0005];
  %delayTypes = {'nodelay','bound','linear','log','No','sqrt'};
  %delayTypes={'log','log','log','log'};
  %feedBackTypes = {'LOGD','Injection'};
  
  options = struct('noiseTypes',{{{'No'}}},...
    'delayTypes',{{{'no','no','no','no'}}},...
    'updateP',{num2cell([1,1,1,1],[1,2])},'feedBackTypes',{{{'LOGD'}}});
  
  %# read the acceptable names
  optionNames = fieldnames(options);
  %# count arguments
  if length(varargin) > 1
    nArgs = length(varargin(2:end));
    if round(nArgs/2)~=nArgs/2
      error('EXAMPLE needs propertyName/propertyValue pairs')
    end
  end
  
  for pair = reshape(varargin(2:end),2,[]) %# pair is {propName;propValue}
    inpName = pair{1}; %# make case insensitive
    
    if any(strcmp(inpName,optionNames))
      %# overwrite options. If you want you can test for the right class here
      %# Also, if you find out that there is an option you keep getting wrong,
      %# you can use "if strcmp(inpName,'problemOption'),testMore,end"-statements
      options(1).(inpName) = pair(2);
    else
      error('%s is not a recognized parameter name',inpName)
    end
  end
  
  
  
  if  options(1).updateP{:} == [1,1,1,1]
    isUseP = false;
  else
    updateP = options(1).updateP{:};
    isUseP = true;
  end
  
  noiseTypes =  options(1).noiseTypes{:};
  delayTypes = options(1).delayTypes{:};
  feedBackTypes =  options(1).feedBackTypes{:};
  isNormalize = false;
  
  
 
  for i = noiseTypes
    tic
    %rng(5);
    for j = feedBackTypes
      if isUseP
        OGD_Primary(T,Y0,N,char(i),char(j),isUseP,~isNormalize,delayTypes);
        OGD_Primary(T,Y0,N,char(i),char(j),isUseP,isNormalize,delayTypes);
      else
        OGD_Primary(T,Y0,N,char(i),char(j),isUseP,~isNormalize,delayTypes);
      end
      toc;
    end
    close all;
  end
 
  %%%%%%%end%%%%%%%%
  toc;
end



function    OGD_Primary(T,Y0,N,noiseType,feedBackType,isUseP,isNormalize,delayTypes)
  
  %%%%%%%%%%%%%%%%%%
  %   SET TITLE    %
  %%%%%%%%%%%%%%%%%%
  lineWidth = 0.8;
  global img_path;
  
  global updateP;
  
  if isUseP
    titleName = sprintf('%s-[p1-p4]%.3f-%.3f-%.3f-%.3f','MLOGD-UpdateP-Link[1-4]',updateP);
    imgName = sprintf('%s','MLOGD-UpdateP-Link[1-4]');
    if isNormalize
      titleName = sprintf('%s-%s',titleName,'Normalize');
      imgName = sprintf('%s-%s',imgName,'Normalize');
    else
      titleName = sprintf('%s-%s',titleName,'No-Normalize');
      imgName = sprintf('%s-%s',imgName,'No-Normalize');
    end
  else
    titleName = sprintf('%s','MLOGD-Link[1-4]');
    imgName =  sprintf('%s','MLOGD-Link[1-4]');
  end
  
  
  imgName = sprintf('%s-Noise:%s',imgName,noiseType);
  titleName = sprintf('%s-Noise:%s',titleName,noiseType);
  
  
  imgName = sprintf('%s %s',imgName,feedBackType);
  titleName = sprintf('%s %s',titleName,feedBackType);
  
  
  fileName = sprintf('%s%s',img_path,imgName);
  fprintf('%s\n',titleName);
  
  fprintf('Begin Loop\n');
  fprintf('Iterate %d turns\n',T);
  % Main iteration
  choices_1=iteration(1,T,Y0,N,isUseP,isNormalize,noiseType,feedBackType,delayTypes);
  fprintf('End Loop\n');
  %draw and save img
  global oP;
  choices_2=ones(T+1,N).*oP;
  
  for i = 1:2:N*2
    lineName{i} =sprintf('p:%d %s',(i+1)/2,char(delayTypes((i+1)/2)));
    lineName{i+1} =sprintf('p%d*',(i+1)/2);
  end
  
  xFig = figure('name',imgName,'NumberTitle','off');
  set(xFig,'position',get(0,'screensize'));
  
  hold on;
  
  % set the optimum prediction and agent prediciton to the same color.
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
  
  print(xFig,strcat('img/',fileName),'-dpng','-r100');
end


function outChoices=iteration(t_b,t_e,Y0,N,isUseP,isNormalize,noiseType,feedBackType,delayTypes)
  
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
  STATE = [0,0];
  %   noiseType ='Log-normal';
  for i=1:N
    heapCells{i}=MinHeap(t_e,ones(1,4)* inf);
    heapCells{i}.ExtractMin();
  end
  
  feedBackSums =zeros(1,N);
  choices=zeros(t_e,N);
  if t_b == 1
    x_0 = project(Y0,X_BOUND,N);
    %x0 feedback
    [G,ETA,STATE]  =  stochasticFunct(G0,G1,G2,ETA0,ETA1,ETA2,STATE,NOISE_P,PT,noiseType);
    
    if isUseP
      gradientTmp = binornd(1,updateP).*gradient(x_0,G,R_STAR,ETA);
      if isNormalize
        gradientTmp = gradientTmp./updateP;
      end
    else
      gradientTmp = gradient(x_0,G,R_STAR,ETA);
    end
    
    y = Y0 - 1*gradientTmp;
    
  end
  
  for t = t_b : t_e
    
    choices(t,:) = project(y,X_BOUND,N);
    
    gDelayedFeedBack(t,t_e,heapCells,choices,delayTypes,noiseType,B,G0,G1,G2,ETA0,ETA1,ETA2,R_STAR,PT,NOISE_P);
    
    feedBackTimes = getFeedBackTime(heapCells,N);
    % check if agent get feedback
    
    checkFeedBack = feedBackTimes==agentIndex.*(t+1);
    
    %if any of user get feedback
    if find(checkFeedBack == 1)
      % count grediant of every user
      [feedBackSums]=getFeedBackSum(t,heapCells,checkFeedBack,feedBackSums,feedBackType);
    end
    
    %y
    %gradientTmp = binornd(1,updateP).*gradient(choices(t,:),G,R_STAR,ETA);
    eta1 = t + 1;
    
    
    if isUseP
      gradientTmp = binornd(1,updateP).*feedBackSums;
      if isNormalize
        gradientTmp = gradientTmp./updateP;
      end
    else
      gradientTmp = feedBackSums;
    end
    
    y = y - (1 / eta1)*gradientTmp;
    
    switch  feedBackType
      case  'Injection'
      case  'LOGD'
        feedBackSums = zeros(1,N);
      otherwise
        error('feedBackType Error!')
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



% noiseTypes = {'Bernoulli','Log-normal','Markovian','No'};
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
      error('Noise type ERR');
  end
  
end

function [G,ETA,outState] = bernoulli(G1,G2,ETA1,ETA2,p)
  
  output = binornd(1,p(1));
  outputs =[output,1-output];
  G =  G1*outputs(1)+G2*outputs(2);
  ETA = ETA1*outputs(1)+ETA2*outputs(2);
  outState = [0,0];
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
  outState = [0,0];
end

function [G,ETA,outState] =  markovian(G1,G2,ETA1,ETA2,lastState,p,PT)
  if lastState == [0,0]
    output = binornd(1,p(1));
    outState =[output,1-output];
  elseif lastState == [1,0]
    output = binornd(1,PT(1,1));
    outState =[output,1-output];
  elseif lastState == [0,1]
    output = binornd(1,PT(2,1));
    outState =[output,1-output];
  end
  
  %   outState=outputs;
  %update probility
  if outState(1) == 1
    G = G1;
    ETA = ETA1;
  else
    G = G2;
    ETA = ETA2;
  end
end



function gDelayedFeedBack(t,t_e,heapCells,choices,delayTypes,noiseType,B,G0,G1,G2,ETA0,ETA1,ETA2,R_STAR,PT,NOISE_P)
  index = 1;
  persistent  STATE;
  if isempty(STATE) || t==t_e
    STATE  = [0,0];
  end
  [G,ETA,STATE] =  stochasticFunct(G0,G1,G2,ETA0,ETA1,ETA2,STATE,NOISE_P,PT,noiseType);
  gradients = gradient(choices(t,:),G,R_STAR,ETA);
  for i = delayTypes
    switch lower(char(i))
      case 'bound'
        [feedBackTime] = randi([1,B])+t;
      case 'linear'
        feedBackTime =  ceil(t/5 )+ t ;
      case 'log'
        if t ~= 1
          feedBackTime = ceil(t*ceil(log(t))/20) + t;
        else
          feedBackTime = t * 2;
        end
      case 'square'
        feedBackTime = t^2 + t;
      case 'sqrt'
        feedBackTime = ceil(sqrt(t)) + t;
      case 'no'
        feedBackTime = randi([1,1])+t;
      otherwise
        error('Delay type err');
    end
    
    
    heapCells{index}.InsertKey([feedBackTime,t,gradients(index),-1]);
    
    index = index + 1;
  end
end


function [feedBackTimes]=getFeedBackTime(heapCells,N)
  feedBackTimes = zeros(1,N);
  
  for i = 1:N
    out = num2cell( heapCells{i}.ReturnMin());
    [feedBackTime,~,~,~] = out{:};
    feedBackTimes(i) = feedBackTime;
  end
  
end

function [feedBackSums]=getFeedBackSum(t,heapCells,checkFeedBack,lastFeedBackSums,feedBackType)
  feedBackSums = zeros(size(checkFeedBack));
  feedBackCounts = zeros(size(checkFeedBack));
  % update agent who get feedback
  for i= find(checkFeedBack==1)
    while heapCells{i}.Count() > 0
      out = num2cell(heapCells{i}.ReturnMin());
      [feedBackTime,~,~,~] = out{:};
      if feedBackTime - 1 > t
        break;
      else
        out = num2cell(heapCells{i}.ExtractMin());
        [~,~,gradient,~] = out{:};
        feedBackSums(i)=feedBackSums(i)+ gradient;
        feedBackCounts(i) = feedBackCounts(i) + 1;
      end
    end
  end
  
  
  switch feedBackType
    case 'Injection'
      % if there is no update will follow the updation of the last iteration
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
