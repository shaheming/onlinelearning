function out = SMD(M)
%use doubling tricking to iterate
% M = 18; % 2 ^ 15 = 32768
% the maxiums turn will iterate T times;
T = 2^(M)-1; % avoid the last value to 0
% T = 50000;

global x_bound;
x_bound = [0,1;0,2*pi];
y0 = [2,8/6*pi];
% D and eta are used in reward function U
global eta;
eta = 1;

global myChoices;
myChoices = zeros(T,2);


%%%%%%%%%%%%%%%%%%
% main function  %
%%%%%%%%%%%%%%%%%%
  isDraw =false;

%    rng(2);
    OGD_Primary(T,y0,isDraw);

%%%%%%%end%%%%%%%%


end



function  OGD_Primary(T,y0,isDraw)

  global myChoices;
  mkdir img;
   
  disp('Begin Loop');
  fprintf('Iterate %d turns',T);
  
    iteration(1,T,y0,false);
  

   disp('End Loop'); 
   step = 100;

   r =0:1/step:1;
   theta = 0:2*pi/step:2*pi;
   [THETA,R]=meshgrid(theta,r);
   G = (3 + sin(5*THETA)+ cos(3*THETA)).*(R.^2).*(5/3-R);
   
   regFig = figure('name','SGD','NumberTitle','off');
   hold on;
   mesh(R.*cos(THETA), R.*sin(THETA), G);
   hold on;
   shading interp
   Z=zeros(1,T);
   for i = 1:T
     Z(i) = userLoss(myChoices(i,:));
   end
   
   Z = ones(1,T)*4;

   r=myChoices(:,1);
   theta = myChoices(:,2);
   plot3(r.*cos(theta),r.*sin(theta),Z,'Color','r');
   hold off;
   
   xFigName = sprintf('%s-%s-r0=%.3f-theta=%.3fend','SGD','X',y0(1),y0(2));
   saveas(regFig,strcat('img/',xFigName),'png');

end


function iteration(t_b,t_e,y0,doubling_flag)

  global x_bound;
  global myChoices;

  
  %theta

  
  % start at 0 OMG this is a serious problem !!! because in matlab for i =
  % i = 1:1 will iterate
  if t_b == 1

    x_0 = project(y0,x_bound);
    %x0 feedback
    y = y0 - 1*gradient(x_0);
  end
   
  for t = t_b : t_e
    %Z(1:end) = D * rand(size(Z,1),1);
    % thetas(t) = G(2:end) * Z;
    % my choice

    if doubling_flag 
      eta1 = t_b+1; 
    else
      eta1 = t + 10;
    end
    
    % get x_t
    myChoices(t,:) = project(y,x_bound);

    y = y - (1 / eta1)*gradient(myChoices(t,:));
        
  end
  

end

function u = updateExpert(experts,t,thetas)
  % note this with change with loss function
  u = 0;
end
% the difference of reward function U
function outX = gradient(x)
  r = x(1);
  theta = x(2);
  

  outX(1) = (3+ sin(5*theta) + cos(3*theta))*(10/3 * r - 3 * r^2);
  outX(2) =(5*cos(5*theta) - 3*sin(3*theta))*r^2*(5/3 - r);
end

% the reward function U
function uout = userLoss(x)
  theta = x(2);
  r = x(1);
  uout = (3+ sin(5*theta) + cos(3*theta))* r^2 *(5/3-r);
end

function uout = expertLoss(r,thetas,t)
  uout = 0;
end
%the projection funciton
function x_t = project(y_t,x_bound)
  for i = 1:2
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




function out = OGD_doubling(M,y1,isDraw)

  global experts;
  global myChoices;
%   global myRewards;
%   global expertsRewards;
%   global thetas;
   yout = y1;
  %rng('shuffle');
  for m = 1 : M

    [yout,regretsOut] = iteration(2^(m-1),2^(m)-1,yout,true);
%    [myChoices(m),experts(m), regrets(m)]=
%    regrets_div_t(m) = regrets(m)/2^m;
  end
 
  if isDraw
    figure('name','The value of Xt','NumberTitle','off','Position',[0,500,700,500]);
    plot(experts,'DisplayName','experts');
    hold on;
    plot(myChoices,'DisplayName','mychoice');
    legend('experts','mychoice');
    hold off;
    figure('name','The aluve of regret','NumberTitle','off','Position',[700,500,700,500]);
    hold on;
    %plot(regret_s'+ones(1,size(regret_s,1))*89);
    
    plot(regretsOut);
    hold off;
    
%     figure('name','Regret div t','NumberTitle','off','Position',[700,0,700,500]);
%     plot(regrets_div_t);
  end
%   diff(1:end) = regrets -(regret_s'-ones(1,size(regret_s,1))*89);

  out = regretsOut;
end

