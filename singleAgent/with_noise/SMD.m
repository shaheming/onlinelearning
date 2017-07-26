function out = SMD(M)
  mkdir img;
  T = 2^(M)-1;
  
  
  global X_BOUND;
  X_BOUND = [0,1;0,2*pi];
  y0 = [5,3*pi];
  
  global eta;
  eta = 1;
  
  global myChoices;
  myChoices = zeros(T,2);
  
  
  %%%%%%%%%%%%%%%%%%
  % main function  %
  %%%%%%%%%%%%%%%%%%
  
  OGD_Primary(T,y0);
  
  %%%%%%%end%%%%%%%%
  
  
end



function  OGD_Primary(T,y0)
  
  global myChoices;
  
  disp('Begin Loop');
  fprintf('Iterate %d turns',T);
  
  iteration(1,T,y0);
  
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
  
  xFigName = sprintf('%s-%s','SGD','X');
  saveas(regFig,strcat('img/',xFigName),'png');
  
end


function iteration(t_b,t_e,y0)
  
  global X_BOUND;
  global myChoices;

  if t_b == 1
    x_0 = project(y0,X_BOUND);
    %x0 feedback
    y = y0 - 1*gradient(x_0);
  end
  
  for t = t_b : t_e
    
    eta1 = t + 1;
    % get x_t
    myChoices(t,:) = project(y,X_BOUND);
    
    y = y - (1 / eta1)*gradient(myChoices(t,:));
    
  end
  
  
end

function outX = gradient(x)
  r = x(1);
  theta = x(2);
  % the partial derivative add noise
  outX(1) = (3+ sin(5*theta) + cos(3*theta))*(10/3 * r - 3 * r^2)+ rand(1)-0.5;
  outX(2) =(5*cos(5*theta) - 3*sin(3*theta))*r^2*(5/3 - r)+rand(1)*2*pi -pi;
end

function uout = userLoss(x)
  theta = x(2);
  r = x(1);
  uout = (3+ sin(5*theta) + cos(3*theta))* r^2 *(5/3-r);
end


%the projection funciton
function x_t = project(y_t,X_BOUND)
  x_t = zeros(size(1,2));
  for i = 1:2
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


