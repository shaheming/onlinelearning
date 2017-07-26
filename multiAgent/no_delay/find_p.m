G = [6,1,2,1,;1,6,1,2;2,1,6,1;1,2,1,6];
eta = [0.1;0.2;0.3;0.1];
R_STAR = [0.5,0.5,0.5,0.5];
x_best=[0.016815,0.023363,0.031101, 0.016220];
rng(1);
x = rand(2*10^6,4)*100;
p = rand(2*10^6,4);

outX = sum( (diag(G).*x'-R_STAR'.*(G*x' - x'.*diag(G)+ eta)).*(p'./diag(G)).*(x-x_best)' );
pos= find(outX < 0);
dis(outP = p(pos',:));

