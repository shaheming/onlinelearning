function outX = gradientTest(x,G,R_STAR,eta)
  x_best = [0.016815,0.023363,0.031101, 0.016220];
  outX = sum( (diag(G).*x'-R_STAR'.*(G*x' - x'.*diag(G)+ eta))./diag(G).*(x-x_best)' );
end

function outXP = gradientTestP(x,G,R_STAR,eta,p)
  x_best = [0.016815,0.023363,0.031101, 0.016220];
  outXP = sum( (diag(G).*x'-R_STAR'.*(G*x' - x'.*diag(G)+ eta)).*(p'./diag(G)).*(x-x_best)' );
end
