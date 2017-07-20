 G1=[1,3;3,1];
 G2=[2,1;1,2];
 G_E = 0.25 * G1 + 0.75*G2;
 
 ETA1=[0.1;0.2];
 ETA2=[0.15;0.05];
 ETA_E = 0.25 * ETA1 + 0.75*ETA2;

 %binornd
 output = binornd(1,0.25);
 outputs =[output,1-output];
 G_O =  G1*outputs(1)+G2*outputsp(2);
 output = binornd(1,0.25);
 outputs =[output,1-output];
 ETA_O = ETA1*outputs(1)+ETA2*outputsp(2);
 
 %log normal G
 G=lognrnd(G_E,[1,1;1,1]);
 %log normal eta
 ETA = lognrnd(ETA_E,1);
 

 %markovian chain
 p = [1/4,3/4];
 PT=[2/5,3/5;1/5,4/5];
 s1={G1,ETA1};
 s2={G2,ETA2};
 S = {s1,s2};
 %begain
 output=binornd(1,p(1));
 outputs =[output,1-output];
 if outputs(1) == 1
   s= S{1};
 else
   s=S{2};
 end
 G = s{1};
 ETA = s{2};
 s =  s1*outputs(1)+s2*outputsp(2);
 %update probility
 p = p*PT;
 %end



 

 
 