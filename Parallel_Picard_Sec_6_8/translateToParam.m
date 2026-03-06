function param=translateToParam(theta)

p=size(theta,2);
theta=[theta zeros(1,14-p)];

param.a1=theta(1);
if p>=2
    param.a2=theta(2);
else
    param.a2 = 1.444183287717442;
end
if p>=3
    param.d2=theta(3);
else
    param.d2 = 0.295260097294696;
end
if p>=4
    param.kappaVV=theta(4);
else
    param.kappaVV =  0.054058; 
end
if p>=5
    param.deltaVV=theta(5);
else
    param.deltaVV =  2.484864086006999;  
end
if p>=6
    param.alphaVV=theta(6);
else
    param.alphaVV =  1.120771;   
end

if p>=7
    param.omegaVV=theta(7);
else
    param.omegaVV =  40.282073356365977;   
end

if p>=8
    param.kappaVSV=theta(8);
else
    param.kappaVSV =  0.065563768;   
end

if p>=9
    param.deltaVSV=theta(9);
else
    param.deltaVSV =  11.003239918554174;   
end

if p>=10
    param.alphaVSV=theta(10);
else
    param.alphaVSV =  1.131130932;   
end

if p>=11
    param.omegaVSV=theta(11);
else
    param.omegaVSV =  38.685206250628028;   
end

if p>=12
    param.kp=theta(12);
else
    param.kp =  9.23124604834137;   
end


if p>=13
    param.kqs=theta(13);
else
    param.kqs =  0.06483950327214;   
end


if p>=14
    param.Psi12=theta(14);
else
    param.Psi12 =  0.000114983672183;   
end




% 
% param.a1 = theta(1);  
% param.a2 = theta(2);
% param.d2 = theta(3);
%  param.kappaVV=theta(4);
%  param.deltaVV=theta(5); 
%  param.alphaVV=theta(6); 
%  param.omegaVV=theta(7); 
%  param.kappaVSV=theta(8); 
%  param.deltaVSV=theta(9); 
%  param.alphaVSV=theta(10); 
%  param.omegaVSV=theta(11);
%  param.kp=theta(12);
%  param.kqs=theta(13);
%  param.Psi12=theta(14);
% 
% 
% param.a1 =  1.662718177543053;  
% param.a2 = 1.444183287717442;
% param.d2 = 0.295260097294696;
% param.kappaVV =  0.054058; 
% param.deltaVV =  2.484864086006999; 
% param.alphaVV =  1.120771; 
% param.omegaVV =  40.282073356365977;  
% param.kappaVSV =  0.065563768;
% param.deltaVSV =  11.003239918554174; 
% param.alphaVSV =  1.131130932;
% param.omegaVSV =  38.685206250628028;
% param.kp = 9.23124604834137;
% param.kqs = 0.06483950327214; 
% param.Psi12 = 0.000114983672183;  