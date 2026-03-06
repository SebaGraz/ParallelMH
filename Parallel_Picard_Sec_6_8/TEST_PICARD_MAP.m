clear all
load('data_20220606.mat');
L_init=0.008;
d=14;
K=8;
% choose start_point5 to start in tail
% choose start_stat_point to start near HPD
%st=load('start_point5.mat');
st=load('start_stat_point.mat');
theta_init=st.ans(1:d);
X0 = log(theta_init);
diag_std_dev=[0.1677    0.1122    0.6662    1.1230    1.6727    0.2050    0.5516    0.8696    1.5909    0.6076    0.5595    1.1281    1.1259    1.6126];
evallogPost_fct=@(theta)evalLogPost(theta,Y,sigma_additive_noise,meanPrior,varPrior,time_resolution);
ZZ = randn(K,d);
UU = rand(K,1);
[acc XXtrue Btrue] = RWM(X0, ZZ, UU, L_init, diag_std_dev, evallogPost_fct);

% CHECK IF TRUE TRAJECTORY PICARD CONVERGES AT THE FIRST ITERATION
% disp("check 1")
% B = Bnew;
% logPiX0 = evallogPost_fct(exp(X0)) + sum(X0);
% [acc gain XXn2 Bnew2] = PICARD_MAP(logPiX0, XXn, ZZ, UU, L_init, B, diag_std_dev, evallogPost_fct);
% [K gain]
% [Bnew - Bnew2]
% [XXn2 - XXn] 


% RUN THE PICARD MAP TO CHECK ITERATIVELY
disp("check 2")
logPiX0 = evallogPost_fct(exp(X0)) + sum(X0);
XX = repmat(X0, K+1,1);
B = zeros(K, 1);
% step 1
[acc gain XX B logPiX] = PICARD_MAP(logPiX0, XX, ZZ, UU, L_init, B, diag_std_dev, evallogPost_fct);
[B Btrue]

%step 2
B = B(gain+1:end);
XX
[acc gain XX B] = PICARD_MAP(logPiX0, XX, ZZ, UU, L_init, B, diag_std_dev, evallogPost_fct);
[B Btrue]
