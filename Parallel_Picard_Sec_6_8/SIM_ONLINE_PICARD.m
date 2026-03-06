clear all
load('data_20220606.mat');
L_init=0.005;
d=14;
N=100000;
K = 8;
% choose start_point5 to start in tail
% choose start_stat_point to start near HPD
%st=load('start_point5.mat');
st=load('start_stat_point.mat');
theta_init=st.ans(1:d);
X0 = log(theta_init);
diag_std_dev= [0.1677    0.1122    0.6662    1.1230    1.6727    0.2050    0.5516    0.8696    1.5909    0.6076    0.5595    1.1281    1.1259    1.6126];
evallogPost_fct=@(theta)evalLogPost(theta,Y,sigma_additive_noise,meanPrior,varPrior,time_resolution);
ZZ = randn(N + 2*K,d);
UU = rand(N + 2*K,1);
logPiX0 = evallogPost_fct(exp(X0)) + sum(X0);
[acc, ts, XXn, Bnew] = RWM(X0, ZZ, UU, L_init, diag_std_dev, evallogPost_fct);
[count, tp, XX_out] = ONLINE_PICARD(X0, ZZ, UU, N, K, L_init, diag_std_dev, evallogPost_fct);
acc
ts

tp

speedup = N/count
eff_speedup = ts/tp


cd ./sim_res/
save(['res' '.mat']);
cd ..