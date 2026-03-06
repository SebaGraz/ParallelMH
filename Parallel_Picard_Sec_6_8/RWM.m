function [ar, ts, XX, B] = RWM(X0, ZZ, UU, L, diag_std_dev, NlogPost)
[K d] = size(ZZ);
NlogPi = NlogPost(exp(X0)) + sum(X0);
X = X0;
B = zeros(K,1);
XX = repmat(X0, K+1,1);
acc = 0;
s = tic;
for j = 1:K
    j
    Xp = X + (L/sqrt(d)*diag_std_dev.*ZZ(j,:)); 
    tic;
    NlogPiProp = NlogPost(exp(Xp)) + sum(Xp);
    toc
    if NlogPi - NlogPiProp >= log(UU(j))
        B(j) = 1;
        NlogPi = NlogPiProp;
        X = Xp;
        acc = acc + 1;
    end
    XX(j+1,:) = X;
    ts = toc(s);
    ar = acc/K;
end
