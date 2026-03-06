function [t1 t2 t3 t4] = TEST_LATENCY(X0, ZZ, L, diag_std_dev, logPost)
s3 = tic;
parfor i=1:8 
    c(:,i) = eig(rand(1000)); 
end
t3 = toc(s3);

s4 = tic;
for i=1:8
    c(:,i) = eig(rand(1000)); 
end
t4 = toc(s4);

s = tic;
d = size(ZZ, 2);
for jj = 1:8
    XProp = X0 + (L/sqrt(d)*diag_std_dev.*ZZ(jj,:)); 
    logPiProp(:, jj) = logPost(exp(XProp)) + sum(XProp);
end
t1 = toc(s);
s2 = tic;
parfor (jj=1:8,8)
   XProp = X0 + (L/sqrt(d)*diag_std_dev.*ZZ(jj,:)); 
   logPiProp(:, jj) = logPost(exp(XProp)) + sum(XProp);
end
t2 = toc(s2);
