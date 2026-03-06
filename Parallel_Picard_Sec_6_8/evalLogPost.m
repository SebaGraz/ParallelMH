function logPost=evalLogPost(theta,Y,sig_add_noise,meanPrior,varPrior,time_resolution)
d=max(size(theta));
param=translateToParam(theta);
logPost=0;
try
    mu=solveODE(param,time_resolution);
catch
    logPost=-1e15;
end

if logPost==0
nb_individuals=size(Y,3);
loglike=0;
for k=1:nb_individuals
    loglike=loglike-(1/(2*sig_add_noise(1)^2))*norm(Y(1,:,k)-mu(1,:))^2 ...
    -(1/(2*sig_add_noise(2)^2))*norm(Y(2,:,k)-mu(2,:)).^2 ...
    -(1/(2*sig_add_noise(3)^2))*norm(Y(3,:,k)-mu(3,:)).^2 ...
    -(1/(2*sig_add_noise(4)^2))*norm(Y(4,:,k)-mu(4,:)).^2;
end
logPost=loglike+log(mvnpdf(theta,meanPrior(1:d),varPrior(1:d)));
end