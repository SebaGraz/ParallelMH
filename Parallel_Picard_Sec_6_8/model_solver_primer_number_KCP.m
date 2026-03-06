function [solTreat PAN] = model_solver_primer_number_KCP(PA,totaltime,Initial_cond_day_6)
%Function to run the solver for the JITC paper

%% Calculate the initial conditions
IStrain1IC = 0;

VStrain1IC = 0;
IStrain2IC = 0;
VStrain2IC = 0;

InitialConditions = [Initial_cond_day_6,VStrain2IC,IStrain2IC];

Mean_dif = (4.6754-3.081693813811469);
PA_Kcp_old = PA.Kcp;
PA_Kcp_new = PA_Kcp_old -Mean_dif;
    
%% Solve the DDE and calculate tumour burden
[solTreat] = WaresDistributedImmunityResistantSolver(totaltime,InitialConditions,PA); %Simulate model 
      
PAN = PA.N;
function [sol] = WaresDistributedImmunityResistantSolver(totaltime,IC,PA) %DDE model without therapy
opts = odeset('RelTol',1e-8,'AbsTol',1e-8,'MaxStep',1e-2);%,'Events',@EventsViralMonths1);
sol = ode15s(@ViralOncologyParameterFit,totaltime,IC,opts);

function dydt = ViralOncologyParameterFit(t,y)


if t>=PA.ViralStartTimeBooster
    PA.Kcp  = PA_Kcp_new;
end

%Quiescent cells
dydt(1) = 2.*(1-PA.nu).*PA.TransitRate.*y(PA.N+4)-(PA.a1+PA.d1+ psiQ(y(PA.N+6),y(1),PA) ).*y(1); 
%G1 Cells
dydt(2) = PA.a1.*y(1)-( PA.a2+PA.d2+ PA.kappaPrimer.*Infection(y(4),PA)+ PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA)+ psiS(y(PA.N+6),y(2),PA) ).*y(2); 
%Infected Cells PRIMER
dydt(3) =  -PA.deltaPrimer.*y(3)+ PA.kappaPrimer.*Infection(y(4),PA).*(y(2)+y(PA.N+7)+y(PA.N+9)+y(PA.N+PA.N+10) ); % PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA).*(y(2)+Sbar); % Infected cells
%Virions PRIMER
dydt(4) = ViralStrainPrimerDose(PA,t) + PA.alphaPrimer.*PA.deltaPrimer.*y(3)-PA.omegaPrimer.*y(4)- PA.kappaPrimer.*Infection(y(4),PA).*(y(2)+y(PA.N+7)+y(PA.N+9)+y(PA.N+PA.N+10)); % PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA ).*(y(2)+Sbar); % Virions
%Writing ODE for first transit compartment
 dydt(5) = PA.a2.*y(2) - PA.TransitRate.*y(5) - ( PA.d3Hat+PA.kappaPrimer.*Infection(y(4),PA) +PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA) + psiS(y(PA.N+6),y(5),PA) ).*y(5);
for jj = 6:PA.N+4
    dydt(jj) = PA.TransitRate.*(y(jj-1)-y(jj)) - ( PA.d3Hat +PA.kappaPrimer.*Infection(y(4),PA) +PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA)+ psiS(y(PA.N+6),y(jj),PA) ).*y(jj); %Transit compartment ODEs
end
%Immune cytokine 
dydt(PA.N+5) =  CProd( psiS(y(PA.N+6),y(2),PA).*(y(2)+y(PA.N+7))+PA.deltaPrimer.*y(3)+PA.deltaBooster.*y(PA.N+PA.N+12)+psiQ(y(PA.N+6),y(1),PA).*y(1),PA,t) - PA.kel.*y(PA.N+5)+Dose(PA,t); 
%Phagocytes
dydt(PA.N+6) =  PA.Kcp.*y(PA.N+5)./(PA.P12+y(PA.N+5)) - PA.gammaP.*y(PA.N+6); 
%ODE for total number of cells in cell cycle
dydt(PA.N+7) = PA.a2.*y(2) - (PA.d3Hat+PA.kappaPrimer.*Infection(y(4),PA)+PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA)+psiS(y(PA.N+6),y(2),PA) ).*y(PA.N+7) - (PA.TransitRate./PA.a2).*y(PA.N+4);
% Resistant compartments
%Resistant Quiescent
dydt(PA.N+8) =   2.*PA.nu.*PA.TransitRate.*y(PA.N+4) + 2.*PA.TransitRate.*y(PA.N+PA.N+9)-(PA.a1R+PA.d1R).*y(PA.N+8); %Resistant quiescence DDE
%Resistant G1
dydt(PA.N+9) =   PA.a1R.*y(PA.N+8)-(PA.a2R+PA.d2R+ PA.kappaPrimer.*Infection(y(4),PA)+ PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA) ).*y(PA.N+9); %Susceptible resistant cells
%Resistant First transit
dydt(PA.N+10) =  PA.a2R.*y(PA.N+9) - PA.TransitRate.*y(PA.N+10) - (PA.d3HatR+PA.kappaPrimer.*Infection(y(4),PA)+PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA)).*y(PA.N+10); %Susceptible resistant first compartment
for jj = PA.N+11:PA.N+PA.N+9
    dydt(jj) =  PA.TransitRate.*(y(jj-1)-y(jj)) - (PA.d3HatR +PA.kappaPrimer.*Infection(y(4),PA)+PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA)   ).*y(jj); %Resistant Transit compartment ODEs
end
%DE for total resistant cells
dydt(PA.N+PA.N+10) =   PA.a2.*y(PA.N+9) - (PA.d3Hat+PA.kappaPrimer.*Infection(y(4),PA) +PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA)).*y(PA.N+PA.N+10) - (PA.TransitRate./PA.a2).*y(PA.N+PA.N+9);
% Virions BOOSTER
dydt(PA.N+PA.N+11) =    ViralStrainBoosterDose(PA,t) + PA.alphaBooster.*PA.deltaBooster.*y(PA.N+PA.N+12)-PA.omegaBooster.*y(PA.N+PA.N+11)- PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA).*(y(2)+y(PA.N+7)+y(PA.N+9)+y(PA.N+PA.N+10)); %
% Infected Cells Booster
dydt(PA.N+PA.N+12) = -PA.deltaBooster.*y(PA.N+PA.N+12)+ PA.kappaBooster.*Infection(y(PA.N+PA.N+11),PA).*(y(2)+y(PA.N+7)+y(PA.N+9)+y(PA.N+PA.N+10) );
dydt = dydt';
end

function [value,isterminal,direction] = EventsViralMonths1(t,y,Z)

PA.TotalCells = sum(InitialConditions);
value(1) = y(1)+ y(2) +y(PA.N+7)  - 2.*PA.TotalCells; 

isterminal(1) = 0;   % 1 = End the integration
direction(1) = 0;   % Positive direction only

value(2)= y(1)+ y(2)+ y(PA.N+7) - 5e5; %PA.TotalCells.*2^(13+PA.DeathTime);  %What we are setting to 0 
isterminal(2) = 1;   % 1 = End the integration
direction(2) = 0;   % Positive direction only

end
end


function g = psiQ(P,Q,PA)% clearance of quiescent cells by immune system
 g = P.*(PA.kp./(1+PA.kq.*Q));
end

function h = psiS(P,S,PA) % clearance of susceptible cells by immune system
h =  P.*(PA.kp./(1+PA.ks.*S));
end

function y = CProd(a,PA,t)
%Cytokine Production
    if t<PA.ViralStartTimePrimer
        rho = 1;
    elseif t<PA.ViralStartTimeBooster
        rho = 0;
    else
        rho = 0.955019371120055;
    end
        y = max(PA.CprodHomeo+ rho*(PA.CprodMax-PA.CprodHomeo).*(a./(PA.C12+a)),0);
end

function f = Infection(V,PA) %Contact function
    if V > 1e-10
        f = V./(PA.eta12+V);
    else
        f = 0;
    end
end

% Immunotherapy dosing function
function DoseWares = Dose(PA,t)
DoseVec = zeros(1,PA.AdminNumber);
TAdmin = PA.StartTime+(0:PA.AdminNumber-1).*PA.Offset-t;
for nn = 1:PA.AdminNumber
    if TAdmin(nn) < 0
    DoseVec(nn) = (PA.AvailFrac.*PA.kabs.*PA.Admin(nn))./PA.Vol.*exp(PA.kabs.*TAdmin(nn));
    else
        DoseVec(nn) = 0;
    end
end
DoseWares = ones(1,PA.AdminNumber)*DoseVec';
end

% Viral dosing function
function DoseViralStrainPrimer = ViralStrainPrimerDose(PA,t)
ViralDoseVecStrainPrimer = zeros(1,PA.ViralAdminNumberStrainPrimer);
TAdminViral = PA.ViralStartTimePrimer+(0:PA.ViralAdminNumberStrainPrimer-1).*PA.ViralOffset-t;
for nn = 1:PA.ViralAdminNumberStrainPrimer
    if TAdminViral(nn) < 0
    ViralDoseVecStrainPrimer(nn) = (PA.Viralkabs.*PA.ViralAdminStrainPrimer(nn).*PA.ViralAvailFrac)./PA.Vol.*exp(PA.Viralkabs.*TAdminViral(nn));
    else
        ViralDoseVecStrainPrimer(nn) = 0;
    end
end
DoseViralStrainPrimer = ones(1,PA.ViralAdminNumberStrainPrimer)*ViralDoseVecStrainPrimer';

end

function DoseViralStrainBooster = ViralStrainBoosterDose(PA,t)
ViralDoseVecStrainBooster = zeros(1,PA.ViralAdminNumberStrainBooster);
TAdminViral = PA.ViralStartTimeBooster+(0:PA.ViralAdminNumberStrainBooster-1).*PA.ViralOffset-t;
for nn = 1:PA.ViralAdminNumberStrainBooster
    if TAdminViral(nn) < 0
    ViralDoseVecStrainBooster(nn) = (PA.Viralkabs.*PA.ViralAdminStrainBooster(nn).*PA.ViralAvailFrac)./PA.Vol.*exp(PA.Viralkabs.*TAdminViral(nn));
    else
        ViralDoseVecStrainBooster(nn) = 0;
    end
end
DoseViralStrainBooster = ones(1,PA.ViralAdminNumberStrainBooster)*ViralDoseVecStrainBooster';

end

end