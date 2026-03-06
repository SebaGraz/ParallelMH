%Script to calculate the tumour burden for various numbers of priming doses.
%Up to 7 different priming doses with the tumour burden being the integral
%from the final priming dose until 15 days later or the integral from the
%first priming dow until 15 days after the last priming dose

%% Load parameters
%Immunotherapy (Turned off)
PA.AdminNumber = 1; %Need this for the code to run, even if no immunotherapy is actually administered
PA.StartTime = 0;
PA.Offset = 1; %Dosed every day 
PA.Admin = 0; % Administer no immunotherapy 
PA.Vol = 7; %volume of absorption
PA.kabs = 6.6311; % absorption rate
PA.AvailFrac =0.85; 

% Viral therapy- The dosing strategy is loaded in the loop
%PA.ViralStartTime = 0;
PA.ViralOffset = 1; %Dosed at a maximum of once daily
PA.Viralkabs = 20; % absorption rate
PA.ViralAvailFrac = 1;

%Set baseline model parameters
PA.a1 =  1.662718177543053; %4T1 tumour growth parameters
PA.a2 = 1.444183287717442;%4T1 tumour growth parameters
PA.d1 =  0;
PA.d2 = 0.295260097294696;%4T1 tumour growth parameters
PA.d3 = PA.d2;
PA.tau = 33.7/24 - 1/PA.a2; % From Sato 2016 mean intermitotic time - the expected time in G1- gives a lower bound on a_2. 
PA.IntermitoticSD = 6.7/24;  % From Sato 2016 SD on intermitotic time

%Distribution Specific Parameters
PA.N = round(PA.tau.^2./PA.IntermitoticSD^2); % round((33.7./24)^2./(6.7./24));
PA.TransitRate = PA.N./PA.tau; % Transit rate across compartments
PA.d3Hat = PA.N./PA.tau.*(exp(PA.d3.*PA.tau./(PA.N+1))-1);
PA.d3HatR = PA.d3Hat;

%Viral parameters VV Strain
PA.kappaPrimer =  0.054058; %Infection rate [1/month]   ;
PA.deltaPrimer =  2.484864086006999; % Death rate of infected cells [1/month] ;
PA.alphaPrimer =  1.120771;  %Dying infected cell to virion conversion [virions/cell];
PA.omegaPrimer =  40.282073356365977;   %Virion decay rate [1/month].
PrimerDose = 1e2; %Amount of Primer (VV) administered

%Viral parameters VSV Strain (BOOSTER)
PA.kappaBooster =  0.065563768; %Infection rate [1/month]   ;
PA.deltaBooster =  11.003239918554174; % Death rate of infected cells [1/month] ;
PA.alphaBooster =  1.131130932;  %Dying infected cell to virion conversion [virions/cell];
PA.omegaBooster =  38.685206250628028;   %Virion decay rate [1/month].
BoosterDose = 1e3; %Amount of Boosters (VSV) administered

%Strain independnent contact rate
PA.eta12 =  0.510538277701167;  % virion half effect contact rate, Strain independent

% Cytokine Parameters
PA.CprodHomeo = 0.00039863;   %Homeostatic cytokine production [ng/mL/month]
PA.CprodMax =  1.429574637713578;   % Maximal cytokine production rate [ng/mL/month]
PA.C12 = 0.739376299393775; % Half effect in cytokine production [cells/month]
PA.kel = 0.16139;   % elimination rate of cytokine [1/month]

%Immune Parameters
PA.kp = 9.23124604834137;    % contact rate with phagocytes [1/month]
PA.kq = 0.06483950327214;  % factor in denominator of Q phagocytosis term [Unitless] 
PA.ks = PA.kq; % factor in denominator of S phagocytosis term [Unitless]
PA.P12 = 0.000114983672183;    % Half effect in cytokine driven phagocyte production [ng/mL]
PA.gammaP = 0.35; % From Barrish 2017 PNAS elimination rate of phagocyte [1/month]
PA.Kcp = 4.6754;  % From Barrish 2017 PNAS conversion of cytokine into phagocyte [cells/ng/mL]
PA.KcpBooster = 3.081693813811469; % Kcp when VV has primed the tumour microenvironment and VSV is injected

%Resistant Parameters
PA.nu = 1e-10;  %  Mutation probability
PA.a1R = PA.a1;
PA.a2R = PA.a2;
PA.d1R = PA.d1;
PA.d2R = PA.d2;
PA.d3R = PA.d3;

PA.a1 =  1.662718177543053; %Transit Rate Quiescent [1/month]
PA.a2 = 1.444183287717442;%1.758233712464858.*30  ; %Transit Rate G1 [1/month]
PA.d1 =  0; %Death Rate Quiescent [1/month]
PA.d2 =  0.295260097294696;%0.539325116600707.*30; %Death Rate G1 [1/month]
PA.d3 = PA.d2; %Death rate Mitotic [1/month]
PA.tau = 33.7/24-1/PA.a2;%(33.7/24 - 30/PA.a2)./30; % From Sato 2016 mean intermitotic time - the expected time in G1- gives a lower bound on a_2. 
PA.IntermitoticSD = 6.7/24;%/30;  % From Sato 2016 SD on intermitotic time
PA.kp =  9.23124604834137; %Infection rate [1/month]   ;
PA.kq =  0.064839;%4.962123414821151.*30; % Death rate of infected cells [1/month] ;
PA.ks = PA.kq; % factor in denominator of S phagocytosis term [Unitless]
PA.Kcp = 4.6754;%.*30;  % From Barrish 2017 PNAS conversion of cytokine into phagocyte [cells/ng/mL]

% Resistant parameters
PA.a1R = PA.a1;
PA.a2R = PA.a2;
PA.d1R = PA.d1;
PA.d2R = PA.d2;
PA.d3R = PA.d3;

PA.TransitRate = PA.N./PA.tau; %must recalculate transit rate, as the delay varies
PA.CStar = PA.CprodHomeo./PA.kel;
PA.PStar = (1./PA.gammaP).*(PA.Kcp.*PA.CStar./(PA.P12+PA.CStar));
PA.d3Hat = PA.N./PA.tau.*(exp(PA.d3.*PA.tau./(PA.N+1))-1);
PA.d3HatR = PA.d3Hat;

PA.TotalCells = 1e5; 
PA.DeathTime = 2 ;
%% Set initial conditions

%Calculate the cell cycle duration
TotalTime = 1/PA.a1 +1/(PA.a2+PA.d2)+PA.tau;
% Initial Conditions
QIC = (1/PA.a1./TotalTime).*PA.TotalCells.*(1-PA.nu); %100;
SIC = (1/(PA.a2+PA.d2)./TotalTime).*PA.TotalCells.*(1-PA.nu);  %100;
TCIC = (PA.tau./TotalTime).*PA.TotalCells.*(1-PA.nu).*ones(1,PA.N)./PA.N; %Transit compartment ICs
NCIC = (PA.tau./TotalTime).*PA.TotalCells.*(1-PA.nu);
IIC = 0;
VIC = 0;
CIC =  PA.CprodHomeo./PA.kel;
PIC =   PA.Kcp.*CIC./((PA.P12+CIC).*PA.gammaP);
% Resistant Strain ICs
RIC =   (1/PA.a1./TotalTime).*PA.TotalCells.*(PA.nu); %
RSIC =   (1/(PA.a2+PA.d2)./TotalTime).*PA.TotalCells.*(PA.nu); %
ResistantTCIC =   (PA.tau./TotalTime).*PA.TotalCells.*(PA.nu).*ones(1,PA.N)./PA.N; 
ResistantTotalCellsIC =   (PA.tau./TotalTime).*PA.TotalCells.*(PA.nu);
InitialConditions = [QIC,SIC,IIC,VIC,TCIC,CIC,PIC,NCIC,RIC,RSIC,ResistantTCIC,ResistantTotalCellsIC];


    
%% Set prime and booster (enhancer and VSV) dosing
PA.ViralAdminNumberStrainPrimer =1; % maximal number of Strain primer
PA.ViralAdminNumberStrainBooster = 1; % maximal number of Strain booster
        
ViralTherapyDosesStrainPrimer = ones(1,1); % Create a dose vector with jj priming dosess administered on consecutive days
ViralTherapyDosesStrainBooster = ones(1,1); % Create a dose vector with jj priming dosess administered on consecutive days
        
PA.ViralStartTimePrimer = 6; %first primer day
PA.ViralStartTimeBooster = 8; %booster administration day
        
PA.ViralAdminStrainPrimer =  ViralTherapyDosesStrainPrimer.*PrimerDose;%*dosage_pert_array(jj);%.*dosage_pert_mat{dcount}{jj}; %Amount of virus administered
PA.ViralAdminStrainBooster = ViralTherapyDosesStrainBooster.*BoosterDose; %Amount of virus administered

%% Run simulation
% Time interval
tf = (20); % End time is 15 days after last priming dose.
tstep = floor(tf./20);
totaltime = [0 tf];
    tic    
[solTreat PAN] =  model_solver_primer_number_KCP(PA,totaltime,InitialConditions);
    toc   
%calculating tumour burden from last primer
TimeSeries = linspace(0,tf,1001); %Create 1000 evenly space points between 0 and tf
EvalSol = deval(solTreat,TimeSeries); %Evaluate the solution at the collocation points
TrapFun = EvalSol(1,:)+ EvalSol(2,:)+ EvalSol(3,:)+ EvalSol(PAN+7,:)+ EvalSol(PAN+8,:)+ EvalSol(PAN+9,:)+ EvalSol(PAN+PAN+10,:)+ EvalSol(PAN+PAN+12,:); % Find the tumour burden at the collocation points.
    
%% Plot figures
figure

subplot(2,2,1)
hold on 
yyaxis left
plot(TimeSeries,EvalSol(1,:),'LineWidth',2)
plot(TimeSeries,EvalSol(2,:),'LineWidth',2)
ylabel('No. of cells')
yyaxis right
plot(TimeSeries,EvalSol(3,:),'LineWidth',2)
plot(TimeSeries,EvalSol(PAN+PAN+11,:),'LineWidth',2)
ylabel('No. of cells')
legend('Q(t)','G_1(t)','I_VSV','I_VV')
set(gca,'FontSize',13)

subplot(2,2,2)
hold on 
yyaxis left
plot(TimeSeries,EvalSol(PAN+8,:),'LineWidth',2)
ylabel('No. of cells')
yyaxis right
plot(TimeSeries,EvalSol(PAN+9,:),'LineWidth',2)
ylabel('No. of cells')
legend('Q_R(t)','G_{1,R}(t)')
set(gca,'FontSize',13)

subplot(2,2,3)
hold on 
yyaxis left
plot(TimeSeries,EvalSol(4,:),'LineWidth',2)
ylabel('VSV virion count')
yyaxis right
plot(TimeSeries,EvalSol(PAN+PAN+12,:),'LineWidth',2)
ylabel('VV virion count')
legend('V_{VSV}(t)','V_{VV}(t)')
set(gca,'FontSize',13)

subplot(2,2,4)
hold on 
yyaxis left
plot(TimeSeries,EvalSol(PAN+5,:),'LineWidth',2)
ylabel('Cytokine concentration')
hold on 
yyaxis right
plot(TimeSeries,EvalSol(PAN+6,:),'LineWidth',2)
ylabel('No. of phagocytes')
legend('C(t)','P(t)')
set(gca,'FontSize',13)
       


