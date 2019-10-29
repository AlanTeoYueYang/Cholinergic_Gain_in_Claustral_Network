% Actual architecture with same ideas as before

gamma = 9;
rate = 10;
%desired_SA = 0.15; % 0.412 Can go to 0.2789
desired_SA = 1.75;

W0 = create_matrix(270,30,0.03,0.4,0.4,0.5,10,gamma);
[Wsoc, SA_values] = soc_function(W0, rate, desired_SA, gamma, 270);
%%
X0 = ini_vec(Wsoc);
params.n_timepoints = 2400;
params.over_tau = 1/200;
params.tfinal = 2400;
params.r0 = 20;
%% ----
disp('0% completed')
params.with_stim = 2;
params.ACh = 0;
dynamics = integrate_dynamics(Wsoc, params, X0);
f_rates_ctrl = dynamics.r;
CC.ctrl_f_ws = f_rates_ctrl(:,1:180);
CS.ctrl_f_ws = f_rates_ctrl(:,181:270);
VIP.ctrl_f_ws = f_rates_ctrl(:,271:280);
SST.ctrl_f_ws = f_rates_ctrl(:,281:290);
PV.ctrl_f_ws = f_rates_ctrl(:,291:end);

disp('50% completed')
params.ACh = 1;
dynamics = integrate_dynamics(Wsoc, params, X0);
f_rates_stim = dynamics.r;
CC.stim_f_ws = f_rates_stim(:,1:180);
CS.stim_f_ws = f_rates_stim(:,181:270);
VIP.stim_f_ws = f_rates_stim(:,271:280);
SST.stim_f_ws = f_rates_stim(:,281:290);
PV.stim_f_ws = f_rates_stim(:,291:end);

disp('100% completed')