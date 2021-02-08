close all
clear

dt_ms = 0.1;
t = 0:dt_ms:200;

t_end_ms = 50; % length of simulink simulatio in ms
t_step = 0.01; % max step size of solver

% create input alpha signal
tau_ms = 2;
C = exp(1) / tau_ms;
output = C * t .* exp(-t/tau_ms);
input_signal = transpose([t; output]);

% model parameters
soma_threshold = 0.25; % threshold of soma before it spikes
peaks = [0.5 1.0 1.5 2.0 2.5]; % peak amplitude of alpha signal

% soma nonlinear parameters
pulse_width_ms = 1;
pulse_height = 6;

% dendrite sigmoid parameters
loc = 1.7;
gain = 3;
sensitivity = 0.3868;

for i = 1:length(peaks)
    % setup of variable parameter
    peak = peaks(i);
    
    % run simulink simulation
    sim_results = sim('lnl_working4.slx');

    % out results
    signal_logs = sim_results.logsout;

    Vd = signal_logs.getElement('Vd');
    Vd_lin = signal_logs.getElement('Vd_lin');
    Vd_nl = signal_logs.getElement('Vd_nl');
    input = signal_logs.getElement('input');
    Vs = signal_logs.getElement('Vs');
    
    input_vals = input.Values.Data(:);

    Vd_time = Vd.Values.Time(:);
    Vd_vals = Vd.Values.Data(:);

    Vs_time = Vs.Values.Time(:);
    Vs_vals = Vs.Values.Data(:);

    Vd_lin_time = Vd_lin.Values.Time(:);
    Vd_lin_vals = Vd_lin.Values.Data(:);

    Vd_nl_time = Vd_nl.Values.Time(:);
    Vd_nl_vals = Vd_nl.Values.Data(:);
    
    trial_output(i).peak = peak;
    trial_output(i).time = Vd_time;
    trial_output(i).Vd = Vd_vals;
    trial_output(i).Vs = Vs_vals;
    trial_output(i).input = input_vals;

%     hold on
%     figure();
%     plot(Vd_time, Vd_vals);
%     plot(Vs_time, Vs_vals);
%     legend('Vd', 'Vs');
%     hold off
end

save('output.mat', 'trial_output');



