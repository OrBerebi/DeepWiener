function [h_l,h_r, delay] = rif_from_parametric(fs, delay, amp, doa, opts)
arguments
    fs (1,1) double % Hz
    delay (:,1) double % sec
    amp (:,1) double
    doa (:,2) double
    opts.N (1,1) double
    opts.isComplexSH (1,1) logical  = true
    opts.nfft (1,1) double          = 2^13; %numbers of frequncy sampels
    opts.eardirs (2,2) double       = [pi/2,pi/2;pi/2,pi/2];
    opts.array_radius (1,1) double  = 0.085;
    opts.c (1,1) double = soundspeed();
    
end
%
% Author: Tom Shlomo, BGU 2020


% input validation
assert( size(delay,1) == size(amp,1), 'length of delay and amp must be the same' );
assert(size(doa,1)==size(delay,1), 'length of doa and delay must me the same');

% set some parameters according to array type

N = opts.N;
c = opts.c;

th_ears =   opts.eardirs(1,:);
ph_ears =   opts.eardirs(2,:);
%Frequency calculation
nfft = opts.nfft;                                   % frequency resolution
f=linspace(0,fs/2,nfft/2+1);                        % frequency range
w=2*pi*f;                                           % radial frequency

delay_freq  = exp(-1j*w.'*delay.');                 %Frequencies X Space

k=w/c;
kr=k*opts.array_radius;

cosTheta_l = cos(doa(:,1))*cos(th_ears(1))+cos(doa(:,2)-ph_ears(1)).*sin(doa(:,1))*sin(th_ears(1));
cosTheta_r = cos(doa(:,1))*cos(th_ears(2))+cos(doa(:,2)-ph_ears(2)).*sin(doa(:,1))*sin(th_ears(2));


delay_l_freq = delay_freq .* exp(1j*kr.'*cosTheta_l.');
delay_r_freq = delay_freq .* exp(1j*kr.'*cosTheta_r.');




        
Yh          = conj(shmat(N, doa, opts.isComplexSH, true));
Yh_amp      = Yh .* amp.';                          %Space X SH
%h           = delay_freq*Yh_amp.';                  %SH X Frequencies
h_l         = Yh_amp*delay_l_freq.';                      %SH X Frequencies
h_r         = Yh_amp*delay_r_freq.';                      %SH X Frequencies



end

