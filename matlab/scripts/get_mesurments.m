clear
close all

current_folder      = "/Users/orberebi/Google Drive/DNN_FINAL_PROJECT/submition/";


addpath(genpath(current_folder));
%addpath(genpath('/Users/orberebi/Documents/AClab/ACLtoolbox'));


%STFT params
%------------------------------------------------
fs_sim = 16e3;             %image methud sample rate (it needs to be high)
winsize = 512;
method = 'ola';
ol      = 256;                  %overlap sample size
win = hann(winsize,'periodic');
fft_len = winsize;


%Load the matlab stft data
%==========================
matlab_data_path =  current_folder + "/data/07_07_21/";
matlab_data_file_name = matlab_data_path +"/data_val.mat";
sound_example_path = current_folder + "/results/07_07_21/";
load(matlab_data_file_name)
%[mat_des,mat_undes,mat_sigma] = get_cell2mat(des,undes,sigma_noise);
[mat_des,mat_undes] = get_cell2mat(des,undes);

%Get the true wigner mask
%snr = (abs(mat_des).^2)./(mat_sigma);
snr = (abs(mat_des).^2)./(abs(mat_undes).^2);
true_mask = snr./(snr+1); %freq X time X omega X idx

 

%Load the python estimated mask
%==================================================================
python_data_path = current_folder + "/results/07_07_21/";
python_data_file_name = python_data_path + "/masks_val.mat";
py_mask = load(python_data_file_name);
mask_DNN_big = permute(py_mask.masks,[3 4 2 1]); %freq X time X omega X idx
%reduce size of DNN_masks
%mask_DNN_big = mask_DNN_big(:,:,:,1:10:end);

% Load HRTF's
%==================================================================
S_t = size(mat_des,2); %num of time bins in STFT
conv_len = ol*(S_t+1);
nfft = conv_len;
disp('Taking care of the HRTFs...');
r_0 = 0.0875;                                         %8.75cm head radius
load earoHRIR_KEMAR_TU_BEM_OnlyHead
hobj.shutUp = false;
if hobj.fs~=fs_sim
    hobj = hobj.resampleData(fs_sim);
end
hobj.micGrid.r = r_0;

if strcmp(hobj.dataDomain{1}, 'TIME')
   hobj = hobj.toFreq(nfft);
   hobj.data = hobj.data(:,1:end/2+1,:);
   hobj.taps = nfft;
end
Y_hrtf = sh2(N,hobj.sourceGrid.elevation,hobj.sourceGrid.azimuth); % spherical harmonics matrix
pY_hrtf = pinv(Y_hrtf);
H_l_nm = double(squeeze(hobj.data(:,:,1)).')*pY_hrtf; % freq X SH transform
H_r_nm = double(squeeze(hobj.data(:,:,2)).')*pY_hrtf; % freq X SH transform
tild_N_mat = tildize(N);

%Get Binaural Signals
%============================
disp('Calculating a binaural signals example...');

mu= 1;
beta = 0.8;

for idx = 1:size(mat_des,4)
    
    %mask_DD_big(:,:,:,idx) = decision_directed(mu,beta,mat_undes(:,:,:,idx), mat_des(:,:,:,idx)+mat_undes(:,:,:,idx),Y,Yp);
    mask_DD_big(:,:,:,idx) = moti_dd(beta,  mat_des(:,:,:,idx)+mat_undes(:,:,:,idx), mat_undes(:,:,:,idx));
    
    mask_DD  = mask_DD_big(:,:,:,idx);
    mask_DNN = mask_DNN_big(:,:,:,idx);
    mask_wigner = true_mask(:,:,:,idx);

    stft_data = mat_des(:,:,:,idx);
    pt_sig(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);
    
    stft_data = mat_undes(:,:,:,idx);
    pt_noise(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    stft_data = mat_des(:,:,:,idx) +mat_undes(:,:,:,idx);
    pt_no_mask(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    stft_data = mask_wigner.*(mat_des(:,:,:,idx) +mat_undes(:,:,:,idx));
    pt_wigner(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    stft_data = mask_DD.*(mat_des(:,:,:,idx) +mat_undes(:,:,:,idx));
    pt_DD(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    stft_data = mat_des(:,:,:,idx) + mat_undes(:,:,:,idx);
    stft_data(1:size(mask_DNN,1),:,:) = mask_DNN.*stft_data(1:size(mask_DNN,1),:,:);
    pt_DNN(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);
    
    
    
    stft_data = mask_wigner.*(mat_des(:,:,:,idx));
    pt_wigner_sig(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    stft_data = mask_DD.*(mat_des(:,:,:,idx));
    pt_DD_sig(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);
    
    stft_data = mat_des(:,:,:,idx);
    stft_data(1:size(mask_DNN,1),:,:) = mask_DNN.*stft_data(1:size(mask_DNN,1),:,:);
    pt_DNN_sig(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);
    
    
    stft_data = mask_wigner.*(mat_undes(:,:,:,idx));
    pt_wigner_noise(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    stft_data = mask_DD.*(mat_undes(:,:,:,idx));
    pt_DD_noise(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    stft_data = mat_undes(:,:,:,idx);
    stft_data(1:size(mask_DNN,1),:,:) = mask_DNN.*stft_data(1:size(mask_DNN,1),:,:);
    pt_DNN_noise(:,:,idx) = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);
    
    
end

%Calc and plot Objective mesures
%=====================
mse_DNN = calc_mask_mse(true_mask,mask_DNN_big);
mse_DD = calc_mask_mse(true_mask,mask_DD_big);
sdr_DNN = calc_sdr(pt_sig,pt_DNN_sig);
sdr_DD = calc_sdr(pt_sig,pt_DD_sig);
sdr_wigner = calc_sdr(pt_sig,pt_wigner_sig);
noise_only_DNN = calc_noise_enargy(pt_noise,pt_DNN_noise);
noise_only_DD = calc_noise_enargy(pt_noise,pt_DD_noise);
noise_only_true_mask = calc_noise_enargy(pt_noise,pt_wigner_noise);

figure
boxplot([sdr_wigner,sdr_DD,sdr_DNN],'Labels',{'Wigner','DD','DNN'})
title("Signal distortion ratio")
grid on
ylabel("dB")
movegui("east")
figure
boxplot([noise_only_true_mask,noise_only_DD,noise_only_DNN],'Labels',{'Wigner','DD','DNN'})
title("Noise (gain) reduction")
grid on
ylabel("dB")
movegui("west")
figure
boxplot([mse_DD,mse_DNN],'Labels',{'DD','DNN'})
title("MSE of est mask vs True wigner mask")
grid on
movegui("south")

%Plot results
%==================================================================
plot_idx = 69;
figure
plot_stuff(pt_sig(:,:,plot_idx),pt_no_mask(:,:,plot_idx),...
    pt_wigner(:,:,plot_idx),pt_DD(:,:,plot_idx),pt_DNN(:,:,plot_idx),...
    true_mask(:,:,:,plot_idx),mask_DD_big(:,:,:,plot_idx),mask_DNN_big(:,:,:,plot_idx),...
    fs_sim)
movegui("north")

%save input signal/true filterd/est filtered as .wav
%==================================================================


tmp = [pt_sig,  pt_no_mask, pt_wigner, pt_DD , pt_DNN];
mama = max(tmp,[],'all');
mkdir(sound_example_path +"/sound_clips/")
for idx = 1:size(pt_sig,3)
    idx_path = sound_example_path +"/sound_clips/"+int2str(idx)+"/";
    mkdir(idx_path)
    norm_clean_out = 0.9*(pt_sig(:,:,idx)/mama);
    audiowrite(idx_path+"clean_output.wav",norm_clean_out,fs_sim);

    norm_noisy_input = 0.9*(pt_no_mask(:,:,idx)/mama);
    audiowrite(idx_path+"noisy_input.wav",norm_noisy_input,fs_sim);

    norm_true_filt =  0.9*(pt_wigner(:,:,idx)/mama);
    audiowrite(idx_path+"wigner_output.wav",norm_true_filt,fs_sim);

    norm_DD_filt = 0.9*(pt_DD(:,:,idx)/mama);
    audiowrite(idx_path+"DD_output.wav",norm_DD_filt,fs_sim);

    norm_py_filt = 0.9*(pt_DNN(:,:,idx)/mama);
    audiowrite(idx_path+"DNN_output.wav",norm_py_filt,fs_sim);

end

%Play results
%==================================================================
% sound(noisy_input,fs_sim)
% sound(true_filt_signal,fs_sim)
% sound(py_filt_signal,fs_sim)


function plot_stuff(pt_sig,pt_no_mask,pt_wigner,pt_DD,pt_DNN,mask_wigner,mask_DD,mask_DNN,fs)
T = linspace(0,length(pt_sig)/fs,length(pt_sig));

tiledlayout(2,3)
nexttile
imagesc(mask_wigner(:,:,2));
colorbar
set(gca,'YDir','normal')
title('True Wigner mask')
nexttile
imagesc(mask_DD(:,:,2));
colorbar
set(gca,'YDir','normal')
title('DD mask')
nexttile
imagesc(mask_DNN(:,:,2));
colorbar
set(gca,'YDir','normal')
title('DNN mask')


nexttile
boxplot([sdr_wigner,sdr_DD,sdr_DNN],'Labels',{'Wigner','DD','DNN'})
title("Signal distortion ratio")
grid on
ylabel("dB")
nexttile
boxplot([noise_only_true_mask,noise_only_DD,noise_only_DNN],'Labels',{'Wigner','DD','DNN'})
title("Noise (gain) reduction")
grid on
ylabel("dB")
nexttile
boxplot([mse_DD,mse_DNN],'Labels',{'DD','DNN'})
title("MSE of est mask vs True wigner mask")
grid on



% nexttile
% plot(T,pt_wigner)
% title('True Wigner masked signal')
% 
% nexttile
% plot(T,pt_DD)
% title('Decision Directed masked signal')
% 
% nexttile
% plot(T,pt_DNN)
% title('DNN masked signal')
% 
% nexttile
% plot(T,pt_sig)
% title('Dry signal')
% nexttile
% plot(T,pt_no_mask)
% title('Noisy signal')

end
function py_filt_signal = filter_with_py_mask(stft_sig,stft_noise,py_mask,fs_sim,win,ol,fft_len)
py_mask = permute(py_mask,[3 4 2 1]);
input_stft_cut = stft_sig(1:size(py_mask,1),1:size(py_mask,2),:,:)+stft_noise(1:size(py_mask,1),1:size(py_mask,2),:,:);
py_filt_signal = input_stft_cut.*py_mask;

%replace filtered part with un filtered
input_of_istft = stft_sig + stft_noise;
input_of_istft(1:size(py_mask,1),1:size(py_mask,2),:,:) = py_filt_signal;
for idx = 1:size(py_filt_signal,4)
    [new_py_filt_signal(:,:,idx),~]   = istft(input_of_istft(:,:,:,idx),...
        fs_sim,'Window',win,'OverlapLength',ol,'FFTLength',...
        fft_len,'Method','ola','FrequencyRange','onesided');
end
%cut the end
py_filt_signal = new_py_filt_signal(50:end-50,:,:);
end
function [true_filt_signal_no_sig,true_filt_signal_no_noise, true_filt_signal,true_mask, sig_input, noise_input, noisy_input] = get_matlab_signals(mat_sigma,sig_only,noise_only,fs_sim,win,ol,fft_len)
snr = (abs(sig_only).^2)./(mat_sigma);
true_mask = snr./(snr+1);
true_mask(257,:,:,:) = ones([1, size(true_mask,2), size(true_mask,3), size(true_mask,4)]);

true_filt_signal = (sig_only+noise_only).*true_mask;
true_filt_signal_no_noise = (sig_only).*true_mask;
true_filt_signal_no_sig = (noise_only).*true_mask;


%Compute the ISTFT and comper results with before
%=========================================================
for idx = 1:size(true_mask,4)
    [new_sig_input(:,:,idx),~]   = istft(sig_only(:,:,:,idx),...
        fs_sim,'Window',win,'OverlapLength',ol,'FFTLength',...
        fft_len,'Method','ola','FrequencyRange','onesided');
    [new_noise_input(:,:,idx),~]   = istft(noise_only(:,:,:,idx),...
        fs_sim,'Window',win,'OverlapLength',ol,'FFTLength',...
        fft_len,'Method','ola','FrequencyRange','onesided');
    [new_noisy_input(:,:,idx),~]   = istft(sig_only(:,:,:,idx) + noise_only(:,:,:,idx),...
        fs_sim,'Window',win,'OverlapLength',ol,'FFTLength',...
        fft_len,'Method','ola','FrequencyRange','onesided');
    [new_true_filt_signal(:,:,idx),~]   = istft(true_filt_signal(:,:,:,idx),...
        fs_sim,'Window',win,'OverlapLength',ol,'FFTLength',...
        fft_len,'Method','ola','FrequencyRange','onesided');
    [new_true_filt_signal_no_noise(:,:,idx),~]   = istft(true_filt_signal_no_noise(:,:,:,idx),...
        fs_sim,'Window',win,'OverlapLength',ol,'FFTLength',...
        fft_len,'Method','ola','FrequencyRange','onesided');
    [new_true_filt_signal_no_sig(:,:,idx),~]   = istft(true_filt_signal_no_sig(:,:,:,idx),...
        fs_sim,'Window',win,'OverlapLength',ol,'FFTLength',...
        fft_len,'Method','ola','FrequencyRange','onesided');
end


smp2cut = 50;
smp2cut_start = 50;
sig_input = new_sig_input(smp2cut_start:end-smp2cut,:,:);
noise_input = new_noise_input(smp2cut_start:end-smp2cut,:,:);
noisy_input = new_noisy_input(smp2cut_start:end-smp2cut,:,:);
true_filt_signal = new_true_filt_signal(smp2cut_start:end-smp2cut,:,:);
true_filt_signal_no_noise = new_true_filt_signal_no_noise(smp2cut_start:end-smp2cut,:,:);
true_filt_signal_no_sig = new_true_filt_signal_no_sig(smp2cut_start:end-smp2cut,:,:);
end

function mse = calc_mask_mse(m_tru,m_est)
%m_est = permute(m_est,[3 4 1 2]); %freq X time X omega X idx
%m_tru = permute(m_tru,[1 2 4 3]); %freq X time X omega X idx
f = linspace(0,8e3,size(m_est,1));
m_tru_cut = m_tru(1:size(m_est,1),:,:,:);

D = abs(m_tru_cut-m_est).^2;
D = squeeze(mean(D,2)); %mean time
D = squeeze(mean(D,1)); %mean freq
mse = squeeze(mean(D,1)).'; %mean omega

% figure
% plot(f,mse)
% grid on


end

function sdr = calc_sdr(clean_sig,filt_sig)

sdr = zeros([size(clean_sig,3),size(clean_sig,2)]);
for ears = 1:size(clean_sig,2)
    for subjects = 1:size(clean_sig,3)
        d = squeeze(clean_sig(:,ears,subjects));
        d_hat = squeeze(filt_sig(:,ears,subjects));
        d_hat = (norm(d_hat).^-2 .*  d_hat'* d).*d_hat;
        D = d-d_hat;
        sdr(subjects,ears)       = 10*log10(squeeze(norm(D)).^2./squeeze(norm(d)).^2);   
    end
end


sdr = squeeze(mean(sdr,2));


end

function res = calc_noise_enargy(noise,clean_noise)
res = zeros([size(noise,2),size(noise,3)]);
%noise = noise(50:end,:,:);
%clean_noise = clean_noise(50:end,:,:);
%noise = permute(noise,[1 3 2]);             %time X example X omega
%clean_noise = permute(clean_noise,[1 3 2]); %time X example X omega

%This loop calculats for every diraction
for ears = 1:size(clean_noise,2)
    for subjects = 1:size(clean_noise,3)
    d                   = squeeze(clean_noise(:,ears,subjects));
    d_hat               = squeeze(noise(:,ears,subjects));
    d_hat               = (norm(d_hat).^-2 .*  d_hat'* d).*d_hat;
    res(ears,subjects)  = 10*log10(squeeze(norm(d_hat)).^2./squeeze(norm(d)).^2);
    end
end
res = mean(res,1).'; %mean on the directions (omega)

end

function [sig_only,noise_only,sigma] = get_cell2mat_small(des,undes,sigma_noise)

num_of_sigs =  length(des);
f_sample = size(des{1},1);
t_sample = size(des{1},2);
omega_sampe = size(des{1},3);
sig_only = zeros(num_of_sigs,f_sample,t_sample,omega_sampe);
noise_only = zeros(num_of_sigs,f_sample,t_sample,omega_sampe);
sigma = zeros(num_of_sigs,f_sample,t_sample,omega_sampe);
for i = 1:num_of_sigs
        sig_only(i,:,:,:) = des{i};
        noise_only(i,:,:,:) = undes{i};
        sigma(i,:,:,:) = repmat(sigma_noise{i},1,t_sample);
end
sig_only = permute(sig_only,[2 3 4 1]);
noise_only = permute(noise_only,[2 3 4 1]);
sigma = permute(sigma,[2 3 4 1]);
end

%function [sig_only,noise_only,sigma] = get_cell2mat(des,undes,sigma_noise)
function [sig_only,noise_only] = get_cell2mat(des,undes)
noise_per_sig = 10;
des_len = length(des);
undes_len = length(undes);
data_len = length(des)*noise_per_sig;
f_sample = size(des{1},1);
t_sample = size(des{1},2);
omega_sampe = size(des{1},3);
sig_only = zeros(data_len,f_sample,t_sample,omega_sampe);
noise_only = zeros(data_len,f_sample,t_sample,omega_sampe);

for i= 0:data_len - 1
    des_idx = mod(i,des_len)+1;
    undes_idx = mod(i,undes_len)+1;
    
    sig_only(i+1,:,:,:) = des{des_idx};
    noise_only(i+1,:,:,:) = undes{undes_idx};
    
end
sig_only = permute(sig_only,[2 3 4 1]);
noise_only = permute(noise_only,[2 3 4 1]);



% num_of_sigs =  length(des);
% f_sample = size(des{1},1);
% t_sample = size(des{1},2);
% omega_sampe = size(des{1},3);
% sig_only = zeros(num_of_sigs,f_sample,t_sample,omega_sampe);
% noise_only = zeros(num_of_sigs,f_sample,t_sample,omega_sampe);
% %sigma = zeros(num_of_sigs,f_sample,t_sample,omega_sampe);
% for i = 1:num_of_sigs
%     for j = 1:num_of_sigs
%         sig_only((i-1)*num_of_sigs + j,:,:,:) = des{i};
%         noise_only((i-1)*num_of_sigs + j,:,:,:) = undes{j};
%         %sigma((i-1)*num_of_sigs + j,:,:,:) = repmat(sigma_noise{j},1,t_sample);
%     end
% end
% sig_only = permute(sig_only,[2 3 4 1]);
% noise_only = permute(noise_only,[2 3 4 1]);
%sigma = permute(sigma,[2 3 4 1]);
end

function mask = decision_directed(mu,beta,data_undes, data_inp, Y,Yp)
%This function calculates the decision_directed PSD estimation for the
%signal, given the noisy signal
%mu - method paramater deafult as 1
%beta - method paramater deafult as 0.8
%x - noisy signal: freq X time X omega
%v - noise signal: freq X time X omega
%th_grid,ph_grid the sampleling grid

a_t_k_nm_des = zeros([size(data_inp,1),size(data_inp,2),size(Yp,2)]);
a_t_k_nm_noise = zeros([size(data_inp,1),size(data_inp,2),size(Yp,2)]);
for n = 1:size(data_inp,2)
    anm_inp(:,n,:)   = squeeze(data_inp(:,n,:)) * Yp; %Omega -> SH
    anm_noise(:,n,:) = squeeze(data_undes(:,n,:)) * Yp; %Omega -> SH
end


anm_noise = permute(anm_noise, [2,3,1]);
anm_inp = permute(anm_inp, [2,3,1]);
time_sampels = size(anm_noise,1);
sh_size = size(anm_noise,2);
space_size = size(Y,2);
k_samples = size(anm_noise,3);
Phi_v       = zeros([k_samples, sh_size,sh_size ]);
Phi_v_qq    = zeros([k_samples, space_size ]);

for k = 1:k_samples
    Phi_v(k,:,:) = (1/time_sampels)*(reshape(anm_noise(:,:,k),[],sh_size).' * conj(reshape(anm_noise(:,:,k),[],sh_size))); %time X freq X SH X SH
end

for k = 1:k_samples
    for q = 1:space_size
        Phi_v_qq(k,q) = Y(:,q).' * squeeze(Phi_v(k,:,:)) * conj(Y(:,q));       %freq X space
    end 
end
Phi_v_qq = repmat(Phi_v_qq,[1 1 time_sampels]); %same noise for all time: %freq X space X time
Phi_v_qq = real(permute(Phi_v_qq, [2,3,1])); %space X time X freq


X_q_tilde   = zeros([space_size,time_sampels,k_samples]);
alpha_q     = zeros([space_size,time_sampels,k_samples]);
Z_q_tilde   = zeros([space_size,time_sampels,k_samples]);
Phi_s_qq    = zeros([space_size,time_sampels,k_samples]);
for q = 1:space_size
    for n = 2:time_sampels
        X_q_tilde(q,n,:) = Y(:,q).' *  squeeze(anm_inp(n,:,:)); % 1Xfreq
        Z_q_tilde(q,n-1,:) = alpha_q(q,n-1,:).*X_q_tilde(q,n-1,:);
        Phi_s_qq(q,n,:) = beta* abs(Z_q_tilde(q,n-1,:)).^2 + (1 - beta)*(max( abs(X_q_tilde(q,n,:)).^2 - Phi_v_qq(q,n,:),0));
        alpha_q (q,n,:) =  Phi_s_qq(q,n,:)./(Phi_s_qq(q,n,:) + mu*Phi_v_qq(q,n,:));
    end
end

mask = permute(alpha_q,[3,2,1]); %freq X time X space

end
function pt = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,nfft,Yp,fs_sim,win,ol,fft_len)
%Compute the ISTFT 
[a_omega_t,~]   = istft(stft_data,...
    fs_sim,'Window',win,'OverlapLength',ol,'FFTLength',...
    fft_len,'Method','ola','FrequencyRange','onesided');
%Omega -> SH , time -> freq
anm_t = a_omega_t*Yp;  %time X SH
anmk = fft(anm_t,size(anm_t,1),1);  %freq X SH
anmk = anmk(1:end/2+1,:); %freq X SH
%anmk SH X freq
pad = (N+1)^2;
anm_tilde = anmk(1:size(H_l_nm,1),:)*tild_N_mat; %freq X SH
pl_SH = sum(padarray(anm_tilde,[0 pad-size(anm_tilde,2)],'post').*padarray(H_l_nm,[0 pad-size(H_l_nm,2)],'post'),2).'; %1Xfreq
pr_SH = sum(padarray(anm_tilde,[0 pad-size(anm_tilde,2)],'post').*padarray(H_r_nm,[0 pad-size(H_r_nm,2)],'post'),2).'; %1Xfreq
%pt = [pl_SH;pr_SH];
pt = calc_pt(pl_SH.',pr_SH.'); %freq single side -> Time time X 2 (left,right)

cut_start = 50;
pt = pt(cut_start:end,:);
end
function Pt = calc_pt(Pl,Pr)
Pl(1)=Pl(1);                   Pr(1)=Pr(1);
Pl(end)=Pl(end);               Pr(end)=Pr(end);
Pl=[Pl;flipud(conj(Pl(2:end-1)))];   Pr=[Pr;flipud(conj(Pr(2:end-1)))];

Pl=ifft(Pl,'symmetric'); %zero pedding length of data
Pr=ifft(Pr,'symmetric');
Pt = [Pl,Pr];
end
function mask = moti_dd(beta, x_tf, n_tf)
    
    %x_theta_phi = x_nmt * Y;
    %n_theta_phi = n_nmt * Y;


    % PHI_n_qq -->   size    Nfft x Q
    %[n_tf, ,] = stft(n_theta_phi, hanning(Nfft), Nfft/2, Nfft, Fs);
    Q = size(x_tf,3);
    Nfft = size(x_tf,1);
    PHI_n_qq = reshape(mean(abs(n_tf).^2,2),[],Q);



    %[x_tf, ,] = stft(x_theta_phi, hanning(Nfft), Nfft/2, Nfft, Fs);

    % calculate the mask
    
    alpha = zeros(size(x_tf));
    PHI_s_qq = zeros(size(x_tf)); % initializing the mask

    for n = 2 : size(x_tf,2)
        PHI_s_qq(:,n,:) = beta * abs(reshape(alpha(:,n-1,:),[],Q) .* reshape(x_tf(:,n-1,:),[],Q)).^2 + (1 - beta) * max(abs(reshape(x_tf(:,n,:),[],Q)).^2 - PHI_n_qq ,0);
        alpha(:,n,:)    =  reshape(PHI_s_qq(:,n,:),[],Q)./(reshape(PHI_s_qq(:,n,:),[],Q) + PHI_n_qq);
    end


    %SNR = zeros(Nfft,size(x_tf,2),Q);

    SNR = PHI_s_qq ./ reshape(PHI_n_qq,Nfft,1,Q);

    mask = real(SNR ./ (SNR + 1));
end
