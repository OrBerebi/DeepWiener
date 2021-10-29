%This script generate multi channel STFT of a desiered and undesired
%sources, simulated by the Image methoud.

clear
close all

%Folders path
%=============================
current_folder              = "/Users/orberebi/Documents/GitHub/DeepWiener/";
dry_speakers_train_path     = "/Users/orberebi/Documents/Data/wsj_wav/si_dt_05";
dry_speakers_test_path      = "/Users/orberebi/Documents/Data/TSP/train/";
dry_noise_path              = "/Users/orberebi/Documents/Data/noises/";
save_path                   = current_folder + "/data/09_08_21_test/";
mkdir(save_path)
addpath(genpath(current_folder));

%data_type
%=============================
train = true;
validation = false;
test = false;
calc_binaural_ex = false;


%data_size
%=============================
num_of_data_points_train = 100;
num_of_data_points_val = 5;
num_of_data_points_test = 5;
%calculate the extra noise realazations
if (train == true)
	noise_data_size = 1000;
elseif (validation == true)
    noise_data_size = 10;
elseif (test == true)
    noise_data_size = 10;
end


if (train == true)
    save_name           = save_path + "data_train.mat";
end
if (validation == true)
    save_name           = save_path + "data_val.mat";
end
if (test == true)
    save_name           = save_path + "data_test.mat";
end



%Simulation Parameters
%----------------------
SNR_source =  1;            %in dB
N=2;                        %Synthesis SH order
c=343;
refCoef=0.82;               %reflecation coeff
roomDims=[8,6,4];           % room dimensions
recPos=[4.2,3.7,1.7];       % Mic position (head center)
fs_sim = 16e3;              %image methud sample rate (it needs to be high)
nfft = 2^7;                 %Not in use
[p,q] = rat(fs_sim/fs_sim);



%STFT params
%------------------------------------------------
winsize = 512;
method = 'ola';
ol      = 256;                  %overlap sample size
win = hann(winsize,'periodic');
noverlap = 256;
tf = iscola(win,ol,method);
fft_len = winsize;
S_t = 160; %num of time bins in STFT
conv_len = ol*(S_t+1);

%Load speakers files
%----------------------
disp('loading dry speakers...');
[dry_speakers_train, train_speakers_mean_norm]  = load_speckers(dry_speakers_train_path,fs_sim,conv_len);
[dry_speakers_test, ~ ]                         = load_speckers(dry_speakers_test_path,fs_sim,conv_len);

%Shuffel the speckers
%----------------------------------------
neworder = randperm(numel(dry_speakers_train));
dry_speakers_train = dry_speakers_train(neworder);
neworder = randperm(numel(dry_speakers_test));
dry_speakers_test = dry_speakers_test(neworder);


%get the right size of dry speakers
%----------------------------------------
dry_speakers_val    = dry_speakers_test(num_of_data_points_test+1:num_of_data_points_test + num_of_data_points_val);
dry_speakers_train  = dry_speakers_train(1:num_of_data_points_train);
dry_speakers_test   = dry_speakers_test(1:num_of_data_points_test);



%Load noise files
%----------------------
disp('loading noise samples...');
dry_noises = load_noise(dry_noise_path,fs_sim);

%Get the Ynm
%------------------------------------------------
[a_grid,th_grid,ph_grid] = uniform_sampling_extended(N);
%[a_grid,th_grid,ph_grid] = equiangle_sampling(N);
Y  = sh2(N,th_grid,ph_grid); 
Yp = diag(a_grid)*Y';

mean_des = 0;
mean_noise = 0;
std_s_des = 0;
std_s_noise = 0;

max_value = 0;
min_value = 0;

if (train == true)
    disp('Calculating the train dataset...');
    dry_speakers = dry_speakers_train; %enter train/test to save the correct set
elseif(validation == true)
    disp('Calculating the validation dataset...');
    dry_speakers = dry_speakers_val; %enter train/test to save the correct set
else
    disp('Calculating the test dataset...');
    dry_speakers = dry_speakers_test; %enter train/test to save the correct set
end

data_size = length(dry_speakers);

bar_sim_white_full  = 0;
bar_sim_white_band  = 0;
bar_sim_pink_full   = 0;
bar_sim_pink_band   = 0;
bar_real            = 0;

for i= 1:data_size
    disp((i/data_size)*100);
    dry = dry_speakers{i};
    scale_room_down = 0.8;  %control the range of rand
    shift_room = 0.1;       %control the range of rand
    
    srcPos_des      = [(rand*scale_room_down + shift_room)*roomDims(1),...
                        (rand*scale_room_down + shift_room)*roomDims(2),...
                        (rand*scale_room_down + shift_room)*roomDims(3)];     %random location of desierd source
    
    srcPos_noise    = [(rand*scale_room_down + shift_room)*roomDims(1),...
                        (rand*scale_room_down + shift_room)*roomDims(2),...
                        (rand*scale_room_down + shift_room)*roomDims(3)];     %random location of noise source
    
    
    [real_or_sim, sim_rand_noise_type, sim_filtered] = noise_rand_param(); %random noise choises
    


    if (train == true)          %Train noise
        real_or_sim = 1;
        sim_rand_noise_type = randi([0 1]);
        sim_filtered = randi([0 1]);
        rand_num_of_tones = randi([1 6]);
        if real_or_sim == 1
            dry_noise = gen_random_noise(train_speakers_mean_norm,length(dry),SNR_source,fs_sim, sim_rand_noise_type, sim_filtered,rand_num_of_tones);
        else
            dry_noise = get_noise(dry_noises,train_speakers_mean_norm,length(dry),SNR_source);
        end
    elseif (validation == true) %Validation noise
        real_or_sim = 1;
        sim_rand_noise_type = randi([0 1]);
        sim_filtered = randi([0 1]);
        rand_num_of_tones = randi([1 6]);
        if real_or_sim == 1
            dry_noise = gen_random_noise(train_speakers_mean_norm,length(dry),SNR_source,fs_sim, sim_rand_noise_type, sim_filtered,rand_num_of_tones);
        else
            dry_noise = get_noise(dry_noises,train_speakers_mean_norm,length(dry),SNR_source);
        end
    else                        %Test noise
        real_or_sim = 1;
        sim_rand_noise_type = randi([0 1]);
        sim_filtered        = randi([0 1]);
        rand_num_of_tones   = randi([1 6]);
        if real_or_sim == 1
            dry_noise = gen_random_noise(train_speakers_mean_norm,length(dry),SNR_source,fs_sim, sim_rand_noise_type, sim_filtered,rand_num_of_tones);
        else
            dry_noise = get_noise(dry_noises,train_speakers_mean_norm,length(dry),SNR_source);
        end
    end
    
    bar_sim_white_full = bar_sim_white_full + (1 - sim_filtered) * sim_rand_noise_type * real_or_sim;
    bar_sim_white_band = bar_sim_white_band + (0 + sim_filtered) * sim_rand_noise_type * real_or_sim;
    bar_sim_pink_full = bar_sim_pink_full + (1 - sim_filtered) * (1 - sim_rand_noise_type) * real_or_sim;
    bar_sim_pink_band = bar_sim_pink_band + (0 + sim_filtered) * (1 - sim_rand_noise_type) * real_or_sim;
    bar_real = bar_real +  (1 - real_or_sim);

    
    [a_t_omega_sig,roomParams]  = calc_a_omega(fs_sim, N, roomDims, refCoef, srcPos_des, recPos, p,q,nfft, Yp);
    [a_t_omega_noise,~]         = calc_a_omega(fs_sim, N, roomDims, refCoef, srcPos_noise, recPos, p,q,nfft, Yp);
    
    %Pad the short IR with zeros to match the longer responce
    %-----------------------------
    [a_t_omega_sig,a_t_omega_noise] = pad_impuls_responce(a_t_omega_sig,a_t_omega_noise);
    %a_t_omega_sig = a_t_omega_sig(:,1:S_IR);
    %a_t_omega_noise = a_t_omega_noise(:,1:S_IR);
    
    
    
    %Conv dry signal with a(t,omega)
    %-----------------------------
    a_t_omega_sig_conv   = miro_fastConv(a_t_omega_sig.',dry).';            %CH X Time
    a_t_omega_noise_conv = miro_fastConv(a_t_omega_noise.',dry_noise).';    %CH X Time
    
    %Cut signals
    %-----------------------------
    a_t_omega_sig_conv      = a_t_omega_sig_conv(:,1:conv_len );
    a_t_omega_noise_conv    = a_t_omega_noise_conv(:,1:conv_len);
    %calc first and second moments of the signals, mean by chanels and sum
    %-----------------------------
    mean_des    = mean_des    + mean(mean(a_t_omega_sig_conv,2));
    mean_noise  = mean_noise  + mean(mean(a_t_omega_noise_conv,2));
    std_s_des   = std_s_des   + mean(std(a_t_omega_sig_conv,0,2).^2);
    std_s_noise = std_s_noise + mean(std(a_t_omega_noise_conv,0,2).^2);


    stft_a_t_omega_des      = stft(a_t_omega_sig_conv.',fs_sim,...
        'Window',win,'OverlapLength',ol,'FFTLength',fft_len,...
        'FrequencyRange','onesided');
    stft_a_t_omega_noise    = stft(a_t_omega_noise_conv.',fs_sim,...
        'Window',win,'OverlapLength',ol,'FFTLength',fft_len,...
        'FrequencyRange','onesided');
    
    %stft_a_t_omega_noise = filter_or_nah(stft_a_t_omega_noise);
    
    data.des{i}             = stft_a_t_omega_des;
    data.undes{i}           = stft_a_t_omega_noise;
    
    %sound(dry+ dry_noise,fs_sim)
        
end


for i= length(dry_speakers)+ 1 :noise_data_size
    disp((i/noise_data_size)*100);
    %dry = dry_speakers{i};
    scale_room_down = 0.8;  %control the range of rand
    shift_room = 0.1;       %control the range of rand
    
    srcPos_des      = [(rand*scale_room_down + shift_room)*roomDims(1),...
                        (rand*scale_room_down + shift_room)*roomDims(2),...
                        (rand*scale_room_down + shift_room)*roomDims(3)];     %random location of desierd source
    
    srcPos_noise    = [(rand*scale_room_down + shift_room)*roomDims(1),...
                        (rand*scale_room_down + shift_room)*roomDims(2),...
                        (rand*scale_room_down + shift_room)*roomDims(3)];     %random location of noise source
    
    
%     [real_or_sim, sim_rand_noise_type, sim_filtered] = noise_rand_param(); %random noise choises
%     bar_sim_white_full = bar_sim_white_full + (1 - sim_filtered) * sim_rand_noise_type * real_or_sim;
%     bar_sim_white_band = bar_sim_white_band + (0 + sim_filtered) * sim_rand_noise_type * real_or_sim;
%     bar_sim_pink_full = bar_sim_pink_full + (1 - sim_filtered) * (1 - sim_rand_noise_type) * real_or_sim;
%     bar_sim_pink_band = bar_sim_pink_band + (0 + sim_filtered) * (1 - sim_rand_noise_type) * real_or_sim;
%     bar_real = bar_real +  (1 - real_or_sim);


    if (train == true) %Train noise
        real_or_sim = 1;
        sim_rand_noise_type = randi([0 1]);
        sim_filtered = randi([0 1]);
        rand_num_of_tones = randi([1 6]);
        if real_or_sim == 1
            dry_noise = gen_random_noise(train_speakers_mean_norm,length(dry),SNR_source,fs_sim, sim_rand_noise_type, sim_filtered,rand_num_of_tones);
        else
            dry_noise = get_noise(dry_noises,train_speakers_mean_norm,length(dry),SNR_source);
        end
    elseif (validation == true) %Validation noise
        real_or_sim = 1;
        sim_rand_noise_type = randi([0 1]);
        sim_filtered = randi([0 1]);
        rand_num_of_tones = randi([1 6]);
        if real_or_sim == 1
            dry_noise = gen_random_noise(train_speakers_mean_norm,length(dry),SNR_source,fs_sim, sim_rand_noise_type, sim_filtered,rand_num_of_tones);
        else
            dry_noise = get_noise(dry_noises,train_speakers_mean_norm,length(dry),SNR_source);
        end
    else                        %Test noise
        real_or_sim = 1;
        sim_rand_noise_type = randi([0 1]);
        sim_filtered = randi([0 1]);
        rand_num_of_tones = randi([1 6]);
        if real_or_sim == 1
            dry_noise = gen_random_noise(train_speakers_mean_norm,length(dry),SNR_source,fs_sim, sim_rand_noise_type, sim_filtered,rand_num_of_tones);
        else
            dry_noise = get_noise(dry_noises,train_speakers_mean_norm,length(dry),SNR_source);
        end
    end

    bar_sim_white_full = bar_sim_white_full + (1 - sim_filtered) * sim_rand_noise_type * real_or_sim;
    bar_sim_white_band = bar_sim_white_band + (0 + sim_filtered) * sim_rand_noise_type * real_or_sim;
    bar_sim_pink_full = bar_sim_pink_full + (1 - sim_filtered) * (1 - sim_rand_noise_type) * real_or_sim;
    bar_sim_pink_band = bar_sim_pink_band + (0 + sim_filtered) * (1 - sim_rand_noise_type) * real_or_sim;
    bar_real = bar_real +  (1 - real_or_sim);
    
    %[a_t_omega_sig,roomParams]  = calc_a_omega(fs_sim, N, roomDims, refCoef, srcPos_des, recPos, p,q,nfft, Yp);
    [a_t_omega_noise,~]         = calc_a_omega(fs_sim, N, roomDims, refCoef, srcPos_noise, recPos, p,q,nfft, Yp);
    
    %Pad the short IR with zeros to match the longer responce
    %-----------------------------
    %[a_t_omega_sig,a_t_omega_noise] = pad_impuls_responce(a_t_omega_sig,a_t_omega_noise);
    %a_t_omega_sig = a_t_omega_sig(:,1:S_IR);
    %a_t_omega_noise = a_t_omega_noise(:,1:S_IR);
    
    
    
    %Conv dry signal with a(t,omega)
    %-----------------------------
    %a_t_omega_sig_conv   = miro_fastConv(a_t_omega_sig.',dry).';            %CH X Time
    a_t_omega_noise_conv = miro_fastConv(a_t_omega_noise.',dry_noise).';    %CH X Time
    
    %Cut signals
    %-----------------------------
    %a_t_omega_sig_conv      = a_t_omega_sig_conv(:,1:conv_len );
    a_t_omega_noise_conv    = a_t_omega_noise_conv(:,1:conv_len);
    %calc first and second moments of the signals, mean by chanels and sum
    %-----------------------------
%     mean_des    = mean_des    + mean(mean(a_t_omega_sig_conv,2));
%     mean_noise  = mean_noise  + mean(mean(a_t_omega_noise_conv,2));
%     std_s_des   = std_s_des   + mean(std(a_t_omega_sig_conv,0,2).^2);
%     std_s_noise = std_s_noise + mean(std(a_t_omega_noise_conv,0,2).^2);


%     stft_a_t_omega_des      = stft(a_t_omega_sig_conv.',fs_sim,...
%         'Window',win,'OverlapLength',ol,'FFTLength',fft_len,...
%         'FrequencyRange','onesided');
    stft_a_t_omega_noise    = stft(a_t_omega_noise_conv.',fs_sim,...
        'Window',win,'OverlapLength',ol,'FFTLength',fft_len,...
        'FrequencyRange','onesided');
    
    %stft_a_t_omega_noise = filter_or_nah(stft_a_t_omega_noise);
    
%    data.des{i}             = stft_a_t_omega_des;
    data.undes{i}           = stft_a_t_omega_noise;
    
    %sound(dry+ dry_noise,fs_sim)
        
end



%Plot room and recording geometry
%=================================================
plot_room(recPos,srcPos_des,srcPos_noise,roomDims)

%Plot noise types distrebution
%=================================================
figure
X = categorical({'real','sim w full','sim p full','sim w band','sim p band'});
X = reordercats(X,{'real','sim w full','sim p full','sim w band','sim p band'});
bar_list = [bar_real bar_sim_white_full bar_sim_pink_full bar_sim_white_band bar_sim_pink_band];
bar(X,bar_list)
title("Train : distribution of noise types")
ylabel("Number of instances")
movegui("east")

mean_des    = mean_des/data_size;
mean_noise  = mean_noise/data_size;
std_s_des   = std_s_des/data_size;
std_s_noise = std_s_noise/data_size;

mean_sum = mean_des + mean_noise;
std_sum = sqrt(std_s_des + std_s_noise);
for i=1:numel(data.undes)
    %data.des{i}             = (data.des{i} - mean_sum)/std_sum;
    %data.undes{i}           = (data.undes{i} - mean_sum)/std_sum;
    data.sigma_noise{i}     = mean(abs(data.undes{i}).^2,2);
    %[max_value,min_value] = find_max_min(data.des{i},data.undes{i},max_value,min_value);
end

% for i=1:data_size
%     data.des{i} = (data.des{i} - min_value)./(max_value-min_value);
%     data.undes{i} = (data.undes{i} - min_value)./(max_value-min_value);
% 
% end

if calc_binaural_ex == true
    % Load HRTF's
    % ------------------------------------------------
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

    %calc the binaural signals
    % ------------------------------------------------
    %Get Binaural Signals
    %============================
    disp('Calculating a binaural signals example...');
    ex_idx = data_size;
    mu= 1;
    beta = 0.8;

    mask_winner = get_w_mask(data.des{ex_idx},data.undes{ex_idx});
    mask_DD = decision_directed(mu,beta,data.undes{ex_idx}, data.des{ex_idx}+data.undes{ex_idx},Y,Yp);

    stft_data = data.des{ex_idx};
    pt_sig = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    stft_data = data.des{ex_idx} + data.undes{ex_idx};
    pt_no_mask = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);


    stft_data = mask_winner.*(data.des{ex_idx} + data.undes{ex_idx});
    pt_winner = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);


    stft_data = mask_DD.*(data.des{ex_idx} + data.undes{ex_idx});
    pt_DD = cal_pt(stft_data,N,H_l_nm,H_r_nm,tild_N_mat,...
        nfft,Yp,fs_sim,win,ol,fft_len);

    %Cut the nasty shit at the begining
    %---------------------------------------
    cut_start = 50;
    pt_sig = pt_sig(cut_start:end,:);
    pt_no_mask = pt_no_mask(cut_start:end,:);
    pt_winner = pt_winner(cut_start:end,:);
    pt_DD = pt_DD(cut_start:end,:);


    %Plot all the signals
    %=============================================================
    figure
    plot_stuff(pt_sig,pt_no_mask,pt_winner,pt_DD,mask_winner,mask_DD,fs_sim)
end

des = data.des;
undes = data.undes;
sigma_noise = data.sigma_noise;
disp('Saving the data...');
%save(save_name,'des','undes','sigma_noise','Y','Yp','N', '-v7.3')
save(save_name,'des','undes','Y','Yp','N', '-v7.3')

%Play the signals before the room, after, after STFT and mixed
%==============================================================
% sound(pt_sig,fs_sim)
% sound(pt_no_mask,fs_sim)
% sound(pt_winner,fs_sim)
% sound(pt_DD,fs_sim)




function plot_stuff(pt_sig,pt_no_mask,pt_winner,pt_DD,mask_winner,mask_DD,fs)
T = linspace(0,length(pt_sig)/fs,length(pt_sig));

tiledlayout(3,2)
nexttile
imagesc(mask_winner(:,:,2));
colorbar
set(gca,'YDir','normal')
title('True wiener mask')
nexttile
imagesc(mask_DD(:,:,2));
colorbar
set(gca,'YDir','normal')
title('DD mask')

nexttile
plot(T,pt_sig)
title('Dry signal')

nexttile
plot(T,pt_no_mask)
title('Noisy signal')

nexttile
plot(T,pt_winner)
title('True wiener masked signal')

nexttile
plot(T,pt_DD)
title('Decision Directed masked signal')

end
function [a_t_omega,roomParams] = calc_a_omega(fs_sim, N, roomDims, refCoef, srcPos, recPos, p,q,nfft, Yp)
[anmt,anmk,parametric, roomParams] = get_anm(fs_sim, N, roomDims, refCoef, srcPos, recPos, p,q,nfft);     %SH X freq and SH X time
a_t_omega = Yp*anmt;
a_t_omega = real(a_t_omega);

%a_t_omega = Yp*anmt;
%a_t_omega = real(a_t_omega);
%     %Plot room and recording geometry
%     %------------------------------------------------
%     [th0,ph0,r0]=c2s(srcPos(1)-recPos(1),srcPos(2)-recPos(2),srcPos(3)-recPos(3));
%     ph0=mod(ph0,2*pi);
%     srcPos_relative=[r0,th0,ph0];
%     roomSimulationPlot_ISF(roomDims, srcPos_relative, recPos) %from ACLtoolbox

end
function Pt = calc_pt(Pl,Pr)
Pl(1)=Pl(1);                   Pr(1)=Pr(1);
Pl(end)=Pl(end);               Pr(end)=Pr(end);
Pl=[Pl;flipud(conj(Pl(2:end-1)))];   Pr=[Pr;flipud(conj(Pr(2:end-1)))];

Pl=ifft(Pl,'symmetric'); %zero pedding length of data
Pr=ifft(Pr,'symmetric');
Pt = [Pl,Pr];
end
function ab = miro_fastConv(a,b)

% Internal use only
% 
 NFFT = size(a,1)+size(b,1)-1;
 A    = fft(a,NFFT);
 B    = fft(b,NFFT);
 AB   = A.*B;
 ab   = ifft(AB);
 %ab = fftfilt([a;zeros(length(b)-1,1)],[b;zeros(length(a)-1,1)]);

%cut = round(size(ab,1).*0.8);
%ab = ab(1:cut,:);
end
function [a_t_omega_des,a_t_omega_noise] = pad_impuls_responce(a_t_omega_des,a_t_omega_noise)
ch_num = length(a_t_omega_des(:,1));
t_smp_des = length(a_t_omega_des(1,:));
t_smp_noise = length(a_t_omega_noise(1,:));

if t_smp_des > t_smp_noise
    a_t_omega_noise = [a_t_omega_noise,zeros(ch_num,t_smp_des-t_smp_noise)];
else
    a_t_omega_des = [a_t_omega_des,zeros(ch_num,t_smp_noise-t_smp_des)];
end
end
function dry_noise = get_pink_noise(dry_norm,dry_len,SNR_source)

dry_noise = pinknoise(dry_len);
dry_noise_norm = norm(dry_noise)^2;

old_SNR = ((dry_norm)/(dry_noise_norm));

alpha =  sqrt(old_SNR * 10^(-1*(SNR_source/10))) ;

dry_noise = alpha*dry_noise;

new_SNR = ((dry_norm)/(norm(dry_noise)^2));
end
function plot_room(recPos,srcPos_des,srcPos_noise,roomDims)
%Plot room and recording geometry
%------------------------------------------------
[th0,ph0,r0]=c2s(srcPos_des(1)-recPos(1),srcPos_des(2)-recPos(2),srcPos_des(3)-recPos(3));
ph0=mod(ph0,2*pi);
srcPos_relative=[r0,th0,ph0];
[th0,ph0,r0]=c2s(srcPos_noise(1)-recPos(1),srcPos_noise(2)-recPos(2),srcPos_noise(3)-recPos(3));
ph0=mod(ph0,2*pi);
srcPos_relative_2=[r0,th0,ph0];
srcPos_relative = [srcPos_relative;srcPos_relative_2];
roomSimulationPlot_ISF(roomDims, srcPos_relative, recPos) %from ACLtoolbox
end
function [max_val,min_val] = find_max_min(stft_sig,stft_noise,max_val,min_val)
current_max_real = max(real(stft_sig),[],'all');
current_max_imag = max(imag(stft_sig),[],'all');
if (max_val < current_max_real)
    max_val = current_max_real;
end
if (max_val < current_max_imag)
    max_val = current_max_imag;
end

current_min_real = min(real(stft_noise),[],'all');
current_min_imag = min(imag(stft_noise),[],'all');
if (min_val < current_min_real)
    min_val = current_min_real;
end
if (min_val < current_min_imag)
    min_val = current_min_imag;
end

if (max_val < current_max_real)
    max_val = current_max_real;
end
if (max_val < current_max_imag)
    max_val = current_max_imag;
end

current_min_real = min(real(stft_noise),[],'all');
current_min_imag = min(imag(stft_noise),[],'all');
if (min_val > current_min_real)
    min_val = current_min_real;
end
if (min_val > current_min_imag)
    min_val = current_min_imag;
end
end

function dry_noises = load_noise(dry_noise_path,fs_sim)
dry_noises{1} = [];
Files=dir(dry_noise_path+"/*.wav");
for k=1:length(Files)
    if not(Files(k).isdir)
        [dry_noise,f_noise] = audioread(strcat(Files(k).folder,'/' ,Files(k).name));
        if (fs_sim ~= f_noise)
            [p,q] = rat(fs_sim/f_noise);
            dry_noise = resample(dry_noise,double(p),double(q));
        end
        %cut the the begining and end
        %-------------------------------
        per = 10;
        smp2cut = floor(size(dry_noise,1)*(per/100));
        dry_noise = dry_noise(smp2cut:end - smp2cut,:);
        
        dry_noises{end+1} = dry_noise;
    end
end
dry_noises(1) = [];
end
function [dry_speakers ,dry_speakers_norm_mena] = load_speckers(dry_speakers_path,fs_sim,conv_len)
dry_speakers_norm = [];
dry_speakers{1} = [];
Dirs=dir(dry_speakers_path);
for k=1:length(Dirs)
    if ~(strcmp(Dirs(k).name,'.') || strcmp(Dirs(k).name,'..')) && (Dirs(k).isdir)
        speaker_snt = dir(strcat(Dirs(k).folder,'/' ,Dirs(k).name,'/*.wav'));
        for i=1:length(speaker_snt)
            [dry_speaker,f_spk] = audioread(strcat(speaker_snt(i).folder,'/' ,speaker_snt(i).name));
            [p,q] = rat(fs_sim/f_spk);
            dry_speaker = resample(dry_speaker,double(p),double(q));
            if length(dry_speaker) >= conv_len 
                dry_speakers_norm = [dry_speakers_norm; norm(dry_speaker)^2];
                dry_speakers{end+1} = dry_speaker;
            end
        end
    end
end
dry_speakers(1) = [];
dry_speakers_norm_mena = mean(dry_speakers_norm);
end
function dry_noise = get_noise(dry_noises,dry_norm,dry_len,SNR_source)
rand_noise_type = randi([1 length(dry_noises)]);
dry_noise = dry_noises{rand_noise_type};
while (size(dry_noise,1) < dry_len)
    rand_noise_type = randi([1 length(dry_noises)]);
    dry_noise = dry_noises{rand_noise_type};
end
dry_noise = dry_noise(:,randi([1 size(dry_noise,2)]));
last_inx = length(dry_noise) - dry_len - 1;
start_idx = randi([1 last_inx]);
dry_noise = dry_noise(start_idx:start_idx+dry_len-1);

dry_noise_norm = norm(dry_noise)^2;
old_SNR = ((dry_norm)/(dry_noise_norm));
alpha =  sqrt(old_SNR * 10^(-1*(SNR_source/10))) ;
dry_noise = alpha*dry_noise;
new_SNR_dB = 10*log10((dry_norm)/(norm(dry_noise)^2));
end

function dry_noise_out = gen_random_noise(dry_norm,dry_len,SNR_source,fs, rand_noise_type, cut_or_nah,rand_num_of_tones)

if rand_noise_type == 1
    dry_noise = randn([dry_len,1]);
else
    dry_noise = pinknoise([dry_len,1]);
end

dry_noise_norm = norm(dry_noise)^2;
old_SNR = ((dry_norm)/(dry_noise_norm));
alpha =  sqrt(old_SNR * 10^(-1*(SNR_source/10))) ;
dry_noise = alpha*dry_noise;
new_SNR_dB = 10*log10((dry_norm)/(norm(dry_noise)^2));


if cut_or_nah == 1
    bw = 15; %in Hz one sided
%     rand_num_of_tones = randi([1 4]);
%     rand_num_of_tones = 1;
    dry_noise_out = zeros(size(dry_noise));
    for i = 1:rand_num_of_tones
        A_rand = unifrnd(sqrt(0.5),sqrt(1.5));
        rand_fc = (rand*(fs/2 - 2*bw))+bw; %uniformly distributed from [bw - (fs/2  - bw)]
        dry_noise_out = dry_noise_out + A_rand*bandpass(dry_noise,[rand_fc-bw rand_fc+bw],fs);
    end
else
    SNR_source = randi([2 5]);
    dry_noise_norm = norm(dry_noise)^2;
    old_SNR = ((dry_norm)/(dry_noise_norm));
    alpha =  sqrt(old_SNR * 10^(-1*(SNR_source/10))) ;
    dry_noise = alpha*dry_noise;
    new_SNR_dB = 10*log10((dry_norm)/(norm(dry_noise)^2));
    dry_noise_out = dry_noise;
end
    
    
dry_noise_norm = norm(dry_noise_out)^2;
old_SNR = 10*log10((dry_norm)/(dry_noise_norm));
% alpha =  sqrt(old_SNR * 10^(-1*(SNR_source/10))) ;
% dry_noise_out = alpha*dry_noise_out;
% new_SNR_dB = 10*log10((dry_norm)/(norm(dry_noise_out)^2));
end
function stft_noise_out = filter_or_nah(stft_noise_in)
    bw_one_side = 2;
    cut_or_nah = randi([0 1]);
    if cut_or_nah == 1
        stft_noise_out = zeros(size(stft_noise_in));
        stft_noise_out = stft_noise_out + 1e-8;
        rand_num_of_tones = randi([1 4]);
        for i = 1:rand_num_of_tones
            rand_fc = randi([1+bw_one_side size(stft_noise_in,1)-bw_one_side]);
            stft_noise_out(rand_fc-bw_one_side:rand_fc+bw_one_side,:) = stft_noise_in(rand_fc-bw_one_side:rand_fc+bw_one_side,:);
        end
        
%         figure
%         imagesc(abs(stft_noise_in))
%         colorbar
%         movegui("east")
%         figure
%         imagesc(abs(stft_noise_out))
%         colorbar
%         movegui("west")
%         close all
    else
        stft_noise_out = stft_noise_in;
    end
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
end

function winner_mask = get_w_mask(stft_sig,stft_noise)
snr = (abs(stft_sig).^2)./(abs(stft_noise).^2);
winner_mask = snr./(snr+1);
end
function mask = decision_directed(mu,beta,data_undes, data_inp, Y,Yp)
%This function calculates the decision_directed PSD estimation for the
%signal, given the noisy signal
%mu - method paramater deafult as 1
%beta - method paramater deafult as 0.8
%x - noisy signal: freq X time X nm
%v - noise signal: freq X time X nm
%th_grid,ph_grid the smapeling grid

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
%tmp = floor(k_samples/2);
%mask = mask(tmp:end,:,:);
end

function [real_or_sim, sim_rand_noise_type, sim_filtered] = noise_rand_param()
    real_or_sim         = randi([0,1]);
    sim_rand_noise_type = randi([0,1]);
    sim_filtered        = randi([0,1]);
end

