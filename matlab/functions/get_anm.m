function [anmt_out, anmk,parametric, roomParams] = get_anm(fs_sim, N, roomDims,refCoef, srcPos, recPos, p, q, nfft)
%input:     simulation sample rate, order, room parameters, source position, 
%           mic position, p and q for down-sampeling, frequncy resolution
%==========================================
    %simulate with tom's image methoud
    %======================================
    [anmt, parametric, roomParams] = image_method.calc_rir(fs_sim, roomDims,...
    srcPos, recPos, refCoef, {"angle_dependence", false,"max_reflection_order", 300}, ...
    {"array_type", "anm", "N", N,"bpfFlag",true});
    anmt_out = anmt.';
    %Down-sample anmt
    %======================================
    anm_N_tmp = zeros(ceil(size(anmt,1)* (p/q)), size(anmt,2));
    for rInd = 1:size(anmt,2)
        anm_N_tmp(:,rInd) = resample( anmt(:,rInd),p,q);
    end
    anmt = anm_N_tmp;
    %Time -> Frequncy
    %======================================
    fftDim = 1;
    anmk = fft(anmt,nfft,fftDim);
    anmk = anmk.';
    anmk = anmk(:,1:nfft/2+1); %take only the positive side
    r_d = image_method.critical_distance_diffuse(roomDims, repmat(refCoef,[6,1]));  %get the room critical disstance
    roomParams.r_d = r_d;
end
