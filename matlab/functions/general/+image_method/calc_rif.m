function [h_l,h_r, parametric] = calc_rif(fs, roomDim, sourcePos, arrayPos, R, calc_parametric_rir_name_value_pairs, rif_from_parametric_name_value_args)
% for details about name-value argument see
% image_method.calc_parametric_rir.m 
% and
% rif_from_parametric.m.

% Author: Tom Shlomo, ACLab BGU, 2020

% calc paramteric info
parametric = image_method.calc_parametric_rir(roomDim, sourcePos, arrayPos,R, calc_parametric_rir_name_value_pairs{:});

% convert to rir signal
[h_l,h_r, parametric.delay] = image_method.rif_from_parametric(fs, parametric.delay, parametric.amp, parametric.omega, rif_from_parametric_name_value_args{:});

end