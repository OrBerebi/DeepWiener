%EXTREMAL_SAMPLING This function loads an extremal sampling scheme that is
%sufficient for SH representation order N. what is speccial about this
%sampling scheme is that it follows exactly the relation (N+1)^2=Q
%% INPUT
% N             : maximum SH that can accurately represented with this
% sampling scheme
%% OUTPUT
%r,th,ph        : radius/ elevation and azimuth of each sampling point
%w              : quadrature weights of each sampling point
%% NOTES
% based on Sloan, Ian H., and Robert S. Womersley. "Extremal systems of points and numerical integration on the sphere." Advances in Computational Mathematics 21.1-2 (2004): 107-125.

function [r,th,ph,w]=extremal_sampling(N)
if nargin<1
    N=15;
end

[dir_path,~,~] = fileparts(which('extremal_sampling'));
dir_path=[dir_path,'/Extremal/Extremal system - Minimum Energy/'];
switch N
    case 1
        fid = fopen([dir_path,'design_1_4.txt'], 'r');
    case 2
        fid = fopen([dir_path,'design_2_9.txt'], 'r');
    case 3
        fid = fopen([dir_path,'design_3_16.txt'], 'r');
    case 4
        fid = fopen([dir_path,'design_4_25.txt'], 'r');
    case 5
        fid = fopen([dir_path,'design_5_36.txt'], 'r');
    case 6
        fid = fopen([dir_path,'design_6_49.txt'], 'r');
    case 7
        fid = fopen([dir_path,'design_7_64.txt'], 'r');
    case 8
        fid = fopen([dir_path,'design_8_81.txt'], 'r');
    case 10
        fid = fopen([dir_path,'design_10_121.txt'], 'r');
    case 12
        fid = fopen([dir_path,'design_12_169.txt'], 'r');
    case 15
        fid = fopen([dir_path,'design_15_256.txt'], 'r');
    case 30
        dir_path=[dir_path(1:end-15),'New2007/'];
        %     Extremal system - New2007
        %     Extremal system - Minimum Energy
        fid = fopen([dir_path,'design_30_961.txt'], 'r');
    otherwise
        error(['Order N=' num2str(N) ' is not available'])
end

tline=fgetl(fid);
tline=fgetl(fid);
tline=fgetl(fid);
a = fscanf(fid, '%g %g %g %g', [4 inf]);    % It has two rows now.
a = a';

%transforms into earo soherical coordinates
[ph,th,r] = cart2sph(a(:,1),a(:,2),a(:,3));
ph(ph>pi)=ph(ph>pi)-2*pi;
th=pi/2-th;
w=a(:,4);

%makes row vector
ph=ph(:).';
th=th(:).';
r=r(:).';
w=w(:).';
fclose(fid);
