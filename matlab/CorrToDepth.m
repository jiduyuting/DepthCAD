clc;clear;close all

load('/data/pre_student/hcy/GLRUN/corr.mat')
freqVec = [40, 1e2 / 3.3, 1e2 / 1.7] * 1e6;
maxd = 10;
nt = 5000;
nf = numel(freqVec);
h = corr_imgs; 
h0mat = h(1:nf,:,:); %cos
h90mat = h(nf+1:end,:,:); %sin
corr_imgs = h0mat + 1i*h90mat;
phase_imgs = angle(corr_imgs);

for fi = 1:nf
    tmp = squeeze(phase_imgs(fi,:,:)<0);
    phase_imgs(fi,tmp) = 2*pi + phase_imgs(fi,tmp);
end

corr_imgs = cat(1,h0mat,h90mat);
delayVec = linspace(0,2*maxd,nt);
depths = PhaseImgs2Depths(freqVec, phase_imgs, delayVec/2);
save('/data/pre_student/hcy/GLRUN/depth.mat')
