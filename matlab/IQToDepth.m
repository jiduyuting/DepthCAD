% IQToDepth.m: Complete pipeline from IQ to Depth
% This script loads IQ data, converts to correlation, and computes depth
%
% Usage:
%   1. Set the IQ file path (can be .npy or .mat)
%   2. Run this script
%   3. Output depth will be saved to depth.mat

clc; clear; close all;

% ========== Configuration ==========
% Path to IQ data file (can be .npy or .mat)
% If .npy, you need to load it using Python first or convert to .mat
IQ_FILE_PATH = 'IQ.mat';  % Modify this path
CORR_SAVE_PATH = 'corr.mat';
DEPTH_SAVE_PATH = 'depth.mat';

% ========== Step 1: Load IQ Data ==========
fprintf('Step 1: Loading IQ data from %s...\n', IQ_FILE_PATH);

% Check if file exists
if ~exist(IQ_FILE_PATH, 'file')
    error('IQ file not found: %s', IQ_FILE_PATH);
end

% Load IQ data
% If it's a .mat file, load directly
if endsWith(IQ_FILE_PATH, '.mat', 'IgnoreCase', true)
    load(IQ_FILE_PATH);
    % Assume variable name is 'IQ' or find the first variable
    if ~exist('IQ', 'var')
        vars = whos('-file', IQ_FILE_PATH);
        if isempty(vars)
            error('No variables found in .mat file');
        end
        load(IQ_FILE_PATH, vars(1).name);
        eval(sprintf('IQ = %s;', vars(1).name));
    end
elseif endsWith(IQ_FILE_PATH, '.npy', 'IgnoreCase', true)
    % For .npy files, you need to use Python to load
    % Option 1: Convert .npy to .mat using Python first
    % Option 2: Use MATLAB's Python interface
    error(['.npy files require Python. ', ...
           'Please convert to .mat first or use Python interface.']);
end

fprintf('Loaded IQ shape: %s\n', mat2str(size(IQ)));

% ========== Step 2: Convert IQ to Correlation ==========
fprintf('\nStep 2: Converting IQ to correlation images...\n');
IQ2corr(IQ, CORR_SAVE_PATH);

% ========== Step 3: Convert Correlation to Depth ==========
fprintf('\nStep 3: Converting correlation to depth...\n');

% Load correlation data
load(CORR_SAVE_PATH);
fprintf('Loaded corr_imgs shape: %s\n', mat2str(size(corr_imgs)));

% Frequency vector: [40, 100/3.3, 100/1.7] MHz
freqVec = [40, 1e2 / 3.3, 1e2 / 1.7] * 1e6;
maxd = 10;  % Maximum depth in meters
nt = 5000;  % Number of depth samples
nf = numel(freqVec);  % Number of frequencies

% Extract cos and sin components
h = corr_imgs;
h0mat = h(1:nf, :, :);      % cos components (I channels)
h90mat = h(nf+1:end, :, :); % sin components (Q channels)

% Create complex correlation images
corr_imgs_complex = h0mat + 1i * h90mat;

% Compute phase images
phase_imgs = angle(corr_imgs_complex);

% Adjust negative phases to [0, 2*pi] range
for fi = 1:nf
    tmp = squeeze(phase_imgs(fi, :, :) < 0);
    phase_imgs(fi, tmp) = 2*pi + phase_imgs(fi, tmp);
end

% Reconstruct corr_imgs (for compatibility)
corr_imgs = cat(1, h0mat, h90mat);

% Create depth range
delayVec = linspace(0, 2*maxd, nt);
depth_range = delayVec / 2;

% Compute depths using PhaseImgs2Depths function
fprintf('Computing depths...\n');
depths = PhaseImgs2Depths(freqVec, phase_imgs, depth_range);

% Save results
fprintf('\nSaving depth data to %s...\n', DEPTH_SAVE_PATH);
save(DEPTH_SAVE_PATH, 'depths', 'freqVec', 'maxd', 'nt', 'nf', 'delayVec', '-v7.3');

fprintf('\n========================================\n');
fprintf('Depth computation completed!\n');
fprintf('Depth map shape: %s\n', mat2str(size(depths)));
fprintf('Depth range: [%.4f, %.4f] meters\n', min(depths(:)), max(depths(:)));
fprintf('========================================\n');


