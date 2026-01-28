% run_IQ_to_Depth.m: Main script to run IQ to Depth pipeline
%
% This script provides a simple interface to convert IQ data to depth maps
% It combines IQ2Corr.m and CorrToDepth.m into one workflow
%
% Usage:
%   1. Modify the IQ_FILE_PATH below to point to your IQ data
%   2. Run: run_IQ_to_Depth
%   3. Check the output at DEPTH_SAVE_PATH

clc; clear; close all;

% ========== Configuration ==========
% Modify these paths according to your setup

% Path to IQ data file
% Options:
%   - .mat file: Load directly
%   - .npy file: Need to convert to .mat first (use Python)
IQ_FILE_PATH = 'IQ.mat';

% Intermediate and output paths
CORR_SAVE_PATH = 'corr.mat';
DEPTH_SAVE_PATH = 'depth.mat';

% ========== Main Pipeline ==========
fprintf('========================================\n');
fprintf('IQ to Depth Pipeline\n');
fprintf('========================================\n');

% Step 1: Load IQ Data
fprintf('\n[Step 1/3] Loading IQ data...\n');
if ~exist(IQ_FILE_PATH, 'file')
    error('IQ file not found: %s\nPlease check the path.', IQ_FILE_PATH);
end

% Load IQ data
if endsWith(IQ_FILE_PATH, '.mat', 'IgnoreCase', true)
    % Get variable names first
    vars = whos('-file', IQ_FILE_PATH);
    if isempty(vars)
        error('No variables found in .mat file');
    end
    
    % Try to find IQ variable (case-insensitive)
    IQ_var_name = [];
    for i = 1:length(vars)
        if strcmpi(vars(i).name, 'IQ')
            IQ_var_name = vars(i).name;
            break;
        end
    end
    
    % If not found, use first variable
    if isempty(IQ_var_name)
        IQ_var_name = vars(1).name;
        fprintf('Warning: IQ variable not found, using first variable: %s\n', IQ_var_name);
    end
    
    % Load the variable
    load(IQ_FILE_PATH, IQ_var_name);
    eval(sprintf('IQ = %s;', IQ_var_name));
    fprintf('Loaded variable: %s\n', IQ_var_name);
else
    error('Only .mat files are supported. For .npy files, convert to .mat first.');
end

% Check IQ shape
IQ_size = size(IQ);
fprintf('IQ shape: %s\n', mat2str(IQ_size));

% Validate IQ shape - should be (6, h, w) or (h, w, 6)
if length(IQ_size) == 2
    error(['IQ data has wrong dimensions: %s. ', ...
           'Expected shape: (6, h, w) or (h, w, 6). ', ...
           'Please check your IQ data format.'], mat2str(IQ_size));
elseif length(IQ_size) == 3
    if IQ_size(1) == 6
        % Shape is (6, h, w) - correct format
        fprintf('IQ format: (6, h, w) - OK\n');
    elseif IQ_size(3) == 6
        % Shape is (h, w, 6) - need to transpose
        fprintf('IQ format: (h, w, 6) - transposing to (6, h, w)...\n');
        IQ = permute(IQ, [3, 1, 2]);
        fprintf('Transposed IQ shape: %s\n', mat2str(size(IQ)));
    else
        error(['IQ data has unexpected dimensions: %s. ', ...
               'Expected first or last dimension to be 6.'], mat2str(IQ_size));
    end
else
    error('IQ data has unexpected number of dimensions: %d', length(IQ_size));
end

% Step 2: Convert IQ to Correlation
fprintf('\n[Step 2/3] Converting IQ to correlation images...\n');

% Reorder IQ channels: I30 Q30 I40 Q40 I58 Q58 -> Q40 Q30 Q58 I40 I30 I58
% Index mapping:
%   IQ(1,:,:) = I30 -> corr_imgs(5,:,:) = I30
%   IQ(2,:,:) = Q30 -> corr_imgs(2,:,:) = Q30
%   IQ(3,:,:) = I40 -> corr_imgs(4,:,:) = I40
%   IQ(4,:,:) = Q40 -> corr_imgs(1,:,:) = Q40
%   IQ(5,:,:) = I58 -> corr_imgs(6,:,:) = I58
%   IQ(6,:,:) = Q58 -> corr_imgs(3,:,:) = Q58
corr_imgs = cat(1, ...
    IQ(4,:,:), ...  % Q40
    IQ(2,:,:), ...  % Q30
    IQ(6,:,:), ...  % Q58
    IQ(3,:,:), ...  % I40
    IQ(1,:,:), ...  % I30
    IQ(5,:,:) ...   % I58
);

% Save correlation images
save(CORR_SAVE_PATH, 'corr_imgs', '-v7.3');
fprintf('Correlation images saved to: %s\n', CORR_SAVE_PATH);
fprintf('Correlation images shape: %s\n', mat2str(size(corr_imgs)));

% Step 3: Convert Correlation to Depth
fprintf('\n[Step 3/3] Converting correlation to depth...\n');

% Load correlation data
load(CORR_SAVE_PATH);
fprintf('Correlation images shape: %s\n', mat2str(size(corr_imgs)));

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

% Reconstruct corr_imgs (for compatibility with existing code)
corr_imgs = cat(1, h0mat, h90mat);

% Create depth range
delayVec = linspace(0, 2*maxd, nt);
depth_range = delayVec / 2;

% Compute depths using PhaseImgs2Depths function
fprintf('Computing depths (this may take a while)...\n');
depths = PhaseImgs2Depths(freqVec, phase_imgs, depth_range);

% Save results
fprintf('\nSaving depth data...\n');
save(DEPTH_SAVE_PATH, 'depths', 'freqVec', 'maxd', 'nt', 'nf', 'delayVec', '-v7.3');

% Display results
fprintf('\n========================================\n');
fprintf('Pipeline completed successfully!\n');
fprintf('========================================\n');
fprintf('Input IQ file: %s\n', IQ_FILE_PATH);
fprintf('Correlation file: %s\n', CORR_SAVE_PATH);
fprintf('Output depth file: %s\n', DEPTH_SAVE_PATH);
fprintf('\nDepth map statistics:\n');
fprintf('  Shape: %s\n', mat2str(size(depths)));
fprintf('  Range: [%.4f, %.4f] meters\n', min(depths(:)), max(depths(:)));
fprintf('  Mean: %.4f meters\n', mean(depths(:)));
fprintf('  Std: %.4f meters\n', std(depths(:)));
fprintf('========================================\n');


