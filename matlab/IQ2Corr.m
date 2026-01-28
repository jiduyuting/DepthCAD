function IQ2corr(IQ, save_path)
% IQ2corr: Convert IQ predicted by GLRUN to correlation for further MATLAB process
%
% Input:
%   IQ: numpy array (loaded from .npy or .mat file)
%       shape: (6, h, w), order: I30 Q30 I40 Q40 I58 Q58
%   save_path: string, path to save correlation images as .mat file
%
% Output:
%   corr_imgs: numpy array
%       shape: (6, h, w), order: Q40 Q30 Q58 I40 I30 I58
%   Saved to .mat file at save_path

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

% Save to .mat file
save(save_path, 'corr_imgs', '-v7.3');
fprintf('Saved correlation images to %s\n', save_path);

end
