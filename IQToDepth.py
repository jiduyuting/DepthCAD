# import numpy as np
# from scipy.io import loadmat, savemat


# def IQ2corr(IQ, save_path=None):
#     """
#     Convert IQ predicted by GLRUN to correlation for further processing.
    
#     Parameters:
#     -----------
#     IQ : numpy.ndarray
#         Input IQ data with shape (6, h, w)
#         Order: I30 Q30 I40 Q40 I58 Q58
#     save_path : str, optional
#         Path to save the correlation images as .mat file
#         If None, only returns the correlation images without saving
    
#     Returns:
#     --------
#     corr_imgs : numpy.ndarray
#         Correlation images with shape (6, h, w)
#         Order: Q40 Q30 Q58 I40 I30 I58
#     """
#     # Reorder IQ channels: I30 Q30 I40 Q40 I58 Q58 -> Q40 Q30 Q58 I40 I30 I58
#     # Index mapping:
#     #   IQ[0] = I30 -> corr_imgs[4] = I30
#     #   IQ[1] = Q30 -> corr_imgs[1] = Q30
#     #   IQ[2] = I40 -> corr_imgs[3] = I40
#     #   IQ[3] = Q40 -> corr_imgs[0] = Q40
#     #   IQ[4] = I58 -> corr_imgs[5] = I58
#     #   IQ[5] = Q58 -> corr_imgs[2] = Q58
#     corr_imgs = np.stack([
#         IQ[3],  # Q40
#         IQ[1],  # Q30
#         IQ[5],  # Q58
#         IQ[2],  # I40
#         IQ[0],  # I30
#         IQ[4]   # I58
#     ], axis=0)
    
#     if save_path is not None:
#         savemat(save_path, {"corr_imgs": corr_imgs})
#         print(f"Saved correlation images to {save_path}")
    
#     return corr_imgs


# def phase_imgs_to_depths(freq_vec, phase_maps, depth_range, phase_offsets=None):
#     """
#     This function computes the final depth, given individual phases and the
#     vector of frequencies. Also, given as input is the depth range - which is
#     the set of possible depth values. Essentially it's doing phase unwrapping
#     from multi-freq measurements.
    
#     Parameters:
#     -----------
#     freq_vec : array-like
#         Vector of frequencies (in Hz)
#     phase_maps : numpy array
#         Phase maps with shape (nFreq, nr, nc) or (nr, nc, nFreq)
#     depth_range : array-like
#         Set of possible depth values (in meters)
#     phase_offsets : array-like, optional
#         Phase offsets for each frequency
    
#     Returns:
#     --------
#     depths : numpy array
#         Computed depth map with shape (nr, nc)
#     """
#     # Convert phase_maps to shape (nr, nc, nFreq)
#     if phase_maps.ndim == 3:
#         if phase_maps.shape[0] < phase_maps.shape[2]:
#             # Shape is (nFreq, nr, nc), need to transpose to (nr, nc, nFreq)
#             phase_maps = np.transpose(phase_maps, (1, 2, 0))
    
#     nr, nc, n_freq = phase_maps.shape
    
#     # Computing the phases for every candidate depth
#     candidate_phases = np.zeros((1, len(depth_range), n_freq))
    
#     for i in range(n_freq):
#         depth_range_ = depth_range.copy()
#         if phase_offsets is not None:
#             depth_range_ = depth_range + phase_offsets[i]
        
#         # Multiply depth range by 2 because light traverses the distance twice
#         # Speed of light: 3e8 m/s
#         candidate_phases[0, :, i] = np.mod(
#             2 * np.pi * 2 * depth_range_ / (3e8 / freq_vec[i]), 
#             2 * np.pi
#         )
    
#     # Expand candidate_phases to match spatial dimensions
#     candidate_phases = np.tile(candidate_phases, (nr, 1, 1))
    
#     # Computing the depths
#     depths = np.zeros((nr, nc))
    
#     # To find a depth that fits all phase images across a number of freqs
#     for i in range(nc):  # Consider one column at a time
#         phase_maps_tmp = phase_maps[:, i, :]  # Shape: (nr, nFreq)
#         phase_maps_tmp = np.tile(
#             phase_maps_tmp[:, np.newaxis, :], 
#             (1, len(depth_range), 1)
#         )  # Shape: (nr, len(depth_range), nFreq)
        
#         # Compute error matrix
#         err_mat = np.sum((phase_maps_tmp - candidate_phases) ** 2, axis=2)
        
#         # Find indices of minimum error
#         indices = np.argmin(err_mat, axis=1)
        
#         depths[:, i] = depth_range[indices]
    
#     return depths


# def corr_to_depth():
#     """
#     Main function to convert correlation images to depth maps.
#     Equivalent to the MATLAB script CorrToDepth.m
#     """
#     # Load correlation data
#     print("Loading correlation data...")
#     corr_data = loadmat('/data/pre_student/hcy/GLRUN/corr.mat')
    
#     # MATLAB load command loads all variables, try to find corr_imgs
#     # scipy.io.loadmat returns a dict with keys for each variable
#     # Filter out MATLAB metadata keys (keys starting with '__')
#     data_keys = [k for k in corr_data.keys() if not k.startswith('__')]
    
#     if 'corr_imgs' in corr_data:
#         corr_imgs = corr_data['corr_imgs']
#     elif len(data_keys) == 1:
#         # If only one variable, use it
#         corr_imgs = corr_data[data_keys[0]]
#         print(f"Using variable '{data_keys[0]}' as corr_imgs")
#     else:
#         raise ValueError(f"Could not find 'corr_imgs' in .mat file. Available keys: {data_keys}")
    
#     print(f"Loaded corr_imgs shape: {corr_imgs.shape}")
    
#     # Frequency vector: [40, 100/3.3, 100/1.7] MHz
#     freq_vec = np.array([40, 1e2 / 3.3, 1e2 / 1.7]) * 1e6
    
#     maxd = 10  # Maximum depth in meters
#     nt = 5000  # Number of depth samples
#     nf = len(freq_vec)  # Number of frequencies
    
#     # Extract cos and sin components
#     # MATLAB indexing: h(1:nf,:,:) and h(nf+1:end,:,:)
#     # Python indexing is 0-based, so [:nf] and [nf:]
#     h = corr_imgs.copy()
#     h0mat = h[:nf, :, :]  # cos components
#     h90mat = h[nf:, :, :]  # sin components
    
#     # Create complex correlation images
#     corr_imgs_complex = h0mat + 1j * h90mat
    
#     # Compute phase images
#     phase_imgs = np.angle(corr_imgs_complex)
    
#     # Adjust negative phases to [0, 2*pi] range
#     # MATLAB: phase_imgs(fi,tmp) = 2*pi + phase_imgs(fi,tmp)
#     for fi in range(nf):
#         tmp = phase_imgs[fi, :, :] < 0
#         phase_imgs[fi, tmp] = 2 * np.pi + phase_imgs[fi, tmp]
    
#     # Reconstruct corr_imgs (cos and sin separately) - matching MATLAB cat(1,h0mat,h90mat)
#     corr_imgs = np.concatenate([h0mat, h90mat], axis=0)
    
#     # Create depth range
#     delay_vec = np.linspace(0, 2 * maxd, nt)
#     depth_range = delay_vec / 2
    
#     # Compute depths
#     print("Computing depths...")
#     depths = phase_imgs_to_depths(freq_vec, phase_imgs, depth_range)
    
#     # Save results
#     # MATLAB save command saves all variables, but we'll save depths and other variables
#     print("Saving depth data...")
#     save_dict = {
#         'depths': depths,
#         'freqVec': freq_vec,
#         'maxd': maxd,
#         'nt': nt,
#         'nf': nf,
#         'delayVec': delay_vec
#     }
#     savemat('/data/pre_student/hcy/GLRUN/depth.mat', save_dict)
    
#     print(f"Depth computation completed!")
#     print(f"Depth map shape: {depths.shape}")
#     print(f"Depth range: [{depths.min():.4f}, {depths.max():.4f}] meters")
    
#     return depths


# def IQ_to_depth(IQ, corr_save_path=None, depth_save_path=None):
#     """
#     Complete pipeline: Convert IQ data to depth map.
    
#     Parameters:
#     -----------
#     IQ : numpy.ndarray
#         Input IQ data with shape (6, h, w)
#         Order: I30 Q30 I40 Q40 I58 Q58
#     corr_save_path : str, optional
#         Path to save intermediate correlation images as .mat file
#     depth_save_path : str, optional
#         Path to save final depth map as .mat file
    
#     Returns:
#     --------
#     depths : numpy.ndarray
#         Computed depth map with shape (h, w)
#     """
#     print("=" * 60)
#     print("IQ to Depth Pipeline")
#     print("=" * 60)
    
#     # Step 1: Convert IQ to correlation images
#     print("\nStep 1: Converting IQ to correlation images...")
#     corr_imgs = IQ2corr(IQ, corr_save_path)
#     print(f"Correlation images shape: {corr_imgs.shape}")
    
#     # Step 2: Convert correlation to depth
#     print("\nStep 2: Converting correlation to depth...")
    
#     # Frequency vector: [40, 100/3.3, 100/1.7] MHz
#     freq_vec = np.array([40, 1e2 / 3.3, 1e2 / 1.7]) * 1e6
    
#     maxd = 10  # Maximum depth in meters
#     nt = 5000  # Number of depth samples
#     nf = len(freq_vec)  # Number of frequencies
    
#     # Extract cos and sin components
#     h = corr_imgs.copy()
#     h0mat = h[:nf, :, :]  # cos components (I channels)
#     h90mat = h[nf:, :, :]  # sin components (Q channels)
    
#     # Create complex correlation images
#     corr_imgs_complex = h0mat + 1j * h90mat
    
#     # Compute phase images
#     phase_imgs = np.angle(corr_imgs_complex)
    
#     # Adjust negative phases to [0, 2*pi] range
#     for fi in range(nf):
#         tmp = phase_imgs[fi, :, :] < 0
#         phase_imgs[fi, tmp] = 2 * np.pi + phase_imgs[fi, tmp]
    
#     # Create depth range
#     delay_vec = np.linspace(0, 2 * maxd, nt)
#     depth_range = delay_vec / 2
    
#     # Compute depths
#     print("Computing depths...")
#     depths = phase_imgs_to_depths(freq_vec, phase_imgs, depth_range)
    
#     # Save results if path provided
#     if depth_save_path is not None:
#         print(f"\nSaving depth data to {depth_save_path}...")
#         save_dict = {
#             'depths': depths,
#             'freqVec': freq_vec,
#             'maxd': maxd,
#             'nt': nt,
#             'nf': nf,
#             'delayVec': delay_vec
#         }
#         savemat(depth_save_path, save_dict)
    
#     print(f"\nDepth computation completed!")
#     print(f"Depth map shape: {depths.shape}")
#     print(f"Depth range: [{depths.min():.4f}, {depths.max():.4f}] meters")
    
#     return depths


# if __name__ == '__main__':
#     # Example usage:
#     # Option 1: Convert correlation images to depth (original MATLAB workflow)
#     depths = corr_to_depth()
    
#     # Option 2: Complete pipeline from IQ to depth
#     # Uncomment and modify the following lines to use:
#     # import numpy as np
#     # IQ = np.load('path/to/your/IQ.npy')  # Shape: (6, h, w)
#     # depths = IQ_to_depth(
#     #     IQ,
#     #     corr_save_path='/data/pre_student/hcy/GLRUN/corr.mat',
#     #     depth_save_path='/data/pre_student/hcy/GLRUN/depth.mat'
#     # )

import numpy as np
import scipy.io as sio

def IQ_to_depth(IQ_data, corr_save_path=None, depth_save_path=None):
    """
    Convert 6-channel IQ data (Predicted by GLRUN/DepthCAD) into a metric Depth Map.
    
    This function combines the logic from:
      1. IQ2corr (Python): Reordering channels
      2. CorrToDepth.m (MATLAB): Phase extraction and constants
      3. PhaseImgs2Depths.m (MATLAB): Multi-frequency phase unwrapping (Brute-force solver)

    Args:
        IQ_data (np.ndarray): Shape (6, H, W). Order: I30, Q30, I40, Q40, I58, Q58.
                              Values should be normalized (typically -1 to 1 or similar scale).
        corr_save_path (str, optional): If provided, saves the intermediate correlation .mat file.
        depth_save_path (str, optional): If provided, saves the final depth .npy file.

    Returns:
        np.ndarray: Depth map of shape (H, W) in meters.
    """
    
    # ==========================================
    # 1. Constants & Frequency Configuration
    # ==========================================
    # From CorrToDepth.m
    # freqVec = [40, 1e2 / 3.3, 1e2 / 1.7] * 1e6
    FREQ_VEC = np.array([40e6, (100e6 / 3.3), (100e6 / 1.7)]) 
    C = 3e8 # Speed of light
    MAX_DEPTH = 10.0 # Meters
    NUM_CANDIDATES = 5000 # Resolution of search
    
    # ==========================================
    # 2. Reorder IQ Data (Replaces IQ2corr)
    # ==========================================
    # Input format:  0:I30, 1:Q30, 2:I40, 3:Q40, 4:I58, 5:Q58
    # Target format for MATLAB logic:
    #   First 3 channels: Real parts (Cos/I) for Freqs [40, 30, 58]
    #   Last 3 channels:  Imag parts (Sin/Q) for Freqs [40, 30, 58]
    
    # Indices map based on Freq Vector [40, 30, 58]:
    # 40MHz: I=idx2, Q=idx3
    # 30MHz: I=idx0, Q=idx1
    # 58MHz: I=idx4, Q=idx5
    
    h0mat = np.stack([IQ_data[2], IQ_data[0], IQ_data[4]], axis=0) # Real (I)
    h90mat = np.stack([IQ_data[3], IQ_data[1], IQ_data[5]], axis=0) # Imag (Q)
    
    # Concatenate for compatibility if saving .mat
    corr_imgs = np.concatenate([h0mat, h90mat], axis=0)
    
    if corr_save_path:
        # Save exactly as CorrToDepth.m expects
        sio.savemat(corr_save_path, {"corr_imgs": corr_imgs})

    # ==========================================
    # 3. Extract Phases (From CorrToDepth.m)
    # ==========================================
    # Form complex numbers: Z = I + jQ
    complex_corr = h0mat + 1j * h90mat
    
    # Calculate angle (Phase)
    measured_phases = np.angle(complex_corr) # Shape: (3, H, W) in radians (-pi, pi)
    
    # MATLAB Logic: tmp = squeeze(phase_imgs(fi,:,:)<0); phase_imgs(fi,tmp) = 2*pi + phase_imgs(fi,tmp);
    # This maps phases from [-pi, pi] to [0, 2pi)
    measured_phases = np.where(measured_phases < 0, measured_phases + 2 * np.pi, measured_phases)

    # ==========================================
    # 4. Phase Unwrapping / Depth Solving (PhaseImgs2Depths.m)
    # ==========================================
    # Create candidate depth vector (0 to 10m)
    # MATLAB: delayVec = linspace(0,2*maxd,nt); depths = PhaseImgs2Depths(..., delayVec/2);
    candidate_depths = np.linspace(0, MAX_DEPTH, NUM_CANDIDATES) # Shape: (D,)
    
    num_freqs, H, W = measured_phases.shape
    num_candidates = len(candidate_depths)

    # Generate Theoretical Phases for all candidates
    # Formula: phi = (4 * pi * distance * freq) / c
    # Modulo 2pi to wrap them
    # Shape: (num_freqs, num_candidates)
    
    # Broadcasting: (3, 1) * (1, 5000) -> (3, 5000)
    theoretical_phases = (4 * np.pi * candidate_depths[None, :] * FREQ_VEC[:, None]) / C
    theoretical_phases = np.mod(theoretical_phases, 2 * np.pi)

    # --- Vectorized Search (Replacing MATLAB loops) ---
    
    # Reshape Measured Phases to: (num_freqs, H*W, 1)
    measured_flat = measured_phases.reshape(num_freqs, -1, 1)
    
    # Reshape Theoretical Phases to: (num_freqs, 1, num_candidates)
    theoretical_flat = theoretical_phases[:, None, :]
    
    # Compute L2 Error for every pixel against every candidate depth
    # (Measured - Theoretical)^2
    # Result Shape: (num_freqs, H*W, num_candidates)
    # Note: Using complex phasor distance is often more robust: abs(exp(j*m) - exp(j*t)), 
    # but strictly following the provided MATLAB code which uses squared difference of angles.
    error_matrix = (measured_flat - theoretical_flat) ** 2
    
    # Sum errors across frequencies
    # Shape: (H*W, num_candidates)
    total_error = np.sum(error_matrix, axis=0)
    
    # Find index of minimum error for each pixel
    best_indices = np.argmin(total_error, axis=1)
    
    # Map indices back to depth values
    recovered_depths_flat = candidate_depths[best_indices]
    
    # Reshape back to image dimensions
    depth_map = recovered_depths_flat.reshape(H, W)

    if depth_save_path:
        np.save(depth_save_path, depth_map)
        
    return depth_map

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Create dummy IQ data for testing (6 channels, 240x320)
    H, W = 240, 320
    dummy_iq = np.random.rand(6, H, W).astype(np.float32)
    
    print("Running IQ to Depth conversion...")
    depth = IQ_to_depth(dummy_iq)
    
    print(f"Output Depth Shape: {depth.shape}")
    print(f"Output Depth Range: {depth.min():.4f}m - {depth.max():.4f}m")