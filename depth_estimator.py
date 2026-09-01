import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

class DepthEstimator:
    def __init__(self, maxd=10.0, nt=5000):
        """
        初始化时预计算理论相位矩阵，并永久保存在实例内存中。
        """
        self.freqVec = np.array([40, 1e2 / 3.3, 1e2 / 1.7], dtype=np.float32) * 1e6
        self.nf = len(self.freqVec)
        
        delayVec = np.linspace(0, 2 * maxd, nt, dtype=np.float32)
        self.DepthRange = delayVec / 2.0
        
        c = 3e8
        CandidatePhases = np.empty((nt, self.nf), dtype=np.float32)
        for fi in range(self.nf):
            CandidatePhases[:, fi] = 2.0 * np.pi * 2.0 * self.DepthRange / (c / self.freqVec[fi])
            
        C_I = np.cos(CandidatePhases)
        C_Q = np.sin(CandidatePhases)
        
        C_concat = np.hstack([C_I, C_Q])
        self.C_T = C_concat.T  # 形状 (2*nf, nt)

    def process(self, IQ):
        """
        极速处理输入矩阵 h

        Input IQ order: [I30, Q30, I40, Q40, I58, Q58] (indices 0-5)
        Output should be: [I40, I30, I58] (Real) and [Q40, Q30, Q58] (Imag)

        According to IQ_to_depth:
          h0mat = np.stack([IQ[2], IQ[0], IQ[4]], axis=0)  # [I40, I30, I58] - Real part
          h90mat = np.stack([IQ[3], IQ[1], IQ[5]], axis=0)  # [Q40, Q30, Q58] - Imag part
        """
        h = np.stack([
            IQ[2], IQ[0], IQ[4],  # [I40, I30, I58] - Real parts (P_I)
            IQ[3], IQ[1], IQ[5]   # [Q40, Q30, Q58] - Imag parts (P_Q)
        ], axis=0)

        if np.iscomplexobj(h):
            P_I = np.real(h).astype(np.float32)
            P_Q = np.imag(h).astype(np.float32)
        else:
            P_I = h[:self.nf, :, :].astype(np.float32)
            P_Q = h[self.nf:, :, :].astype(np.float32)

        _, nr, nc = P_I.shape
        N = nr * nc

        P_I_flat = P_I.transpose(1, 2, 0).reshape(N, self.nf)
        P_Q_flat = P_Q.transpose(1, 2, 0).reshape(N, self.nf)

        # 归一化
        amp = np.sqrt(P_I_flat**2 + P_Q_flat**2) + 1e-12
        P_I_flat /= amp
        P_Q_flat /= amp
        
        P_concat = np.hstack([P_I_flat, P_Q_flat])

        # 极速矩阵乘法分块处理
        Depths_flat = np.empty(N, dtype=np.float32)
        chunk_size = 38400 
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            P_chunk = P_concat[i:end]
            
            # 使用实例缓存的 self.C_T 和 self.DepthRange
            score = np.dot(P_chunk, self.C_T) 
            idx = np.argmax(score, axis=1)
            Depths_flat[i:end] = self.DepthRange[idx]

        return Depths_flat.reshape((nr, nc))


class DepthEstimatorTorch:
    def __init__(self, maxd=10.0, nt=5000, device=None):
        if torch is None:
            raise ImportError("DepthEstimatorTorch requires torch, but torch could not be imported.")
        if device is None:
            device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.freqVec = np.array([40, 1e2 / 3.3, 1e2 / 1.7], dtype=np.float32) * 1e6
        self.nf = len(self.freqVec)
        
        delayVec = np.linspace(0, 2 * maxd, nt, dtype=np.float32)
        DepthRange_np = delayVec / 2.0
        
        c = 3e8
        CandidatePhases = np.empty((nt, self.nf), dtype=np.float32)
        for fi in range(self.nf):
            CandidatePhases[:, fi] = 2.0 * np.pi * 2.0 * DepthRange_np / (c / self.freqVec[fi])
            
        C_I = np.cos(CandidatePhases)
        C_Q = np.sin(CandidatePhases)
        C_concat = np.hstack([C_I, C_Q])
        
        self.C_T = torch.tensor(C_concat.T, dtype=torch.float32, device=self.device)
        self.DepthRange = torch.tensor(DepthRange_np, dtype=torch.float32, device=self.device)

    def process(self, IQ):
        # Ensure IQ is on the correct device
        if IQ.device != self.device:
            IQ = IQ.to(self.device)
            # Re-create tensors on the correct device if IQ was moved
            self.C_T = self.C_T.to(self.device)
            self.DepthRange = self.DepthRange.to(self.device)

        B, C, H, W = IQ.shape

        assert C >= 6, f"严重错误: IQ 张量需要至少 6 个通道，但当前仅收到 {C} 个！"

        N_per_batch = H * W
        N_total = B * N_per_batch

        idx_tensor = torch.tensor([2, 0, 4, 3, 1, 5], device=self.device, dtype=torch.long)
        h = torch.index_select(IQ, dim=1, index=idx_tensor)

        P_I = h[:, :self.nf, :, :]
        P_Q = h[:, self.nf:, :, :]

        # 归一化
        amp = torch.sqrt(P_I**2 + P_Q**2) + 1e-12
        P_I = P_I / amp
        P_Q = P_Q / amp

        P_I_flat = P_I.permute(0, 2, 3, 1).reshape(N_total, self.nf)
        P_Q_flat = P_Q.permute(0, 2, 3, 1).reshape(N_total, self.nf)
        P_concat = torch.cat([P_I_flat, P_Q_flat], dim=-1)

        depths_list = []
        chunk_size = N_per_batch 
        
        for i in range(0, N_total, chunk_size):
            end = min(i + chunk_size, N_total)
            P_chunk = P_concat[i:end]
            
            score = torch.matmul(P_chunk, self.C_T) 
            
            prob = torch.softmax(score * 50.0, dim=1) 
            depth_chunk = torch.sum(prob * self.DepthRange, dim=1)
            
            depths_list.append(depth_chunk)

        Depths_flat = torch.cat(depths_list, dim=0)

        return Depths_flat.reshape(B, H, W)


# class DepthEstimatorTorchArgmax:
#     """与 IQ_to_depth 算法一致的 PyTorch 版本 (使用 argmax 而非 softmax)"""
#     def __init__(self, maxd=10.0, nt=5000, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.freqVec = np.array([40, 1e2 / 3.3, 1e2 / 1.7], dtype=np.float32) * 1e6
#         self.nf = len(self.freqVec)

#         delayVec = np.linspace(0, 2 * maxd, nt, dtype=np.float32)
#         DepthRange_np = delayVec / 2.0

#         c = 3e8
#         CandidatePhases = np.empty((nt, self.nf), dtype=np.float32)
#         for fi in range(self.nf):
#             CandidatePhases[:, fi] = 2.0 * np.pi * 2.0 * DepthRange_np / (c / self.freqVec[fi])

#         C_I = np.cos(CandidatePhases)
#         C_Q = np.sin(CandidatePhases)
#         C_concat = np.hstack([C_I, C_Q])

#         self.C_T = torch.tensor(C_concat.T, dtype=torch.float32, device=self.device)
#         self.DepthRange = torch.tensor(DepthRange_np, dtype=torch.float32, device=self.device)

#     def process(self, IQ):
#         """使用 argmax 的深度估计（与 IQ_to_depth 一致）"""
#         if IQ.device != self.device:
#             IQ = IQ.to(self.device)
#             self.C_T = self.C_T.to(self.device)
#             self.DepthRange = self.DepthRange.to(self.device)

#         B, C, H, W = IQ.shape
#         assert C >= 6, f"严重错误: IQ 张量需要至少 6 个通道，但当前仅收到 {C} 个！"

#         N_per_batch = H * W
#         N_total = B * N_per_batch

#         idx_tensor = torch.tensor([2, 0, 4, 3, 1, 5], device=self.device, dtype=torch.long)
#         h = torch.index_select(IQ, dim=1, index=idx_tensor)

#         P_I = h[:, :self.nf, :, :]
#         P_Q = h[:, self.nf:, :, :]

#         # 归一化
#         amp = torch.sqrt(P_I**2 + P_Q**2) + 1e-12
#         P_I = P_I / amp
#         P_Q = P_Q / amp

#         P_I_flat = P_I.permute(0, 2, 3, 1).reshape(N_total, self.nf)
#         P_Q_flat = P_Q.permute(0, 2, 3, 1).reshape(N_total, self.nf)
#         P_concat = torch.cat([P_I_flat, P_Q_flat], dim=-1)

#         depths_list = []
#         chunk_size = N_per_batch

#         for i in range(0, N_total, chunk_size):
#             end = min(i + chunk_size, N_total)
#             P_chunk = P_concat[i:end]

#             score = torch.matmul(P_chunk, self.C_T)
#             # 使用 argmax（与 IQ_to_depth 一致）
#             idx = torch.argmax(score, dim=1)
#             depth_chunk = self.DepthRange[idx]

#             depths_list.append(depth_chunk)

#         Depths_flat = torch.cat(depths_list, dim=0)
#         return Depths_flat.reshape(B, H, W)


if __name__ == '__main__':
    estimator = DepthEstimator()
    estimator_torch = DepthEstimatorTorch()
    
    iq_np = np.load("/data/pre_student/hcy/Unrolling_NCGTV/results/compound_L_FLAT/iq/iso12233.npy")[:, 500:740, 500:820].astype(np.float32)
    iq = np.stack([iq_np, iq_np], 0)
    iq = torch.from_numpy(iq)

    import time
    t0 = time.time()
    depth_ori = estimator.process(iq_np)
    t1 = time.time()
    print(t1 - t0)   
    
    depth_new = estimator_torch.process(iq)


    depth_new1 = depth_new[0].detach().cpu().numpy()
    depth_new2 = depth_new[0].detach().cpu().numpy()
    diff = np.abs(depth_ori - depth_new1)
    print(diff.shape, diff.min(), diff.max(), diff.mean())


