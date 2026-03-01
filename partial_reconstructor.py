import numpy as np
from scipy.linalg import pinv
from scipy.signal import firwin


def gen_analysis_filter(n_chan, m_fold):
    num_taps = n_chan * m_fold
    cutoff = 1.0 / n_chan
    h = firwin(num_taps, cutoff, window='blackmanharris', pass_zero=True)
    return h


def gen_bls_filter_solver(h, M, N):
    Q1 = len(h)
    Q2 = Q1
    l_half = int(np.floor((Q1 - 1) / M))
    
    # Build H^T @ H and H^T @ T directly without storing full H matrix
    HTH = np.zeros((Q2, Q2))
    HTT = np.zeros(Q2)
    
    for u in range(-l_half, l_half + 1):
        t_val = 1.0 / M if u == 0 else 0.0
        for k in range(N):
            indices = np.arange(k, Q2, N)
            shift = u * M
            valid_indices = indices + shift
            valid_mask = (valid_indices >= 0) & (valid_indices < len(h))
            
            if np.any(valid_mask):
                h_row = np.zeros(Q2)
                h_row[valid_indices[valid_mask]] = h[indices[valid_mask]]
                HTH += np.outer(h_row, h_row)
                HTT += h_row * t_val
    
    x = np.linalg.solve(HTH, HTT)
    return x


class PartialBandReconstructor:
    def __init__(self, synth_filter_x, L_bins, R_interp):
        self.x = np.asarray(synth_filter_x)
        self.L = int(L_bins)
        self.R = int(R_interp)
        self.M_fold = len(self.x) // self.L
        self.poly_phases = self.x.reshape((self.M_fold, self.L))
        self.buffer = np.zeros((self.M_fold, self.L), dtype=np.complex128)
        self.block_index = 0

    def process_synthesis_block(self, selected_bins):
        selected_bins = np.asarray(selected_bins)
        assert len(selected_bins) == self.L
        k_indices = np.arange(self.L)
        phase_correction = np.exp(1j * 2 * np.pi * self.block_index * k_indices * self.R / self.L)
        corrected_bins = selected_bins * phase_correction
        ifft_out = np.fft.ifft(corrected_bins)
        self.buffer = np.roll(self.buffer, 1, axis=0)
        self.buffer[0, :] = ifft_out
        filtered_branches = np.sum(self.buffer * self.poly_phases, axis=0)
        output_stream = filtered_branches[:self.R]
        self.block_index += 1
        return output_stream


def fm_demodulate(complex_samples):
    # Angle-difference FM demodulator
    phase = np.angle(complex_samples)
    dphase = np.diff(np.unwrap(phase))
    # prepend zero to keep length
    return np.concatenate(([0.0], dphase))
