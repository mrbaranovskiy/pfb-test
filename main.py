import argparse
import numpy as np
from scipy.linalg import pinv
from scipy.signal import firwin, resample
"""
Minimal PyQt5 GUI for partial-band FM demod using UHD (or simulation).

Dependencies: PyQt5, numpy, scipy, sounddevice, (optional) pyuhd

This main replaces the previous demo script and delegates the reconstructor to
`partial_reconstructor.py` in the same folder.
"""
import sys
import threading
import queue
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sounddevice as sd

from partial_reconstructor import gen_analysis_filter, gen_bls_filter_solver, PartialBandReconstructor, fm_demodulate

try:
    import uhd
except Exception as e:
    # in this environment UHD is expected to be available; if import fails we
    # want an early error rather than toggling a flag.
    raise


class MainWindow(QtWidgets.QWidget):
    # signal to carry spectrum data (freqs, mags)
    spectrumUpdated = QtCore.pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Partial-Band FM Demodulator')
        form_layout = QtWidgets.QFormLayout()

        self.freq = QtWidgets.QLineEdit('98e6')
        self.nchan = QtWidgets.QLineEdit('128')
        self.mfold = QtWidgets.QLineEdit('8')
        self.L = QtWidgets.QLineEdit('8')
        self.R = QtWidgets.QLineEdit('8')
        self.start_bin = QtWidgets.QLineEdit('0')
        self.num_bins = QtWidgets.QLineEdit('16')

        form_layout.addRow('Center Frequency (Hz):', self.freq)
        form_layout.addRow('Num channels K:', self.nchan)
        form_layout.addRow('Decimation / folding P:', self.mfold)
        form_layout.addRow('Num bins L:', self.L)
        form_layout.addRow('Synthesis interp R:', self.R)
        form_layout.addRow('Start bin:', self.start_bin)
        form_layout.addRow('Num bins to extract:', self.num_bins)

        self.start_btn = QtWidgets.QPushButton('Start')
        self.start_btn.clicked.connect(self.start)
        form_layout.addRow(self.start_btn)

        # container layout that holds form and spectrum canvas
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)

        # spectrum plot setup
        self.spectrum_fig = Figure(figsize=(5, 2))
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        main_layout.addWidget(self.spectrum_canvas)

        self.setLayout(main_layout)

        # connect signal
        self.spectrumUpdated.connect(self.update_spectrum)

        # runtime flag
        self.running = False

    def update_spectrum(self, freqs, mags):
        """Slot to update plot when new spectrum data arrives."""
        self.spectrum_ax.cla()
        self.spectrum_ax.plot(freqs, mags)
        self.spectrum_ax.set_title("Narrowband Spectrum")
        self.spectrum_ax.set_xlabel("Normalized frequency")
        self.spectrum_ax.set_ylabel("Magnitude")
        self.spectrum_canvas.draw()


    def start(self):
        if self.running:
            self.running = False
            self.start_btn.setText('Start')
            return
        self.running = True
        self.start_btn.setText('Stop')
        t = threading.Thread(target=self.worker, daemon=True)
        t.start()

    def worker(self):
        fs_audio = 48000
        # parse params
        freq = float(self.freq.text())
        K = int(self.nchan.text())
        P = int(self.mfold.text())
        L = int(self.L.text())
        R = int(self.R.text())
        start_bin = int(self.start_bin.text())
        num_bins = int(self.num_bins.text())

        # design filters and reconstructor
        h = gen_analysis_filter(K, P)
        x = gen_bls_filter_solver(h, K, P)
        recon = PartialBandReconstructor(x, L, R)

        # streaming setup: either UHD or simulated complex tones
        samp_rate = 10e6  # default SDR rate (adjustable)
        block_size = K

        # calculate reconstructor output sample rate:
        # one block represents K samples at samp_rate, recon outputs R samples per block
        recon_rate = R * (samp_rate / float(K))

        # prepare UHD hardware once (outside of inner loop)
        usrp = uhd.usrp.MultiUSRP()
        usrp.set_rx_rate(samp_rate)
        usrp.set_rx_freq(uhd.types.TuneRequest(freq))
        # StreamArgs setup (same as before)
        if hasattr(uhd.usrp, 'StreamArgs'):
            try:
                stream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
            except TypeError:
                try:
                    stream_args = uhd.usrp.StreamArgs('fc32', 'fc32')
                except TypeError:
                    stream_args = uhd.usrp.StreamArgs('fc32')
        elif hasattr(uhd.usrp, 'stream_args'):
            try:
                stream_args = uhd.usrp.stream_args('fc32', 'sc16')
            except TypeError:
                try:
                    stream_args = uhd.usrp.stream_args('fc32', 'fc32')
                except TypeError:
                    stream_args = uhd.usrp.stream_args('fc32')
        else:
            raise RuntimeError('No compatible StreamArgs constructor found')

        rx_streamer = usrp.get_rx_stream(stream_args)

        def get_block():
            nonlocal rx_streamer
            if rx_streamer is not None:
                # read a single block (blocking) using a pre-allocated numpy buffer
                try:
                    buf = np.zeros(block_size, dtype=np.complex64)
                    try:
                        md = uhd.types.RXMetadata()
                    except Exception:
                        md = None

                    if md is not None:
                        got = rx_streamer.recv(buf, md, timeout=0.1)
                    else:
                        got = rx_streamer.recv(buf)

                    if not got:
                        raise RuntimeError('received zero samples from UHD')

                    buffs = buf[:got]
                    return buffs.astype(np.complex128)
                except Exception as e:
                    print('UHD recv failed, falling back to simulation:', e)
                    # disable further UHD attempts by clearing streamer
                    rx_streamer = None

            # simulation: generate multitone complex signal
            t = np.arange(block_size)
            harmonics = [3, 7, 13, 29]
            wide = sum(np.exp(1j * 2 * np.pi * hfreq * t / K) for hfreq in harmonics)
            wide = wide + 0.1 * (np.random.randn(block_size) + 1j * np.random.randn(block_size))
            return wide

        # prepare audio queue and output stream with callback to keep UI thread free
        audio_buf = []  # legacy, still kept if needed elsewhere
        audio_queue = queue.Queue()

        def audio_callback(outdata, frames, time, status):
            # this runs in sounddevice's audio thread
            try:
                data = audio_queue.get_nowait()
                # ensure correct length
                if len(data) < frames:
                    out = np.zeros((frames, 1), dtype=np.float32)
                    out[:len(data), 0] = data
                else:
                    out = data[:frames].reshape(-1, 1)
                outdata[:] = out
            except queue.Empty:
                outdata.fill(0)

        try:
            audio_stream = sd.OutputStream(samplerate=fs_audio,
                                            channels=1,
                                            dtype='float32',
                                            callback=audio_callback)
            audio_stream.start()
        except Exception as e:
            print('audio output stream failed to open:', e)
            audio_stream = None

        # keep a timestamp so we don't flood the GUI with spectrum updates
        last_spec = 0.0
        while self.running:
            wide_block = get_block()
            # FFT -> channels
            channels = np.fft.fft(wide_block)
            sel = channels[start_bin:start_bin + L]
            narrow = recon.process_synthesis_block(sel)
            # FM demod -> instantaneous frequency (radians/sample difference)
            audio = fm_demodulate(narrow)
            audio_real = np.real(audio)

            # resample from reconstructor rate to audio sample rate
            try:
                n_out = int(np.round(len(audio_real) * fs_audio / recon_rate))
                if n_out > 1:
                    audio_resamp = resample(audio_real, n_out)
                else:
                    audio_resamp = audio_real
            except Exception:
                audio_resamp = audio_real

            # simple de-emphasis (75us time-constant) for FM broadcast
            try:
                tau = 75e-6
                alpha = np.exp(-1.0 / (fs_audio * tau))
                deemp = np.zeros_like(audio_resamp)
                for i, v in enumerate(audio_resamp):
                    deemp[i] = alpha * (deemp[i-1] if i>0 else 0.0) + (1 - alpha) * v
                audio_real = deemp
            except Exception:
                audio_real = audio_resamp

            # normalize to -0.5..0.5
            audio_real = audio_real / (np.max(np.abs(audio_real)) + 1e-12) * 0.5
            # send to audio device via persistent stream if available
            if audio_stream is not None:
                try:
                    # enqueue float32 samples (mono)
                    audio_queue.put_nowait(audio_real.astype(np.float32))
                except queue.Full:
                    # drop audio if queue backs up
                    pass
            else:
                # fallback to blocking play once
                try:
                    sd.play(audio_real, samplerate=fs_audio)
                    sd.wait()
                except Exception:
                    pass

            # compute spectrum once per interval and emit to UI thread
            now = time.time()
            if now - last_spec >= 0.1:  # ~10 Hz update cap
                last_spec = now
                nfft = max(256, 1 << (len(narrow) - 1).bit_length())
                fft_vals = np.fft.fft(narrow, n=nfft)
                mags = np.abs(np.fft.fftshift(fft_vals))
                freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0))
                self.spectrumUpdated.emit(freqs, mags)

        sd.stop()
        if 'audio_stream' in locals() and audio_stream is not None:
            try:
                audio_stream.stop()
                audio_stream.close()
            except Exception:
                pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
        
