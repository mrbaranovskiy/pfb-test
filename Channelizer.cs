using System;
using System.Buffers;
using System.Linq;
using Complex = System.Numerics.Complex;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.IntegralTransforms;

namespace WasmDotnet
{
    public static class FilterDesign
    {
        /// <summary>
        /// Generate a prototype lowpass analysis filter using a Blackman-Harris window.
        /// </summary>
        public static float[] GenAnalysisFilter(int nChan, int mFold)
        {
            int numTaps = nChan * mFold;
            var h = new float[numTaps];
            var window = new float[numTaps];

            // coefficients for Blackman-Harris
            const float a0 = 0.35875f;
            const float a1 = 0.48829f;
            const float a2 = 0.14128f;
            const float a3 = 0.01168f;

            // compute window values.  This is currently written as a simple scalar loop,
            // but it could be rewritten using System.Numerics.Vector<float> or
            // hardware intrinsics to process several taps at once.  For clarity we keep
            // the straightforward version here.
            for (int i = 0; i < numTaps; i++)
            {
                float x = 2.0f * MathF.PI * i / (numTaps - 1);
                window[i] = a0
                            - a1 * MathF.Cos(x)
                            + a2 * MathF.Cos(2 * x)
                            - a3 * MathF.Cos(3 * x);
            }

            float cutoff = 1.0f / nChan; // normalized nyquist
            float mid = (numTaps - 1) / 2.0f;

            // build sinc * window
            for (int i = 0; i < numTaps; i++)
            {
                float arg = (i - mid) * cutoff * MathF.PI;
                float sinc = arg == 0 ? 1.0f : MathF.Sin(arg) / arg;
                h[i] = sinc * window[i];
            }

            return h;
        }
    }

    public static class BlsSolver
    {
        /// <summary>
        /// Solve for the synthesis filter x given analysis prototype h using SVD-based pseudo-inverse.
        /// </summary>
        public static float[] Solve(float[] h, int M, int N)
        {
            int Q1 = h.Length;
            int Q2 = Q1;
            int lHalf = (int)MathF.Floor((Q1 - 1) / (float)M);
            int numConstraints = (2 * lHalf + 1) * N;

            // Build constraint matrix H and target vector T using MathNet
            var H = Matrix<double>.Build.Dense(numConstraints, Q2, 0.0);
            var T = Vector<double>.Build.Dense(numConstraints, 0.0);

            int rowIdx = 0;
            for (int u = -lHalf; u <= lHalf; u++)
            {
                if (u == 0)
                {
                    T[rowIdx] = 1.0 / M;
                }
                // else T[rowIdx:rowIdx+N] remains 0

                for (int k = 0; k < N; k++)
                {
                    for (int q = k; q < Q2; q += N)
                    {
                        int shifted = q + u * M;
                        if (shifted >= 0 && shifted < Q2)
                            H[rowIdx, shifted] = h[q];
                    }
                    rowIdx++;
                }
            }

            // Compute pseudo-inverse using SVD: x = pinv(H) * T
            var pinvH = H.PseudoInverse();
            var x_vec = pinvH.Multiply(T);

            // Convert back to float[]
            var x = new float[Q2];
            for (int i = 0; i < Q2; i++)
                x[i] = (float)x_vec[i];

            return x;
        }
    }

    public class PartialBandReconstructor
    {
        private readonly float[] x;
        private readonly int L;
        private readonly int R;
        private readonly int Mfold;
        private readonly Complex[] ifftWork;
        private readonly Complex[,] polyPhases;
        private readonly Complex[,] buffer;
        private int blockIndex;

        public PartialBandReconstructor(float[] synthFilterX, int L_bins, int R_interp)
        {
            x = synthFilterX;
            L = L_bins;
            R = R_interp;
            Mfold = x.Length / L;
            ifftWork = new Complex[L];

            polyPhases = new Complex[Mfold, L];
            buffer = new Complex[Mfold, L];

            // reshape x into polyphase branches
            for (int m = 0; m < Mfold; m++)
            for (int l = 0; l < L; l++)
                polyPhases[m, l] = x[m * L + l];
        }

        public void ProcessSynthesisBlock(ReadOnlySpan<Complex> selectedBins, Span<Complex> output)
        {
            if (selectedBins.Length != L)
                throw new ArgumentException($"Must provide exactly {L} bins.");
            if (output.Length < R)
                throw new ArgumentException($"Output span must have at least {R} samples.");

            Complex[] correctedBuffer = ArrayPool<Complex>.Shared.Rent(L);
            try
            {
                Span<Complex> corrected = correctedBuffer.AsSpan(0, L);
                for (int k = 0; k < L; k++)
                {
                    double angle = 2.0 * Math.PI * blockIndex * k * R / (double)L;
                    corrected[k] = selectedBins[k] * Complex.Exp(new Complex(0, angle));
                }

                corrected.CopyTo(ifftWork);

                // Perform IFFT using MathNet
                Fourier.Inverse(ifftWork, FourierOptions.Matlab);

                // shift buffer down
                for (int m = Mfold - 1; m > 0; m--)
                    for (int l = 0; l < L; l++)
                        buffer[m, l] = buffer[m - 1, l];

                for (int l = 0; l < L; l++)
                    buffer[0, l] = ifftWork[l];

                for (int l = 0; l < R; l++)
                {
                    Complex sum = Complex.Zero;
                    for (int m = 0; m < Mfold; m++)
                        sum += buffer[m, l] * polyPhases[m, l];
                    output[l] = sum;
                }

                blockIndex++;
            }
            finally
            {
                correctedBuffer.AsSpan(0, L).Clear();
                ArrayPool<Complex>.Shared.Return(correctedBuffer);
            }
        }

        [Obsolete("Allocates an output array. Prefer ProcessSynthesisBlock(ReadOnlySpan<Complex>, Span<Complex>) for hot paths.")]
        public Complex[] ProcessSynthesisBlock(Complex[] selectedBins)
        {
            var output = new Complex[R];
            ProcessSynthesisBlock(selectedBins, output);
            return output;
        }
    }
}
