using System;
using System.Linq;
using Complex = System.Numerics.Complex;
using MathNet.Numerics.IntegralTransforms;

namespace WasmDotnet.Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            // Parameters: match Python demo
            int K = 64;       // Number of frequency bins (channels)
            int P = 4;        // Decimation / folding factor
            int L = 16;       // Subband width to reconstruct
            int R = 8;        // Synthesis interpolation
            int startBin = 10; // Which bin to start extracting from

            // 1. Design analysis and synthesis filters
            Console.WriteLine("Designing filters...");
            float[] h = FilterDesign.GenAnalysisFilter(K, P);
            float[] x = BlsSolver.Solve(h, K, P);
            Console.WriteLine($"  Analysis filter: {h.Length} taps");
            Console.WriteLine($"  Synthesis filter: {x.Length} taps");

            var recon = new PartialBandReconstructor(x, L, R);

            // 2. Generate a time-domain wideband signal with harmonics
            Console.WriteLine($"\nGenerating wideband signal ({K} samples)...");
            var timeDomain = new Complex[K];
            int[] harmonics = { 3, 7, 13, 29 };
            for (int hidx = 0; hidx < harmonics.Length; hidx++)
            {
                int hfreq = harmonics[hidx];
                for (int k = 0; k < K; k++)
                {
                    timeDomain[k] += Complex.Exp(new Complex(0, 2 * Math.PI * hfreq * k / K));
                }
            }
            Console.WriteLine($"  Added {harmonics.Length} harmonic components");

            // 3. Perform FFT to get frequency-domain channels
            Console.WriteLine("\nPerforming FFT (analysis)...");
            var fullBand = new Complex[K];
            Array.Copy(timeDomain, fullBand, K);
            Fourier.Forward(fullBand, FourierOptions.Matlab);

            // 4. Extract a subband (L consecutive bins starting at startBin)
            Console.WriteLine($"Extracting subband: bins [{startBin}, {startBin + L})...");
            var selected = new Complex[L];
            Array.Copy(fullBand, startBin, selected, 0, L);

            // Show magnitudes of selected bins
            double avgMag = selected.Average(c => c.Magnitude);
            Console.WriteLine($"  Selected bins avg magnitude: {avgMag:F4}");

            // 5. Synthesize back to time domain
            Console.WriteLine("\nPerforming synthesis (IFFT + polyphase filtering)...");
            var narrow = new Complex[R];
            recon.ProcessSynthesisBlock(selected, narrow);

            Console.WriteLine($"\n=== Results ===");
            Console.WriteLine($"Input: {K} frequency bins");
            Console.WriteLine($"Extracted: {L} bins from [{startBin}, {startBin + L})");
            Console.WriteLine($"Reconstructed: {narrow.Length} time-domain samples");
            Console.WriteLine($"Reconstruction signal magnitude (avg): {narrow.Average(c => c.Magnitude):F4}");

            // 6. Optional: Show first few samples
            Console.WriteLine($"\nFirst 5 reconstructed samples (Real, Imag):");
            for (int i = 0; i < Math.Min(5, narrow.Length); i++)
            {
                Console.WriteLine($"  [{i}] = ({narrow[i].Real:F4}, {narrow[i].Imaginary:F4})");
            }
        }
    }
}
