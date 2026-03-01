using MathNet.Numerics.IntegralTransforms;
using Complex = System.Numerics.Complex;

namespace WasmDotnet.Tests;

public class PartialBandReconstructorTests
{
    [Fact]
    public void PartialReconstruction_MergedInBandContent_MatchesOriginalInBandSignal()
    {
        const int K = 64;
        const int P = 4;
        const int L = 16;
        const int R = 8;
        const int startBin = 12;
        const int numBlocks = 128;

        float[] h = FilterDesign.GenAnalysisFilter(K, P);
        float[] x = BlsSolver.Solve(h, K, P);

        var reconFromWideband = new PartialBandReconstructor(x, L, R);
        var reconFromInBandOnly = new PartialBandReconstructor(x, L, R);

        int[] inBandBins = [startBin + 1, startBin + 6, startBin + 11];
        int[] outOfBandBins = [2, 29, 52];
        double[] inBandAmplitudes = [1.0, 0.8, 0.6];
        double[] outBandAmplitudes = [0.9, 0.7, 0.5];

        var reconstructedFromWideband = new List<Complex>(numBlocks * R);
        var reconstructedFromInBandOnly = new List<Complex>(numBlocks * R);
        var narrowWideband = new Complex[R];
        var narrowInBand = new Complex[R];

        for (int block = 0; block < numBlocks; block++)
        {
            Complex[] widebandSignal = GenerateWidebandBlock(K, inBandBins, inBandAmplitudes);
            Complex[] outBandSignal = GenerateWidebandBlock(K, outOfBandBins, outBandAmplitudes);
            for (int n = 0; n < K; n++)
                widebandSignal[n] += outBandSignal[n];

            Complex[] inBandOnlySignal = GenerateWidebandBlock(K, inBandBins, inBandAmplitudes);

            var widebandSpectrum = (Complex[])widebandSignal.Clone();
            var inBandSpectrum = (Complex[])inBandOnlySignal.Clone();
            Fourier.Forward(widebandSpectrum, FourierOptions.Matlab);
            Fourier.Forward(inBandSpectrum, FourierOptions.Matlab);

            var selectedFromWideband = new Complex[L];
            var selectedFromInBand = new Complex[L];
            Array.Copy(widebandSpectrum, startBin, selectedFromWideband, 0, L);
            Array.Copy(inBandSpectrum, startBin, selectedFromInBand, 0, L);

            reconFromWideband.ProcessSynthesisBlock(selectedFromWideband, narrowWideband);
            reconFromInBandOnly.ProcessSynthesisBlock(selectedFromInBand, narrowInBand);
            reconstructedFromWideband.AddRange(narrowWideband);
            reconstructedFromInBandOnly.AddRange(narrowInBand);
        }

        int warmupBlocks = x.Length / L;
        int warmupSamples = warmupBlocks * R;

        var stableWideband = reconstructedFromWideband.Skip(warmupSamples).ToArray();
        var stableInBand = reconstructedFromInBandOnly.Skip(warmupSamples).ToArray();

        Assert.Equal(stableInBand.Length, stableWideband.Length);

        double referenceRms = Rms(stableInBand);
        double errorRms = RmsError(stableWideband, stableInBand);
        double relativeError = errorRms / (referenceRms + 1e-12);

        Assert.True(referenceRms > 1e-6, $"Reference signal too small: {referenceRms}");
        Assert.True(relativeError < 1e-6, $"Relative RMS error too high: {relativeError}");
    }

    private static Complex[] GenerateWidebandBlock(int K, int[] bins, double[] amplitudes)
    {
        var signal = new Complex[K];
        for (int n = 0; n < K; n++)
        {
            Complex value = Complex.Zero;
            for (int i = 0; i < bins.Length; i++)
            {
                double angle = 2.0 * Math.PI * bins[i] * n / K;
                value += amplitudes[i] * Complex.Exp(new Complex(0, angle));
            }
            signal[n] = value;
        }

        return signal;
    }

    private static double Rms(Complex[] x)
    {
        double sum = 0.0;
        for (int i = 0; i < x.Length; i++)
            sum += x[i].Magnitude * x[i].Magnitude;

        return Math.Sqrt(sum / x.Length);
    }

    private static double RmsError(Complex[] actual, Complex[] expected)
    {
        double sum = 0.0;
        for (int i = 0; i < actual.Length; i++)
        {
            double mag = (actual[i] - expected[i]).Magnitude;
            sum += mag * mag;
        }

        return Math.Sqrt(sum / actual.Length);
    }
}
