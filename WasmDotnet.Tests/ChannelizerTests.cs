using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

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

public class IqWaveFileTests
{
    [Fact]
    public void ReadWrite_RoundTripsStereo16BitIqWav()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.wav");
        try
        {
            Complex[] samples =
            [
                new Complex(0.25, -0.5),
                new Complex(-0.75, 0.125),
                new Complex(0.0, 0.0),
                new Complex(0.99, -0.99)
            ];

            var wav = new IqWaveFile(48_000, samples);
            wav.Write(path);

            IqWaveFile roundTrip = IqWaveFile.Read(path);

            Assert.Equal(48_000, roundTrip.SampleRate);
            Assert.Equal(samples.Length, roundTrip.Samples.Length);

            for (int i = 0; i < samples.Length; i++)
            {
                Assert.True(Math.Abs(samples[i].Real - roundTrip.Samples[i].Real) < 1e-4);
                Assert.True(Math.Abs(samples[i].Imaginary - roundTrip.Samples[i].Imaginary) < 1e-4);
            }
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}

public class IqChannelizerTests
{
    [Fact]
    public void Channelize_CompositeSignal_MatchesInBandOnlyReference()
    {
        const int sampleRate = 48_000;
        const int sampleCount = 48_000;
        const double lowerCut = 2_000.0;
        const double upperCut = 4_000.0;

        Complex[] inBand = GenerateTone(sampleRate, sampleCount, 3_000.0, 0.8);
        Complex[] composite = GenerateTone(sampleRate, sampleCount, 3_000.0, 0.8);
        Complex[] outOfBand = GenerateTone(sampleRate, sampleCount, 15_000.0, 0.7);

        for (int i = 0; i < composite.Length; i++)
            composite[i] += outOfBand[i];

        ChannelizationResult extractedFromComposite = IqChannelizer.Channelize(composite, sampleRate, lowerCut, upperCut);
        ChannelizationResult extractedFromReference = IqChannelizer.Channelize(inBand, sampleRate, lowerCut, upperCut);

        Assert.Equal(extractedFromReference.Parameters.OutputSampleRate, extractedFromComposite.Parameters.OutputSampleRate);
        Assert.Equal(extractedFromReference.Samples.Length, extractedFromComposite.Samples.Length);

        int trim = extractedFromComposite.Parameters.FilterTapCount / extractedFromComposite.Parameters.DecimationFactor;
        Complex[] stableComposite = extractedFromComposite.Samples.Skip(trim).Take(extractedFromComposite.Samples.Length - (2 * trim)).ToArray();
        Complex[] stableReference = extractedFromReference.Samples.Skip(trim).Take(extractedFromReference.Samples.Length - (2 * trim)).ToArray();

        Assert.NotEmpty(stableComposite);
        Assert.Equal(stableReference.Length, stableComposite.Length);

        double referenceRms = Rms(stableReference);
        double errorRms = RmsError(stableComposite, stableReference);
        double relativeError = errorRms / (referenceRms + 1e-12);

        Assert.True(relativeError < 0.02, $"Relative RMS error too high: {relativeError}");
    }

    private static Complex[] GenerateTone(int sampleRate, int sampleCount, double frequencyHz, double amplitude)
    {
        var samples = new Complex[sampleCount];
        for (int n = 0; n < sampleCount; n++)
        {
            double phase = 2.0 * Math.PI * frequencyHz * n / sampleRate;
            samples[n] = amplitude * Complex.FromPolarCoordinates(1.0, phase);
        }

        return samples;
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
