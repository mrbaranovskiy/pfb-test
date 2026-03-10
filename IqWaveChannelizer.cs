using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace WasmDotnet;

public sealed record IqWaveFile(int SampleRate, Complex[] Samples)
{
    private const short BitsPerSample = 16;
    private const short ChannelCount = 2;
    private const short PcmFormat = 1;

    public static IqWaveFile Read(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);

        string riffId = new(reader.ReadChars(4));
        if (!string.Equals(riffId, "RIFF", StringComparison.Ordinal))
            throw new InvalidDataException("Input is not a RIFF file.");

        _ = reader.ReadInt32();

        string waveId = new(reader.ReadChars(4));
        if (!string.Equals(waveId, "WAVE", StringComparison.Ordinal))
            throw new InvalidDataException("Input is not a WAVE file.");

        WaveFormat? format = null;
        byte[]? data = null;

        while (stream.Position <= stream.Length - 8)
        {
            string chunkId = new(reader.ReadChars(4));
            int chunkSize = reader.ReadInt32();
            long nextChunk = stream.Position + chunkSize + (chunkSize & 1);

            switch (chunkId)
            {
                case "fmt ":
                    format = ReadFormatChunk(reader, chunkSize);
                    break;
                case "data":
                    data = reader.ReadBytes(chunkSize);
                    break;
                default:
                    stream.Position += chunkSize;
                    break;
            }

            stream.Position = nextChunk;
        }

        if (format is null)
            throw new InvalidDataException("WAV file is missing a fmt chunk.");
        if (data is null)
            throw new InvalidDataException("WAV file is missing a data chunk.");

        ValidateFormat(format);
        return new IqWaveFile(format.SampleRate, DecodeIqSamples(data, format.BlockAlign));
    }

    public void Write(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: false);

        int blockAlign = ChannelCount * (BitsPerSample / 8);
        int byteRate = SampleRate * blockAlign;
        int dataSize = Samples.Length * blockAlign;

        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + dataSize);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));

        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16);
        writer.Write(PcmFormat);
        writer.Write(ChannelCount);
        writer.Write(SampleRate);
        writer.Write(byteRate);
        writer.Write((short)blockAlign);
        writer.Write(BitsPerSample);

        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataSize);

        double scale = FindComponentPeak(Samples);
        if (scale < 1.0)
            scale = 1.0;

        for (int i = 0; i < Samples.Length; i++)
        {
            short iSample = ToPcm16(Samples[i].Real / scale);
            short qSample = ToPcm16(Samples[i].Imaginary / scale);
            writer.Write(iSample);
            writer.Write(qSample);
        }
    }

    private static WaveFormat ReadFormatChunk(BinaryReader reader, int chunkSize)
    {
        if (chunkSize < 16)
            throw new InvalidDataException("fmt chunk is too small.");

        short audioFormat = reader.ReadInt16();
        short channels = reader.ReadInt16();
        int sampleRate = reader.ReadInt32();
        _ = reader.ReadInt32();
        short blockAlign = reader.ReadInt16();
        short bitsPerSample = reader.ReadInt16();

        int extraBytes = chunkSize - 16;
        if (extraBytes > 0)
            reader.ReadBytes(extraBytes);

        return new WaveFormat(audioFormat, channels, sampleRate, blockAlign, bitsPerSample);
    }

    private static void ValidateFormat(WaveFormat format)
    {
        if (format.AudioFormat != PcmFormat)
            throw new NotSupportedException($"Only PCM WAV files are supported. Found format {format.AudioFormat}.");
        if (format.ChannelCount != ChannelCount)
            throw new NotSupportedException($"Only stereo IQ WAV files are supported. Found {format.ChannelCount} channels.");
        if (format.BitsPerSample != BitsPerSample)
            throw new NotSupportedException($"Only 16-bit WAV files are supported. Found {format.BitsPerSample}-bit.");
        if (format.BlockAlign != ChannelCount * (BitsPerSample / 8))
            throw new InvalidDataException("Unexpected block alignment for 16-bit stereo PCM.");
    }

    private static Complex[] DecodeIqSamples(byte[] data, int blockAlign)
    {
        if (data.Length % blockAlign != 0)
            throw new InvalidDataException("WAV data chunk is not aligned to full IQ frames.");

        int sampleCount = data.Length / blockAlign;
        var samples = new Complex[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            int offset = i * blockAlign;
            short iSample = BitConverter.ToInt16(data, offset);
            short qSample = BitConverter.ToInt16(data, offset + 2);
            samples[i] = new Complex(iSample / 32768.0, qSample / 32768.0);
        }

        return samples;
    }

    private static double FindComponentPeak(IReadOnlyList<Complex> samples)
    {
        double peak = 0.0;
        for (int i = 0; i < samples.Count; i++)
        {
            peak = Math.Max(peak, Math.Abs(samples[i].Real));
            peak = Math.Max(peak, Math.Abs(samples[i].Imaginary));
        }

        return peak;
    }

    private static short ToPcm16(double value)
    {
        double clamped = Math.Clamp(value, -1.0, 0.999969482421875);
        return (short)Math.Round(clamped * short.MaxValue);
    }

    private sealed record WaveFormat(short AudioFormat, short ChannelCount, int SampleRate, short BlockAlign, short BitsPerSample);
}

public sealed record ChannelizationParameters(
    double LowerFrequencyHz,
    double UpperFrequencyHz,
    double CenterFrequencyHz,
    double BandwidthHz,
    int InputSampleRate,
    int OutputSampleRate,
    int DecimationFactor,
    int FilterTapCount);

public sealed record ChannelizationResult(Complex[] Samples, ChannelizationParameters Parameters);

public static class IqChannelizer
{
    private const double OversamplingFactor = 1.25;

    public static ChannelizationResult Channelize(
        IReadOnlyList<Complex> input,
        int inputSampleRate,
        double lowerFrequencyHz,
        double upperFrequencyHz)
    {
        if (input.Count == 0)
            throw new ArgumentException("Input sample buffer is empty.", nameof(input));
        if (inputSampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputSampleRate), "Sample rate must be positive.");
        if (upperFrequencyHz <= lowerFrequencyHz)
            throw new ArgumentException("Upper cut frequency must be greater than lower cut frequency.");

        double nyquist = inputSampleRate / 2.0;
        if (lowerFrequencyHz < -nyquist || upperFrequencyHz > nyquist)
            throw new ArgumentOutOfRangeException(nameof(lowerFrequencyHz), $"Requested band must stay within [{-nyquist}, {nyquist}] Hz.");

        double bandwidth = upperFrequencyHz - lowerFrequencyHz;
        double centerFrequency = (lowerFrequencyHz + upperFrequencyHz) / 2.0;
        double minimumOutputRate = bandwidth * OversamplingFactor;
        int decimation = EstimateDecimation(inputSampleRate, minimumOutputRate);
        int outputSampleRate = inputSampleRate / decimation;

        double passbandEdge = bandwidth / 2.0;
        double outputNyquist = outputSampleRate / 2.0;
        if (passbandEdge >= outputNyquist)
            throw new InvalidOperationException("Requested band is too wide for the estimated decimation.");

        double transitionBudget = outputNyquist - passbandEdge;
        double cutoff = passbandEdge + transitionBudget * 0.35;
        int tapCount = EstimateTapCount(inputSampleRate, transitionBudget);
        double[] taps = DesignLowPass(cutoff, inputSampleRate, tapCount);

        Complex[] mixed = FrequencyShift(input, centerFrequency, inputSampleRate);
        Complex[] output = FilterAndDecimate(mixed, taps, decimation);

        var parameters = new ChannelizationParameters(
            lowerFrequencyHz,
            upperFrequencyHz,
            centerFrequency,
            bandwidth,
            inputSampleRate,
            outputSampleRate,
            decimation,
            tapCount);

        return new ChannelizationResult(output, parameters);
    }

    private static int EstimateDecimation(int inputSampleRate, double minimumOutputRate)
    {
        if (minimumOutputRate > inputSampleRate)
            throw new InvalidOperationException("Requested band exceeds the input sample rate.");

        int maxDecimation = Math.Max(1, (int)Math.Floor(inputSampleRate / minimumOutputRate));
        for (int decimation = maxDecimation; decimation >= 1; decimation--)
        {
            if (inputSampleRate % decimation == 0)
                return decimation;
        }

        return 1;
    }

    private static int EstimateTapCount(int inputSampleRate, double transitionWidthHz)
    {
        double safeTransition = Math.Max(transitionWidthHz, inputSampleRate * 0.0025);
        int tapCount = (int)Math.Ceiling(5.5 * inputSampleRate / safeTransition);
        tapCount = Math.Clamp(tapCount, 63, 2047);
        if (tapCount % 2 == 0)
            tapCount++;

        return tapCount;
    }

    private static double[] DesignLowPass(double cutoffHz, int sampleRate, int tapCount)
    {
        var taps = new double[tapCount];
        double normalizedCutoff = cutoffHz / sampleRate;
        double midpoint = (tapCount - 1) / 2.0;
        double sum = 0.0;

        for (int i = 0; i < tapCount; i++)
        {
            double n = i - midpoint;
            double sinc = Math.Abs(n) < double.Epsilon
                ? 2.0 * normalizedCutoff
                : Math.Sin(2.0 * Math.PI * normalizedCutoff * n) / (Math.PI * n);
            double window = BlackmanHarris(i, tapCount);
            taps[i] = sinc * window;
            sum += taps[i];
        }

        for (int i = 0; i < taps.Length; i++)
            taps[i] /= sum;

        return taps;
    }

    private static Complex[] FrequencyShift(IReadOnlyList<Complex> input, double centerFrequencyHz, int sampleRate)
    {
        var shifted = new Complex[input.Count];
        double phaseStep = -2.0 * Math.PI * centerFrequencyHz / sampleRate;
        Complex rotation = Complex.FromPolarCoordinates(1.0, phaseStep);
        Complex oscillator = Complex.One;

        for (int i = 0; i < input.Count; i++)
        {
            shifted[i] = input[i] * oscillator;
            oscillator *= rotation;

            if ((i & 1023) == 1023)
            {
                double phase = phaseStep * (i + 1L);
                oscillator = Complex.FromPolarCoordinates(1.0, phase);
            }
        }

        return shifted;
    }

    private static Complex[] FilterAndDecimate(Complex[] input, double[] taps, int decimation)
    {
        int halfLength = taps.Length / 2;
        int usableSamples = input.Length - (2 * halfLength);
        if (usableSamples <= 0)
            throw new InvalidOperationException("Input is too short for the designed filter. Use a longer file or a wider channel.");

        int outputLength = 1 + ((usableSamples - 1) / decimation);
        var output = new Complex[outputLength];
        long workEstimate = (long)outputLength * taps.Length;

        if (workEstimate < 131_072 || Environment.ProcessorCount == 1)
        {
            for (int outputIndex = 0; outputIndex < outputLength; outputIndex++)
                output[outputIndex] = FilterOutput(input, taps, decimation, halfLength, outputIndex);

            return output;
        }

        Parallel.For(0, outputLength, outputIndex =>
        {
            output[outputIndex] = FilterOutput(input, taps, decimation, halfLength, outputIndex);
        });

        return output;
    }

    private static Complex FilterOutput(Complex[] input, double[] taps, int decimation, int halfLength, int outputIndex)
    {
        int center = halfLength + (outputIndex * decimation);
        int start = center - halfLength;
        Complex sum = Complex.Zero;

        for (int tap = 0; tap < taps.Length; tap++)
            sum += input[start + tap] * taps[tap];

        return sum;
    }

    private static double BlackmanHarris(int index, int length)
    {
        if (length == 1)
            return 1.0;

        double x = 2.0 * Math.PI * index / (length - 1);
        return 0.35875
               - 0.48829 * Math.Cos(x)
               + 0.14128 * Math.Cos(2.0 * x)
               - 0.01168 * Math.Cos(3.0 * x);
    }
}
