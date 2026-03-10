using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.IntegralTransforms;

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
    int FilterTapCount,
    int FftSize,
    int FoldFactor,
    int SelectedBinCount,
    int ReconstructionRate,
    int StartBin);

public sealed record ChannelizationResult(Complex[] Samples, ChannelizationParameters Parameters);

public static class IqChannelizer
{
    private const double OversamplingFactor = 1.25;
    private const int DefaultFoldFactor = 4;
    private const int TargetSelectedBins = 16;
    private static readonly int[] SupportedFftSizes = [64, 128, 256];
    private static readonly ConcurrentDictionary<(int FftSize, int FoldFactor), float[]> SynthesisFilterCache = new();

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

        ChannelizationParameters parameters = EstimateParameters(input.Count, inputSampleRate, lowerFrequencyHz, upperFrequencyHz);
        float[] synthFilter = GetSynthesisFilter(parameters.FftSize, parameters.FoldFactor);
        var reconstructor = new PartialBandReconstructor(synthFilter, parameters.SelectedBinCount, parameters.ReconstructionRate);
        Complex[] output = ChannelizeWithReconstructor(input, parameters, reconstructor);

        return new ChannelizationResult(output, parameters);
    }

    private static ChannelizationParameters EstimateParameters(
        int inputLength,
        int inputSampleRate,
        double lowerFrequencyHz,
        double upperFrequencyHz)
    {
        double bandwidth = upperFrequencyHz - lowerFrequencyHz;
        double centerFrequency = (lowerFrequencyHz + upperFrequencyHz) / 2.0;
        int fftSize = ChooseFftSize(inputLength, inputSampleRate, bandwidth);
        double binWidth = inputSampleRate / (double)fftSize;
        int selectedBinCount = NextPowerOfTwo((int)Math.Ceiling((bandwidth * fftSize / inputSampleRate) * OversamplingFactor));
        selectedBinCount = Math.Clamp(selectedBinCount, 4, fftSize);

        int reconstructionRate = selectedBinCount;
        int decimationFactor = fftSize / reconstructionRate;
        int outputSampleRate = inputSampleRate / decimationFactor;
        int startBin = EstimateStartBin(fftSize, inputSampleRate, centerFrequency, selectedBinCount);
        int filterTapCount = fftSize * DefaultFoldFactor;

        return new ChannelizationParameters(
            lowerFrequencyHz,
            upperFrequencyHz,
            centerFrequency,
            bandwidth,
            inputSampleRate,
            outputSampleRate,
            decimationFactor,
            filterTapCount,
            fftSize,
            DefaultFoldFactor,
            selectedBinCount,
            reconstructionRate,
            startBin);
    }

    private static Complex[] ChannelizeWithReconstructor(
        IReadOnlyList<Complex> input,
        ChannelizationParameters parameters,
        PartialBandReconstructor reconstructor)
    {
        int blockCount = input.Count / parameters.FftSize;
        if (blockCount == 0)
            throw new InvalidOperationException($"Input is too short for FFT size {parameters.FftSize}. Need at least one full block.");

        var output = new Complex[blockCount * parameters.ReconstructionRate];
        var fftBuffer = new Complex[parameters.FftSize];
        var selectedBins = new Complex[parameters.SelectedBinCount];
        var narrowband = new Complex[parameters.ReconstructionRate];

        for (int blockIndex = 0; blockIndex < blockCount; blockIndex++)
        {
            int inputOffset = blockIndex * parameters.FftSize;
            for (int i = 0; i < parameters.FftSize; i++)
                fftBuffer[i] = input[inputOffset + i];

            Fourier.Forward(fftBuffer, FourierOptions.Matlab);
            SelectShiftedBins(fftBuffer, parameters.StartBin, selectedBins);
            reconstructor.ProcessSynthesisBlock(selectedBins, narrowband);
            Array.Copy(narrowband, 0, output, blockIndex * parameters.ReconstructionRate, parameters.ReconstructionRate);
        }

        return output;
    }

    private static float[] GetSynthesisFilter(int fftSize, int foldFactor)
    {
        return SynthesisFilterCache.GetOrAdd((fftSize, foldFactor), key =>
        {
            float[] analysis = FilterDesign.GenAnalysisFilter(key.FftSize, key.FoldFactor);
            return BlsSolver.Solve(analysis, key.FftSize, key.FoldFactor);
        });
    }

    private static int ChooseFftSize(int inputLength, int inputSampleRate, double bandwidthHz)
    {
        int bestFftSize = 0;
        double bestScore = double.MaxValue;

        for (int i = 0; i < SupportedFftSizes.Length; i++)
        {
            int candidate = SupportedFftSizes[i];
            if (candidate > inputLength)
                continue;

            double binsAcrossBand = bandwidthHz * candidate / inputSampleRate;
            double score = Math.Abs(binsAcrossBand - TargetSelectedBins);
            if (score < bestScore)
            {
                bestScore = score;
                bestFftSize = candidate;
            }
        }

        if (bestFftSize == 0)
            throw new InvalidOperationException("Input is too short for the supported reconstructor FFT sizes.");

        return bestFftSize;
    }

    private static int EstimateStartBin(int fftSize, int inputSampleRate, double centerFrequencyHz, int selectedBinCount)
    {
        double shiftedCenter = (centerFrequencyHz + (inputSampleRate / 2.0)) * fftSize / inputSampleRate;
        int centerBin = (int)Math.Round(shiftedCenter);
        int startBin = centerBin - (selectedBinCount / 2);
        return Math.Clamp(startBin, 0, fftSize - selectedBinCount);
    }

    private static void SelectShiftedBins(Complex[] fftBins, int startBin, Span<Complex> selectedBins)
    {
        int fftSize = fftBins.Length;
        int halfFft = fftSize / 2;

        for (int i = 0; i < selectedBins.Length; i++)
        {
            int shiftedIndex = startBin + i;
            int sourceIndex = (shiftedIndex + halfFft) % fftSize;
            selectedBins[i] = fftBins[sourceIndex];
        }
    }

    private static int NextPowerOfTwo(int value)
    {
        int result = 1;
        while (result < value)
            result <<= 1;

        return result;
    }
}
