using System;
using System.Collections.Generic;
using System.Globalization;
using WasmDotnet;

namespace WasmDotnet.Demo;

internal static class Program
{
    private static int Main(string[] args)
    {
        try
        {
            CommandLineOptions options = CommandLineOptions.Parse(args);

            Console.WriteLine($"Reading IQ WAV: {options.InputPath}");
            IqWaveFile input = IqWaveFile.Read(options.InputPath);
            Console.WriteLine($"  Sample rate: {input.SampleRate} Hz");
            Console.WriteLine($"  IQ samples:  {input.Samples.Length}");
            Console.WriteLine($"  Duration:    {input.Samples.Length / (double)input.SampleRate:F3} s");

            ChannelizationResult result = IqChannelizer.Channelize(
                input.Samples,
                input.SampleRate,
                options.LowerFrequencyHz,
                options.UpperFrequencyHz);

            ChannelizationParameters parameters = result.Parameters;
            Console.WriteLine();
            Console.WriteLine("Estimated channelization parameters:");
            Console.WriteLine($"  Lower cut:       {parameters.LowerFrequencyHz:F3} Hz");
            Console.WriteLine($"  Upper cut:       {parameters.UpperFrequencyHz:F3} Hz");
            Console.WriteLine($"  Center shift:    {parameters.CenterFrequencyHz:F3} Hz");
            Console.WriteLine($"  Bandwidth:       {parameters.BandwidthHz:F3} Hz");
            Console.WriteLine($"  Decimation:      {parameters.DecimationFactor}");
            Console.WriteLine($"  Output rate:     {parameters.OutputSampleRate} Hz");
            Console.WriteLine($"  Filter taps:     {parameters.FilterTapCount}");
            Console.WriteLine($"  Output IQ count: {result.Samples.Length}");

            Console.WriteLine();
            Console.WriteLine($"Writing channelized IQ WAV: {options.OutputPath}");
            new IqWaveFile(parameters.OutputSampleRate, result.Samples).Write(options.OutputPath);

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            Console.Error.WriteLine();
            Console.Error.WriteLine(CommandLineOptions.Usage);
            return 1;
        }
    }

    private sealed record CommandLineOptions(
        string InputPath,
        double LowerFrequencyHz,
        double UpperFrequencyHz,
        string OutputPath)
    {
        public const string Usage =
            """
            Usage:
              dotnet run --project Demo -- <input.wav> --lf <hz> --uf <hz> --output <output.wav>

            Options:
              --wavefile <path>  Input 16-bit stereo IQ WAV file. Optional if passed positionally.
              --lf <hz>          Lower cut frequency in Hz.
              --uf <hz>          Upper cut frequency in Hz.
              --output <path>    Output path for the channelized 16-bit stereo IQ WAV file.
            """;

        public static CommandLineOptions Parse(string[] args)
        {
            if (args.Length == 0 || Array.Exists(args, static arg => arg is "--help" or "-h"))
                throw new ArgumentException("Missing required arguments.");

            string? inputPath = null;
            string? outputPath = null;
            double? lowerFrequency = null;
            double? upperFrequency = null;

            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i];
                switch (arg)
                {
                    case "--wavefile":
                        inputPath = ReadRequiredValue(args, ref i, "--wavefile");
                        break;
                    case "--lf":
                        lowerFrequency = ParseDouble(ReadRequiredValue(args, ref i, "--lf"), "--lf");
                        break;
                    case "--uf":
                        upperFrequency = ParseDouble(ReadRequiredValue(args, ref i, "--uf"), "--uf");
                        break;
                    case "--output":
                        outputPath = ReadRequiredValue(args, ref i, "--output");
                        break;
                    default:
                        if (arg.StartsWith("--", StringComparison.Ordinal))
                            throw new ArgumentException($"Unknown option '{arg}'.");
                        inputPath ??= arg;
                        break;
                }
            }

            if (string.IsNullOrWhiteSpace(inputPath))
                throw new ArgumentException("Input WAV file is required.");
            if (lowerFrequency is null)
                throw new ArgumentException("--lf is required.");
            if (upperFrequency is null)
                throw new ArgumentException("--uf is required.");
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("--output is required.");

            return new CommandLineOptions(inputPath, lowerFrequency.Value, upperFrequency.Value, outputPath);
        }

        private static string ReadRequiredValue(IReadOnlyList<string> args, ref int index, string option)
        {
            if (index + 1 >= args.Count)
                throw new ArgumentException($"Missing value for {option}.");

            index++;
            return args[index];
        }

        private static double ParseDouble(string value, string option)
        {
            if (!double.TryParse(value, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out double parsed))
                throw new ArgumentException($"Invalid numeric value '{value}' for {option}.");

            return parsed;
        }
    }
}
