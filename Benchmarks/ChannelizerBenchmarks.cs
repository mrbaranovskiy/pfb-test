using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System;
using System.Numerics;

namespace WasmDotnet.Benchmarks
{
    [MemoryDiagnoser]
    public class ChannelizerBenchmarks
    {
        private float[] hPrototype;
        private float[] synthX;
        private PartialBandReconstructor recon;
        private Complex[] selectedBins;
        private Complex[] outputBuffer;

        [Params(64, 256)]
        public int Nchan;

        [GlobalSetup]
        public void Setup()
        {
            int mFold = Nchan / 16; // just choose something
            hPrototype = FilterDesign.GenAnalysisFilter(Nchan, mFold);
            synthX = BlsSolver.Solve(hPrototype, Nchan, mFold);
            int L = 32;
            int R = 16;
            recon = new PartialBandReconstructor(synthX, L, R);
            selectedBins = new Complex[L];
            outputBuffer = new Complex[R];
            var rng = new Random(123);
            for (int i = 0; i < L; i++)
                selectedBins[i] = new Complex(rng.NextDouble(), rng.NextDouble());
        }

        [Benchmark]
        public float[] GenerateAnalysisFilter() => FilterDesign.GenAnalysisFilter(Nchan, Nchan/16);

        [Benchmark]
        public float[] SolveBls() => BlsSolver.Solve(hPrototype, Nchan, Nchan/16);

        [Benchmark]
        public Complex ProcessBlock()
        {
            recon.ProcessSynthesisBlock(selectedBins, outputBuffer);
            return outputBuffer[0];
        }
    }

    class Program
    {
        static void Main(string[] args) => BenchmarkRunner.Run<ChannelizerBenchmarks>();
    }
}
