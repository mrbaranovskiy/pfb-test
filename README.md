# WasmDotnet Channelizer Demo

This repository contains two versions of a polyphase channelizer / partial-band reconstructor:

* `main.py` - Python prototype that uses NumPy and SciPy for filter design, FFT, and linear algebra.
* `Channelizer.cs` + `WasmDotnet.csproj` - C# port prepared for .NET using SIMD intrinsics, tensor primitives, and a simple demo harness.

## Getting Started (C#)

### Prerequisites

* [.NET 7 SDK](https://dotnet.microsoft.com/download) or later (older versions may work but 7 is used here).
* (Optional) MathNet.Numerics if you need real FFT or pseudo-inverse support.

### Building

```bash
cd d:/source/WasmDotnet
dotnet restore
dotnet build
```

### Running

```bash
dotnet run --project WasmDotnet.csproj
```

The program will design the analysis/synthesis filters, generate a test wideband signal with a handful of harmonics, and perform one partial-band reconstruction. Output is printed to the console.

To integrate into your .NET project, copy `Channelizer.cs` and reference the `System.Numerics.Tensors` package (and optionally `MathNet.Numerics`).

## Notes

* The current C# solver (`BlsSolver.Solve`) contains placeholders; real linear algebra (SVD/pseudo-inverse) should be added via a library.
* The IFFT in `PartialBandReconstructor` is not implemented; plug in MathNet or another FFT engine.
* Window generation and filtering loops can be vectorized further using `System.Numerics.Vector<T>` or hardware intrinsics.
* `Tensor` types are used minimally; you can expand to dense tensor operations for GPU/accelerator backends.

The Python script remains available for quick experimentation. In addition to the original
`main.py` demo, a PyQt5-based GUI (`main.py`) now provides a simple SDR/FM receiver
front-end. It can stream from a USRP via the UHD driver (if the `pyuhd` binding is
installed) or simulate a wideband signal for testing. The GUI lets you adjust center
frequency, channelizer parameters (K/P/L/R), select a subband, and visualizes the
narrowband spectrum while demodulating FM to your audio device.

## Python Dependencies

```bash
pip install numpy scipy matplotlib pyqt5 sounddevice
# optional: pip install pyuhd       # UHD/USRP support
```

Run the GUI with:

```bash
python main.py
```

On headless systems you may need additional Qt platform plugins (e.g. `qt5-default`,
`qtwayland5`) or run under X11.

The Python code also serves as a reference implementation of the analysis/synthesis
filter design and the polyphase reconstructor used by the C# code.
