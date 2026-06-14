# MisoTTS Native Swift-MLX Port Validation Utility

A standalone, lightweight Swift package and developer reference demonstrating how to build native Apple Silicon applications using the **MisoTTS 8B** model weights with Apple's **`mlx-swift`** framework.

This validation utility executes native Metal GPU operations and verifies zero-copy memory-mapped loading of converted MisoTTS `.safetensors` weights directly into Swift memory.

---

## Key Features

1. **Metal GPU Acceleration Test**
   * Computes a floating-point matrix multiplication (`matmul`) natively on Apple Silicon’s unified GPU using the Metal shading language.
   * Confirms your local Swift-MLX toolchain is correctly configured and linked to the hardware GPU backend.

2. **Native Zero-Copy Weight Parsing**
   * Leverages the MLX-Swift `loadArrays(url:)` engine to natively parse converted `model.safetensors` or `quantized_model_4bit.safetensors` files.
   * Resolves, maps, and reads **over 1,000 neural network weights** in under **20 milliseconds** utilizing standard OS-level memory mapping (`mmap`).

3. **Source-Relative Resource Location**
   * Uses `#filePath` at compile-time to automatically walk up to the repository root. This allows Xcode to seamlessly locate your converted local weights regardless of where its active build directories or DerivedData products are located.

---

## Prerequisites

* **OS:** macOS 14.0 (Sonoma) or newer.
* **Hardware:** Apple Silicon Mac (M-series chip: M1, M2, M3, M4, or newer).
* **Software:** Xcode 15.0+ (or Swift 5.9+ command-line tools).

---

## Getting Started

### Option A: Open and Run with Xcode (Recommended)

1. **Launch Xcode:** Open Xcode, choose **File -> Open...**, and select the `miso_swift` directory (or its `Package.swift` file).
2. **Resolve Dependencies:** Xcode will automatically fetch, resolve, and compile the `mlx-swift` package version `0.10.0+`.
3. **Select Target:** Choose the executable product **MisoSwiftDemo** next to your active Mac run destination in the top-left scheme selector.
4. **Execute Utility:** Press **`Cmd + R`** (or select **Product -> Run**) to build and run.

> [!TIP]
> **Resolving Cached MatMul Type Crashes:**
> If you built the project previously and Xcode crashes with the error:
> `Fatal error: [matmul] Only inexact types are supported but int32 and int32 were provided`
> It means Xcode is running a stale, cached build before array types were explicitly enforced as `as [Float]`.
> Simply select **Product -> Clean Build Folder** (**`Cmd + Shift + K`**) in Xcode to wipe the stale cache, and re-run.

---

### Option B: Build and Run via Swift Package Manager (CLI)

Open your terminal, navigate to the `miso_swift` folder, and compile/run using Swift's native compiler:

```bash
cd miso_swift
swift run
```

*(Note: Executing from the terminal may log sandboxing warnings if OS permissions block loading the default `.metallib` shader library outside of Xcode's packaged environment.)*

---

## Technical Performance Architecture

### Why are safetensors loaded in 19 milliseconds?
Traditionally, loading neural network checkpoints (e.g., PyTorch `.pt` or `.bin` files via pickle) requires expensive serialization and deserialization steps, parsing structured payloads, and copying byte streams into secondary memory allocations.

In contrast, `.safetensors` combined with MLX-Swift uses Apple Silicon's **Unified Memory Architecture (UMA)** and memory-mapping (`mmap`). When `loadArrays` is invoked:
1. The metadata header of the safetensors file is read to identify the layout, shape, and byte-offsets of each tensor array.
2. The operating system creates a direct virtual memory mapping to those offsets on disk.
3. **Zero Copying occurs:** No tensor weights are copied into memory allocations on startup. The arrays are referenced directly on disk, and the operating system page-faults them directly into Apple Silicon GPU Unified Memory on demand as layers are called during inference.

This allows native Swift-MLX apps to startup instantly with zero overhead, making it ideal for client-side deployments on macOS and iOS.
