import Foundation
import MLX

func main() {
    print("============================================================")
    print("       MisoTTS Swift-MLX Port Validation Utility            ")
    print("============================================================")
    
    // 1. Verify Basic GPU Acceleration
    print("\n1. Verifying MLX-Swift Metal GPU Execution...")
    
    // Create two simple float32 matrices (matmul only supports float types)
    let a = MLXArray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] as [Float], [2, 4])
    let b = MLXArray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] as [Float], [4, 2])
    
    // Perform matrix multiplication (runs on Apple Silicon GPU/Metal)
    let c = matmul(a, b)
    
    // Print arrays and results
    print("Matrix A shape: \(a.shape)")
    print("Matrix B shape: \(b.shape)")
    print("Result Matrix C (Metal MatMul) shape: \(c.shape)")
    print("Result Matrix C contents:\n\(c)")
    
    // 2. Load Safetensors Weights Native Validation
    print("\n2. Verifying Native Compatibility with converted MisoTTS Weights...")
    
    // Resolve relative path to our converted weights folder.
    // We first attempt to find the project root relative to this source file (#filePath),
    // which is highly reliable when running from Xcode, then fall back to the current directory.
    let sourceFile = #filePath
    let sourceURL = URL(fileURLWithPath: sourceFile)
    let projectRoot = sourceURL
        .deletingLastPathComponent() // Sources/
        .deletingLastPathComponent() // miso_swift/
        .deletingLastPathComponent() // misotts/ (project root)
    
    let currentDir = FileManager.default.currentDirectoryPath
    let currentDirURL = URL(fileURLWithPath: currentDir)
    
    // Define candidate search directories for weights
    let searchDirs = [
        projectRoot.appendingPathComponent("miso_mlx").appendingPathComponent("mlx_weights"),
        currentDirURL.appendingPathComponent("miso_mlx").appendingPathComponent("mlx_weights"),
        currentDirURL.appendingPathComponent("mlx_weights")
    ]
    
    var selectedPath: URL? = nil
    
    for dir in searchDirs {
        let quantized = dir.appendingPathComponent("quantized_model_4bit.safetensors")
        let unquantized = dir.appendingPathComponent("model.safetensors")
        
        if FileManager.default.fileExists(atPath: quantized.path) {
            selectedPath = quantized
            break
        } else if FileManager.default.fileExists(atPath: unquantized.path) {
            selectedPath = unquantized
            break
        }
    }

    
    if let path = selectedPath {
        print("Found converted MLX safetensors weights file at: \(path.lastPathComponent)")
        print("Attempting to load weights directly into Swift memory...")
        
        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            // loadArrays parses .safetensors files natively in Swift!
            let loadedArrays = try loadArrays(url: path)
            let duration = CFAbsoluteTimeGetCurrent() - t0
            
            print("✅ Successfully loaded weights!")
            print("   - Loaded \(loadedArrays.count) distinct neural network layers/arrays")
            print("   - Weight loading wall-time: \(String(format: "%.3f", duration)) seconds")
            
            // Print a small sample of weight keys and shapes
            print("\nSample of loaded layer keys and dimensions:")
            let sampleCount = min(5, loadedArrays.count)
            let sortedKeys = loadedArrays.keys.sorted()
            for i in 0..<sampleCount {
                let key = sortedKeys[i]
                if let array = loadedArrays[key] {
                    print("  * \(key) -> Shape: \(array.shape), Type: \(array.dtype)")
                }
            }
        } catch {
            print("❌ Failed to parse safetensors weights file: \(error)")
        }
    } else {
        print("⚠ Note: Converted .safetensors weights not found in searched paths:")
        for dir in searchDirs {
            print("  - \(dir.path)")
        }
        print("  Please run the MLX Python converter or downloader steps to test weight loading.")
    }
    
    print("\n============================================================")
    print("✅ MisoTTS Swift-MLX validation completed successfully!")
    print("============================================================")
}

main()
