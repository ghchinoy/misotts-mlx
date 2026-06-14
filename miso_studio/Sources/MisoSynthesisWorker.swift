import Foundation
import Combine

/// A thread-safe background worker that handles spawning the local MisoTTS Python backend 
/// as a subprocess, streaming real-time compilation and execution logs, and returning 
/// completed audio assets.
@MainActor
class MisoSynthesisWorker: ObservableObject {
    
    enum AppState: String {
        case idle = "Idle"
        case running = "Synthesizing Speech..."
        case success = "Ready"
        case failed = "Error Occurred"
    }
    
    @Published var appState: AppState = .idle
    @Published var consoleLogs: String = ""
    @Published var generatedAudioURL: URL? = nil
    @Published var generationStats: MisoStats? = nil
    @Published var progressPercent: Double = 0.0
    @Published var progressMessage: String = ""
    
    private var activeProcess: Process? = nil
    private var outputPipe: Pipe? = nil
    private var stdoutBuffer = Data()
    
    /// Resolves the absolute path of the misotts project root relative to this Swift source file.
    private var projectRoot: URL {
        // Tactic 1: Check compile-time file path parent directories (best for development/debug)
        let sourceFile = #filePath
        let sourceURL = URL(fileURLWithPath: sourceFile)
        let devRoot = sourceURL
            .deletingLastPathComponent() // Sources/
            .deletingLastPathComponent() // miso_studio/
            .deletingLastPathComponent() // misotts/ (project root)
        
        if FileManager.default.fileExists(atPath: devRoot.appendingPathComponent(".venv").path) {
            return devRoot
        }
        
        // Tactic 2: Check standard macOS standalone App Bundle location (the folder containing the .app bundle)
        let bundleURL = Bundle.main.bundleURL
        let bundleParentRoot = bundleURL
            .deletingLastPathComponent() // outputs/
            .deletingLastPathComponent() // misotts/
        if FileManager.default.fileExists(atPath: bundleParentRoot.appendingPathComponent(".venv").path) {
            return bundleParentRoot
        }
        
        // Tactic 2b: What if the app bundle is inside the workspace root directly (e.g. outputs/ not present)?
        let directParentRoot = bundleURL.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: directParentRoot.appendingPathComponent(".venv").path) {
            return directParentRoot
        }

        // Tactic 3: Fallback to Current Working Directory
        let cwdPath = FileManager.default.currentDirectoryPath
        let cwdURL = URL(fileURLWithPath: cwdPath)
        if FileManager.default.fileExists(atPath: cwdURL.appendingPathComponent(".venv").path) {
            return cwdURL
        }
        
        // Absolute fallback: return devRoot
        return devRoot
    }
    
    /// Cancels any active speech synthesis task.
    func cancelActiveTask() {
        if let process = activeProcess, process.isRunning {
            process.terminate()
            appendLog("\n🛑 Process terminated by user.")
        }
        activeProcess = nil
        outputPipe = nil
        appState = .idle
        progressPercent = 0.0
        progressMessage = ""
    }
    
    /// Spawns the python backend subprocess for speech synthesis or voice cloning.
    /// - Parameters:
    ///   - text: The input text prompt to synthesize.
    ///   - speaker: The speaker ID.
    ///   - isClone: True if voice cloning should be executed.
    ///   - cloneAudioPath: The path to the cloning prompt audio.
    ///   - clonePromptText: The transcript text of the cloning audio prompt.
    ///   - tempStart: Starting temperature.
    ///   - tempMin: Minimum temperature.
    ///   - tempDecay: Temperature decay steps.
    ///   - cfgScale: Classifier-Free Guidance scale.
    ///   - bypassWatermark: If true, bypasses the watermarking stage.
    ///   - useQuant: If true, uses the 4-bit quantized checkpoint.
    ///   - maxLengthMs: Maximum duration of output audio in milliseconds.
    func synthesize(
        text: String,
        speaker: Int,
        isClone: Bool,
        cloneAudioPath: String?,
        clonePromptText: String?,
        tempStart: Double,
        tempMin: Double,
        tempDecay: Int,
        cfgScale: Double,
        bypassWatermark: Bool,
        useQuant: Bool,
        maxLengthMs: Double = 15000.0
    ) {
        cancelActiveTask()
        appState = .running
        progressPercent = 0.0
        progressMessage = "Initializing model layers..."
        consoleLogs = "============================================================\n"
        consoleLogs += "       MisoTTS Native Subprocess Inference Session           \n"
        consoleLogs += "============================================================\n\n"
        
        // Locate python interpreter in virtual environment
        let pythonInterpreter = projectRoot
            .appendingPathComponent(".venv")
            .appendingPathComponent("bin")
            .appendingPathComponent("python")
            
        guard FileManager.default.fileExists(atPath: pythonInterpreter.path) else {
            appState = .failed
            consoleLogs += "❌ ERROR: Virtual environment Python interpreter not found.\n"
            consoleLogs += "Expected at: \(pythonInterpreter.path)\n"
            consoleLogs += "Please run 'uv sync' in the workspace root to set up dependencies."
            return
        }
        
        let cliScript = projectRoot
            .appendingPathComponent("miso_mlx")
            .appendingPathComponent("miso_mlx_cli.py")
            
        guard FileManager.default.fileExists(atPath: cliScript.path) else {
            appState = .failed
            consoleLogs += "❌ ERROR: MisoTTS MLX CLI script not found at:\n\(cliScript.path)"
            return
        }
        
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: Date())
        
        let detailSuffix = isClone ? "clone" : "spk\(speaker)"
        let paramsStr = "t\(String(format: "%.1f", tempStart))_cfg\(String(format: "%.1f", cfgScale))"
        let slug = makeSlug(from: text)
        let uniqueFilename = "miso_\(timestamp)_\(detailSuffix)_\(paramsStr)\(slug.isEmpty ? "" : "_\(slug)").wav"
        
        let outputsDir = projectRoot.appendingPathComponent("outputs")
        
        // Ensure outputs folder exists
        try? FileManager.default.createDirectory(
            at: outputsDir,
            withIntermediateDirectories: true
        )
        
        let outputWav = outputsDir.appendingPathComponent(uniqueFilename)
        
        // Build arguments
        var args: [String] = [cliScript.path]
        
        if isClone {
            args.append("clone")
            args.append(contentsOf: ["--text", text])
            
            guard let audioPath = cloneAudioPath, !audioPath.isEmpty,
                  let promptText = clonePromptText, !clonePromptText!.isEmpty else {
                appState = .failed
                consoleLogs += "❌ ERROR: Voice Cloning requires both prompt audio and transcript."
                return
            }
            
            args.append(contentsOf: ["--prompt-audio", audioPath])
            args.append(contentsOf: ["--prompt-text", promptText])
        } else {
            args.append("speak")
            args.append(contentsOf: ["--text", text])
            args.append(contentsOf: ["--speaker", String(speaker)])
        }
        
        // MLX execution backend
        args.append("--mlx")
        
        // Quantization checkpoint
        if useQuant {
            args.append("--quant")
        }
        
        // Quality parameters
        args.append(contentsOf: ["--temp-start", String(format: "%.2f", tempStart)])
        args.append(contentsOf: ["--temp-min", String(format: "%.2f", tempMin)])
        args.append(contentsOf: ["--temp-decay-steps", String(tempDecay)])
        args.append(contentsOf: ["--cfg-scale", String(format: "%.2f", cfgScale)])
        
        if bypassWatermark {
            args.append("--no-watermark")
        }
        
        args.append(contentsOf: ["--output", outputWav.path])
        
        // Maximum duration limit
        args.append(contentsOf: ["--max_length_ms", String(format: "%.0f", maxLengthMs)])
        
        // Request machine-readable JSON stats on stdout, logs on stderr
        args.append("--json")
        
        appendLog("🚀 Launching MisoTTS MLX Backend process...")
        appendLog("Interpreter: \(pythonInterpreter.lastPathComponent)")
        appendLog("Command Args: \(args.joined(separator: " "))\n\n")
        
        // Spawn Process (conforming to macOS HIG async guidelines)
        let process = Process()
        process.executableURL = pythonInterpreter
        process.arguments = args
        process.currentDirectoryURL = projectRoot
        
        // Separate Pipes: stdout for JSON payload, stderr for continuous debug logging
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        
        self.activeProcess = process
        
        // Storage for stdout data containing the JSON block
        self.stdoutBuffer = Data()
        
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor in
                self?.handleStdoutData(data)
            }
        }
        
        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            if let str = String(data: data, encoding: .utf8) {
                Task { @MainActor in
                    self?.appendLog(str)
                }
            }
        }
        
        // Handle process exit asynchronously
        process.terminationHandler = { [weak self] completedProcess in
            Task { @MainActor in
                // Remove pipe handlers to prevent resource leaks
                stdoutPipe.fileHandleForReading.readabilityHandler = nil
                stderrPipe.fileHandleForReading.readabilityHandler = nil
                
                // Read any leftover data
                if let leftoverData = try? stdoutPipe.fileHandleForReading.readToEnd(), !leftoverData.isEmpty {
                    self?.handleStdoutData(leftoverData)
                }
                if let leftoverErrorData = try? stderrPipe.fileHandleForReading.readToEnd(), !leftoverErrorData.isEmpty {
                    if let str = String(data: leftoverErrorData, encoding: .utf8) {
                        self?.appendLog(str)
                    }
                }
                
                // If there's any remaining unparsed content in stdoutBuffer, process it
                if let buffer = self?.stdoutBuffer, !buffer.isEmpty {
                    if let lineStr = String(data: buffer, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !lineStr.isEmpty {
                        self?.processStdoutLine(lineStr)
                    }
                }
                
                if completedProcess.terminationStatus == 0 {
                    self?.appState = .success
                    self?.generatedAudioURL = outputWav
                    
                    // Maintain latest studio_output.wav for backwards compatibility and easy access
                    if let latestWav = self?.projectRoot.appendingPathComponent("outputs").appendingPathComponent("studio_output.wav") {
                        let fileManager = FileManager.default
                        if fileManager.fileExists(atPath: latestWav.path) {
                            try? fileManager.removeItem(at: latestWav)
                        }
                        try? fileManager.copyItem(at: outputWav, to: latestWav)
                    }
                    
                    self?.appendLog("\n\n============================================================")
                    self?.appendLog("\n✅ SPEECH SYNTHESIS SUCCESS!")
                    self?.appendLog("\nAudio file archived at: outputs/\(outputWav.lastPathComponent)")
                    self?.appendLog("\nAudio file copied to: outputs/studio_output.wav")
                    self?.appendLog("\n============================================================")
                } else {
                    self?.appState = .failed
                    self?.appendLog("\n\n❌ SPEECH SYNTHESIS FAILED (Exit Code: \(completedProcess.terminationStatus))")
                }
                self?.activeProcess = nil
            }
        }
        
        do {
            try process.run()
        } catch {
            appState = .failed
            consoleLogs += "❌ Failed to launch child process: \(error.localizedDescription)"
            activeProcess = nil
        }
    }
    
    private func makeSlug(from text: String, maxLength: Int = 15) -> String {
        let allowedChars = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " _-"))
        let filtered = text.components(separatedBy: allowedChars.inverted).joined()
        let trimmed = filtered.trimmingCharacters(in: .whitespacesAndNewlines)
        let safeText = trimmed.replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "__", with: "_")
            .lowercased()
        return String(safeText.prefix(maxLength))
    }
    
    private func appendLog(_ text: String) {
        consoleLogs += text
    }
    
    private func parseStats(data: Data) -> MisoStats? {
        guard !data.isEmpty else { return nil }
        do {
            let decoder = JSONDecoder()
            return try decoder.decode(MisoStats.self, from: data)
        } catch {
            print("❌ Failed to decode generation stats JSON: \(error)")
            // Let's dump whatever string was captured for debugging
            if let str = String(data: data, encoding: .utf8) {
                print("Captured stdout: \(str)")
            }
            return nil
        }
    }
    
    private func handleStdoutData(_ data: Data) {
        stdoutBuffer.append(data)
        
        while let newlineIndex = stdoutBuffer.firstIndex(of: 10) {
            let lineData = stdoutBuffer.prefix(through: newlineIndex)
            stdoutBuffer.removeSubrange(..<(newlineIndex + 1))
            
            if let lineStr = String(data: lineData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !lineStr.isEmpty {
                processStdoutLine(lineStr)
            }
        }
    }
    
    private func processStdoutLine(_ line: String) {
        guard let lineData = line.data(using: .utf8) else { return }
        let decoder = JSONDecoder()
        
        if let progressStep = try? decoder.decode(MisoProgressStep.self, from: lineData) {
            let progressFraction = Double(progressStep.step) / Double(progressStep.totalSteps)
            self.progressPercent = progressFraction
            self.progressMessage = "Step \(progressStep.step)/\(progressStep.totalSteps) (CFG: \(String(format: "%.1f", progressStep.cfg)), Temp: \(String(format: "%.2f", progressStep.temp)))"
            return
        }
        
        if let stats = try? decoder.decode(MisoStats.self, from: lineData) {
            self.generationStats = stats
            return
        }
    }
}

// MARK: - Machine-Readable Telemetry Models

struct MisoProgressStep: Codable {
    let type: String
    let step: Int
    let totalSteps: Int
    let cfg: Double
    let temp: Double
    
    enum CodingKeys: String, CodingKey {
        case type, step, cfg, temp
        case totalSteps = "total_steps"
    }
}

struct MisoStats: Codable {
    let status: String
    let backend: String
    let quantizationEnabled: Bool
    let text: String
    let output: String
    let sampleRate: Int
    let stats: StatsDetail
    
    enum CodingKeys: String, CodingKey {
        case status, backend, text, output
        case quantizationEnabled = "quantization_enabled"
        case sampleRate = "sample_rate"
        case stats
    }
}

struct StatsDetail: Codable {
    let generationTimeSec: Double
    let audioDurationSec: Double
    let realTimeFactor: Double
    let totalSteps: Int
    let stepsPerSecond: Double
    let peakMemoryMb: Double
    let memoryIncreaseMb: Double
    let profiling: ProfilingDetail?
    
    enum CodingKeys: String, CodingKey {
        case generationTimeSec = "generation_time_sec"
        case audioDurationSec = "audio_duration_sec"
        case realTimeFactor = "real_time_factor"
        case totalSteps = "total_steps"
        case stepsPerSecond = "steps_per_second"
        case peakMemoryMb = "peak_memory_mb"
        case memoryIncreaseMb = "memory_increase_mb"
        case profiling
    }
}

struct ProfilingDetail: Codable {
    let promptTokenizationTime: Double?
    let warmupTime: Double?
    let generationLoopTime: Double?
    let firstStepTime: Double?
    let subsequentStepsTime: Double?
    let avgSubsequentStepTime: Double?
    let mimiDecodingTime: Double?
    let watermarkingTime: Double?
    
    enum CodingKeys: String, CodingKey {
        case promptTokenizationTime = "prompt_tokenization_time"
        case warmupTime = "warmup_time"
        case generationLoopTime = "generation_loop_time"
        case firstStepTime = "first_step_time"
        case subsequentStepsTime = "subsequent_steps_time"
        case avgSubsequentStepTime = "avg_subsequent_step_time"
        case mimiDecodingTime = "mimi_decoding_time"
        case watermarkingTime = "watermarking_time"
    }
}
