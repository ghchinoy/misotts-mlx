import SwiftUI

struct ContentView: View {
    @StateObject private var worker = MisoSynthesisWorker()
    @StateObject private var player = MisoAudioPlayer()
    
    // Synthesis Parameters
    @State private var textPrompt: String = "Hello from MisoTTS! This is a real-time, local Metal-accelerated synthesis session on my Apple Silicon GPU."
    @State private var selectedSpeaker: Int = 0
    
    // Advanced Configuration
    @State private var tempStart: Double = 0.70
    @State private var tempMin: Double = 0.40
    @State private var tempDecay: Int = 30
    @State private var cfgScale: Double = 2.00
    @State private var bypassWatermark: Bool = false
    @State private var useQuant: Bool = true
    @State private var maxAudioLengthSec: Double = 15.0
    
    // Voice Cloning
    @State private var isCloningMode: Bool = false
    @State private var cloneAudioPath: String = ""
    @State private var clonePromptText: String = ""
    
    // Focus States (macos-hig-interaction Keyboard centricity)
    @FocusState private var isTextPromptFocused: Bool
    
    var body: some View {
        HSplitView {
            // LEFT COLUMN: Controls & Input
            VStack(alignment: .leading, spacing: 0) {
                // Fixed Header & Text Input Area (NOT scrollable, preventing any scroll-interaction conflicts)
                VStack(alignment: .leading, spacing: 14) {
                    // Header branding
                    HStack(spacing: 8) {
                        Image(systemName: "waveform.path.ecg")
                            .font(.title2)
                            .foregroundColor(.accentColor)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("MisoTTS Studio")
                                .font(.title3.bold())
                            Text("Apple Silicon MLX Workspace")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // Section 1: Speech Text Input
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Speech Script Input")
                            .font(.headline)
                            .foregroundColor(.primary)
                        
                        TextEditor(text: $textPrompt)
                            .font(.body)
                            .frame(height: 120)
                            .focused($isTextPromptFocused)
                            .scrollContentBackground(.hidden)
                            .padding(6)
                            .background(Color(NSColor.controlBackgroundColor))
                            .cornerRadius(6)
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                            )
                        
                        // Interactive speaking-duration estimation and cutoff warning
                        if wordCount > 0 {
                            HStack(spacing: 10) {
                                Image(systemName: isLengthWarningActive ? "exclamationmark.triangle.fill" : "hourglass.badge.plus")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(isLengthWarningActive ? .orange : .accentColor)
                                
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 6) {
                                        Text("\(wordCount) words")
                                            .font(.caption.bold())
                                            .foregroundColor(.secondary)
                                        Text("•")
                                            .font(.caption)
                                            .foregroundColor(.secondary.opacity(0.5))
                                        Text(String(format: "Est. Speaking Time: ~%.1fs", estimatedSpeakingDuration))
                                            .font(.caption.bold())
                                            .foregroundColor(isLengthWarningActive ? .orange : .primary)
                                    }
                                    
                                    if isLengthWarningActive {
                                        Text("Warning: Exceeds slider limit (\(String(format: "%.1f", maxAudioLengthSec))s). Speech will cut off mid-word!")
                                            .font(.system(size: 10, weight: .medium))
                                            .foregroundColor(.orange)
                                    } else {
                                        Text("Comfortably fits within your \(String(format: "%.1f", maxAudioLengthSec))s limit.")
                                            .font(.system(size: 10))
                                            .foregroundColor(.secondary)
                                    }
                                }
                                
                                Spacer()
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(isLengthWarningActive ? Color.orange.opacity(0.12) : Color.accentColor.opacity(0.06))
                            .cornerRadius(6)
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(isLengthWarningActive ? Color.orange.opacity(0.3) : Color.accentColor.opacity(0.15), lineWidth: 1)
                            )
                            .transition(.opacity.combined(with: .move(edge: .top)))
                            .animation(.easeInOut, value: isLengthWarningActive)
                        }
                        
                        Text("Tip: Use Speaker ID inline prefixes (e.g. '[0] Hello! [1] Welcome back.') for multi-speaker dialogs.")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 20)
                .padding(.bottom, 16)
                
                Divider()
                
                // Scrollable parameter panel
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        // Section 2: Mode Toggle (Speak vs Clone)
                        Picker("Synthesis Mode", selection: $isCloningMode) {
                        Text("Preset Speaker Embeddings").tag(false)
                        Text("Zero-Shot Voice Cloning").tag(true)
                    }
                    .pickerStyle(.segmented)
                    
                    if !isCloningMode {
                        // Speaker Presets
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Select Speaker Embeddings")
                                .font(.headline)
                            
                            HStack {
                                Text("Active Speaker ID:")
                                    .font(.subheadline)
                                Spacer()
                                Picker("", selection: $selectedSpeaker) {
                                    ForEach(0...30, id: \.self) { id in
                                        Text("Speaker \(id)").tag(id)
                                    }
                                }
                                .frame(width: 140)
                            }
                        }
                    } else {
                        // Voice Cloning Section
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Voice Cloning Configuration")
                                .font(.headline)
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Prompt Reference Audio (.wav):")
                                    .font(.caption.bold())
                                HStack {
                                    TextField("Absolute path to 16kHz WAV file...", text: $cloneAudioPath)
                                        .textFieldStyle(.roundedBorder)
                                    Button("Browse...") {
                                        selectLocalWavFile()
                                    }
                                }
                            }
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Prompt Reference Transcript:")
                                    .font(.caption.bold())
                                TextField("Exact words spoken in the prompt audio...", text: $clonePromptText)
                                    .textFieldStyle(.roundedBorder)
                            }
                        }
                        .padding(10)
                        .background(Color.secondary.opacity(0.06))
                        .cornerRadius(8)
                        .transition(.slide.combined(with: .opacity))
                    }
                    
                    Divider()
                    
                    // Section 3: Model & Inference Controls
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Model & Generation Parameters")
                            .font(.headline)
                        
                        // Toggle Group
                        HStack(spacing: 20) {
                            Toggle("4-bit Quantized", isOn: $useQuant)
                                .toggleStyle(.checkbox)
                                .help("Saves 10GB RAM and speeds up synthesis up to 3.8x using 4-bit local weights.")
                            
                            Toggle("Bypass Watermark", isOn: $bypassWatermark)
                                .toggleStyle(.checkbox)
                                .help("Removes default SilentCipher 44.1kHz acoustic watermark from the generated file.")
                        }
                        
                        // Temperature start slider
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Start Temperature:")
                                Spacer()
                                Text(String(format: "%.2f", tempStart))
                                    .foregroundColor(.secondary)
                            }
                            .font(.subheadline)
                            Slider(value: $tempStart, in: 0.1...1.5)
                        }
                        
                        // Temperature min slider
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Minimum Temperature:")
                                Spacer()
                                Text(String(format: "%.2f", tempMin))
                                    .foregroundColor(.secondary)
                            }
                            .font(.subheadline)
                            Slider(value: $tempMin, in: 0.0...1.0)
                        }
                        
                        // Temp decay steps slider
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Temperature Decay Steps:")
                                Spacer()
                                Text("\(tempDecay) steps")
                                    .foregroundColor(.secondary)
                            }
                            .font(.subheadline)
                            Slider(value: Binding(
                                get: { Double(tempDecay) },
                                set: { tempDecay = Int($0) }
                            ), in: 5...100)
                        }
                        
                        // CFG Guidance slider
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Classifier-Free Guidance (CFG):")
                                Spacer()
                                Text(String(format: "%.2f", cfgScale))
                                    .foregroundColor(.secondary)
                            }
                            .font(.subheadline)
                            Slider(value: $cfgScale, in: 0.5...5.0)
                        }
                        
                        // Max Audio Duration slider
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Max Audio Duration:")
                                Spacer()
                                Text(String(format: "%.1f seconds", maxAudioLengthSec))
                                    .foregroundColor(.secondary)
                            }
                            .font(.subheadline)
                            Slider(value: $maxAudioLengthSec, in: 5.0...60.0, step: 0.5)
                        }
                    }
                    
                    Spacer(minLength: 20)
                    
                    // Section 4: Synthesize / Cancel Button (pulsating)
                    if worker.appState == .running {
                        Button(action: {
                            worker.cancelActiveTask()
                        }) {
                            HStack {
                                Image(systemName: "stop.fill")
                                Text("Cancel Speech Synthesis")
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(PulsatingGradientButtonStyle(isRunning: true))
                    } else {
                        Button(action: {
                            worker.synthesize(
                                text: textPrompt,
                                speaker: selectedSpeaker,
                                isClone: isCloningMode,
                                cloneAudioPath: cloneAudioPath,
                                clonePromptText: clonePromptText,
                                tempStart: tempStart,
                                tempMin: tempMin,
                                tempDecay: tempDecay,
                                cfgScale: cfgScale,
                                bypassWatermark: bypassWatermark,
                                useQuant: useQuant,
                                maxLengthMs: maxAudioLengthSec * 1000.0
                            )
                        }) {
                            HStack {
                                Image(systemName: "waveform")
                                Text("Synthesize Speech (Metal GPU)")
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(PulsatingGradientButtonStyle(isRunning: false))
                        .disabled(textPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    
                    // Reset Button (macos-hig-interaction Menu Bar fallback)
                    Button("Reset Parameters to Defaults") {
                        resetParameters()
                    }
                    .buttonStyle(.borderless)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .font(.caption)
                    
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
            }
            .frame(minWidth: 360, idealWidth: 420, maxWidth: 600)
            .background(Color(NSColor.windowBackgroundColor).opacity(0.95))
            
            // RIGHT COLUMN: Active Monitor, Live Console, and Audio Player
            VStack(spacing: 0) {
                
                // Status Panel
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("System Status")
                            .font(.caption.bold())
                            .foregroundColor(.secondary)
                        
                        HStack(spacing: 8) {
                            Circle()
                                .fill(statusColor)
                                .frame(width: 10, height: 10)
                            Text(worker.appState.rawValue)
                                .font(.headline)
                        }
                    }
                    
                    Spacer()
                    
                    if worker.appState == .running {
                        VStack(alignment: .trailing, spacing: 4) {
                            ProgressView(value: worker.progressPercent, total: 1.0)
                                .progressViewStyle(.linear)
                                .frame(width: 180)
                            Text(worker.progressMessage)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(16)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
                
                Divider()
                
                // Real-Time Monospace Console Log Panel
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(worker.consoleLogs)
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(Color(white: 0.9))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(16)
                            .id("consoleText")
                    }
                    .background(Color(white: 0.08))
                    .onChange(of: worker.consoleLogs) {
                        withAnimation {
                            proxy.scrollTo("consoleText", anchor: .bottom)
                        }
                    }
                }
                
                Divider()
                
                // Native Audio Player Panel
                VStack(spacing: 12) {
                    HStack {
                        Image(systemName: "music.note")
                            .foregroundColor(.secondary)
                        Text("Active Output Audio Asset")
                            .font(.headline)
                        Spacer()
                    }
                    
                    if let stats = worker.generationStats {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: "gauge.with.needle.fill")
                                    .foregroundColor(.accentColor)
                                Text("Metal GPU Inference Telemetry")
                                    .font(.subheadline.bold())
                                Spacer()
                                Text(stats.quantizationEnabled ? "4-Bit INT4" : "16-Bit BF16")
                                    .font(.caption2.bold())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(stats.quantizationEnabled ? Color.orange.opacity(0.2) : Color.blue.opacity(0.2))
                                    .foregroundColor(stats.quantizationEnabled ? .orange : .blue)
                                    .cornerRadius(4)
                            }
                            
                            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                                GridRow {
                                    Text("Real-Time Factor (RTF):")
                                        .foregroundColor(.secondary)
                                    Text(String(format: "%.3fx (%@)", stats.stats.realTimeFactor, stats.stats.realTimeFactor < 1.0 ? "Faster than Real-Time" : "Slower"))
                                        .bold()
                                        .foregroundColor(stats.stats.realTimeFactor < 1.0 ? .green : .orange)
                                    
                                    Text("Inference Speed:")
                                        .foregroundColor(.secondary)
                                    Text(String(format: "%.1f frames/sec", stats.stats.stepsPerSecond))
                                        .bold()
                                }
                                
                                GridRow {
                                    Text("Total Latency:")
                                        .foregroundColor(.secondary)
                                    Text(String(format: "%.2fs", stats.stats.generationTimeSec))
                                        .bold()
                                    
                                    Text("Peak RAM:")
                                        .foregroundColor(.secondary)
                                    Text(String(format: "%.1f MB", stats.stats.peakMemoryMb))
                                        .bold()
                                }
                            }
                            .font(.caption)
                            
                            if let prof = stats.stats.profiling {
                                Divider()
                                    .padding(.vertical, 2)
                                
                                HStack(spacing: 12) {
                                    if let warmup = prof.warmupTime, warmup > 0.05 {
                                        Label(String(format: "JIT Warmup: %.2fs", warmup), systemImage: "bolt.horizontal.fill")
                                    }
                                    if let firstStep = prof.firstStepTime {
                                        Label(String(format: "First Step JIT: %.2fs", firstStep), systemImage: "clock.fill")
                                    }
                                    if let avgStep = prof.avgSubsequentStepTime {
                                        Label(String(format: "Avg Step: %.1fms", avgStep * 1000.0), systemImage: "arrow.right.circle.fill")
                                    }
                                }
                                .font(.system(size: 9, weight: .medium, design: .monospaced))
                                .foregroundColor(.secondary)
                            }
                        }
                        .padding(12)
                        .background(Color.secondary.opacity(0.08))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
                        )
                        .transition(.move(edge: .top).combined(with: .opacity))
                    }
                    
                    if let audioURL = worker.generatedAudioURL {
                        VStack(spacing: 10) {
                            
                            // Audio details card
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(audioURL.lastPathComponent)
                                        .font(.subheadline.bold())
                                    Text("Local AI-synthesized 24kHz Audio")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Text(HIGFormatter.formatTime(player.duration))
                                    .font(.system(.body, design: .monospaced))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.secondary.opacity(0.12))
                                    .cornerRadius(4)
                            }
                            .padding(10)
                            .background(Color.secondary.opacity(0.06))
                            .cornerRadius(6)
                            
                            // Scrubbing Progress Bar
                            VStack(spacing: 4) {
                                Slider(value: Binding(
                                    get: { player.currentTime },
                                    set: { player.seek(to: $0) }
                                ), in: 0...max(0.1, player.duration))
                                .disabled(!player.isPlaying && player.duration == 0)
                                
                                HStack {
                                    Text(HIGFormatter.formatTime(player.currentTime))
                                    Spacer()
                                    Text(HIGFormatter.formatTime(player.duration))
                                }
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundColor(.secondary)
                            }
                            
                            // Playback controls
                            HStack(spacing: 20) {
                                // Volume
                                Image(systemName: player.volume == 0 ? "speaker.slash.fill" : "speaker.wave.2.fill")
                                    .foregroundColor(.secondary)
                                    .frame(width: 20)
                                Slider(value: $player.volume, in: 0...1)
                                    .frame(width: 70)
                                
                                Spacer()
                                
                                // Play / Pause
                                if player.isPlaying {
                                    Button(action: { player.pause() }) {
                                        Image(systemName: "pause.circle.fill")
                                            .font(.system(size: 32))
                                    }
                                    .buttonStyle(.plain)
                                } else {
                                    Button(action: { player.play() }) {
                                        Image(systemName: "play.circle.fill")
                                            .font(.system(size: 32))
                                    }
                                    .buttonStyle(.plain)
                                }
                                
                                Spacer()
                                
                                // Speed Rate multiplier picker
                                Menu {
                                    ForEach([0.5, 0.75, 1.0, 1.25, 1.5, 2.0], id: \.self) { rate in
                                        Button("\(String(format: "%.2fx", rate))") {
                                            player.speed = Float(rate)
                                        }
                                    }
                                } label: {
                                    Text(String(format: "%.2fx Speed", player.speed))
                                        .font(.caption2.bold())
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 3)
                                        .background(Color.secondary.opacity(0.15))
                                        .cornerRadius(4)
                                }
                                .menuStyle(.borderlessButton)
                                .frame(width: 80)
                            }
                            
                        }
                        .padding(14)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
                        )
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                        .onAppear {
                            player.load(url: audioURL)
                        }
                        .onChange(of: worker.generatedAudioURL) { _, newURL in
                            if let url = newURL {
                                player.load(url: url)
                            }
                        }
                    } else {
                        // Empty/No audio state
                        VStack(spacing: 8) {
                            Image(systemName: "waveform.badge.minus")
                                .font(.system(size: 24))
                                .foregroundColor(.secondary)
                            Text("No Audio Generated Yet")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Text("Enter a script and click 'Synthesize' to produce speech locally.")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 30)
                        .background(Color.secondary.opacity(0.03))
                        .cornerRadius(8)
                    }
                    
                }
                .padding(16)
                .background(Color(NSColor.windowBackgroundColor))
                
            }
            .frame(minWidth: 480, idealWidth: 640, maxWidth: .infinity)
            .background(Color(white: 0.12))
            
        }
        .frame(minWidth: 850, minHeight: 650)
        .onAppear {
            #if os(macOS)
            NSApplication.shared.activate(ignoringOtherApps: true)
            #endif
            isTextPromptFocused = true
        }
        .onReceive(NotificationCenter.default.publisher(for: Notification.Name("MisoTTS_DoSynthesis"))) { _ in
            if worker.appState != .running && !textPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                worker.synthesize(
                    text: textPrompt,
                    speaker: selectedSpeaker,
                    isClone: isCloningMode,
                    cloneAudioPath: cloneAudioPath,
                    clonePromptText: clonePromptText,
                    tempStart: tempStart,
                    tempMin: tempMin,
                    tempDecay: tempDecay,
                    cfgScale: cfgScale,
                    bypassWatermark: bypassWatermark,
                    useQuant: useQuant,
                    maxLengthMs: maxAudioLengthSec * 1000.0
                )
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: Notification.Name("MisoTTS_DoReset"))) { _ in
            resetParameters()
        }
    }
    
    // MARK: - Speech Metrics & Heuristics
    
    private var wordCount: Int {
        textPrompt.split { $0.isWhitespace || $0.isNewline }.count
    }
    
    private var estimatedSpeakingDuration: Double {
        guard wordCount > 0 else { return 0.0 }
        // Human conversational speech heuristic: ~2.2 words per second + 2.0s padding for pauses
        return (Double(wordCount) / 2.2) + 2.0
    }
    
    private var isLengthWarningActive: Bool {
        estimatedSpeakingDuration > maxAudioLengthSec
    }
    
    // MARK: - Helper Logic
    
    private var statusColor: Color {
        switch worker.appState {
        case .idle: return .secondary
        case .running: return .blue
        case .success: return .green
        case .failed: return .red
        }
    }
    
    private func resetParameters() {
        textPrompt = "Hello from MisoTTS! This is a real-time, local Metal-accelerated synthesis session on my Apple Silicon GPU."
        selectedSpeaker = 0
        tempStart = 0.70
        tempMin = 0.40
        tempDecay = 30
        cfgScale = 2.00
        bypassWatermark = false
        useQuant = true
        maxAudioLengthSec = 15.0
        isCloningMode = false
        cloneAudioPath = ""
        clonePromptText = ""
    }
    
    /// Native macOS File Selection Dialog (macos-hig-interaction File Management)
    private func selectLocalWavFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.wav]
        panel.message = "Choose a clean 16kHz WAV file as a voice cloning template"
        
        if panel.runModal() == .OK {
            if let url = panel.url {
                cloneAudioPath = url.path
            }
        }
    }
}
