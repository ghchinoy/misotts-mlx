import Foundation
import AVFoundation

/// A robust audio player wrapper around AVAudioPlayer to manage local synthesised audio playback, 
/// supporting play/pause states, seeking, playback speed control, and live duration monitoring.
class MisoAudioPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var volume: Float = 0.8 {
        didSet {
            audioPlayer?.volume = volume
        }
    }
    @Published var speed: Float = 1.0 {
        didSet {
            audioPlayer?.rate = speed
        }
    }
    
    private var audioPlayer: AVAudioPlayer? = nil
    private var timer: Timer? = nil
    
    /// Loads a local WAV audio asset and prepares it for playback.
    func load(url: URL) {
        stop()
        
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            player.enableRate = true
            player.volume = volume
            player.rate = speed
            player.prepareToPlay()
            
            self.audioPlayer = player
            self.duration = player.duration
            self.currentTime = 0
        } catch {
            print("❌ Failed to initialize AVAudioPlayer: \(error.localizedDescription)")
        }
    }
    
    /// Starts or resumes playback.
    func play() {
        guard let player = audioPlayer else { return }
        player.play()
        isPlaying = true
        startProgressTimer()
    }
    
    /// Pauses current playback.
    func pause() {
        audioPlayer?.pause()
        isPlaying = false
        stopProgressTimer()
    }
    
    /// Stops playback and resets playhead.
    func stop() {
        audioPlayer?.stop()
        audioPlayer?.currentTime = 0
        currentTime = 0
        isPlaying = false
        stopProgressTimer()
    }
    
    /// Seeks the playhead to a specific time interval.
    func seek(to time: TimeInterval) {
        guard let player = audioPlayer else { return }
        player.currentTime = time
        currentTime = time
    }
    
    private func startProgressTimer() {
        stopProgressTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let player = self.audioPlayer else { return }
            if self.isPlaying {
                self.currentTime = player.currentTime
            }
        }
    }
    
    private func stopProgressTimer() {
        timer?.invalidate()
        timer = nil
    }
    
    // MARK: - AVAudioPlayerDelegate
    
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        currentTime = 0
        stopProgressTimer()
    }
}
