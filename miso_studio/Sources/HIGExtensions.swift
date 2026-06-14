import SwiftUI

// MARK: - macOS HIG Active State Styles
struct HIGActiveStateModifier: ViewModifier {
    @Environment(\.controlActiveState) var activeState
    
    func body(content: Content) -> some View {
        content
            .opacity(activeState == .inactive ? 0.75 : 1.0)
            .saturation(activeState == .inactive ? 0.8 : 1.0)
            .animation(.easeInOut(duration: 0.2), value: activeState)
    }
}

extension View {
    /// Reduces opacity and saturation slightly when the macOS application window is inactive,
    /// conforming to HIG window management guidelines to minimize background visual clutter.
    func applyHIGActiveState() -> some View {
        self.modifier(HIGActiveStateModifier())
    }
}

// MARK: - Premium Button Design
struct PulsatingGradientButtonStyle: ButtonStyle {
    var isRunning: Bool
    @State private var isHovered = false
    
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline.weight(.semibold))
            .foregroundColor(.white)
            .padding(.vertical, 12)
            .padding(.horizontal, 24)
            .background(
                ZgGlowBackground(isHovered: isHovered, isRunning: isRunning)
            )
            .cornerRadius(10)
            .scaleEffect(configuration.isPressed ? 0.98 : (isHovered ? 1.02 : 1.0))
            .shadow(
                color: Color.accentColor.opacity(isHovered && !isRunning ? 0.4 : 0.1),
                radius: isHovered ? 8 : 4,
                x: 0,
                y: isHovered ? 4 : 2
            )
            .onHover { hover in
                withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
                    isHovered = hover
                }
            }
    }
}

fileprivate struct ZgGlowBackground: View {
    var isHovered: Bool
    var isRunning: Bool
    
    var body: some View {
        Group {
            if isRunning {
                LinearGradient(
                    gradient: Gradient(colors: [Color.orange, Color.red]),
                    startPoint: .leading,
                    endPoint: .trailing
                )
            } else {
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color.blue,
                        Color.purple,
                        isHovered ? Color.pink : Color.blue
                    ]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }
        }
        .animation(.easeInOut(duration: 0.4), value: isHovered)
        .animation(.easeInOut(duration: 0.4), value: isRunning)
    }
}

// MARK: - Format Utilities
enum HIGFormatter {
    /// Formats speech duration into clean, user-friendly MM:SS readouts.
    static func formatTime(_ time: TimeInterval) -> String {
        guard !time.isNaN else { return "00:00" }
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
}
