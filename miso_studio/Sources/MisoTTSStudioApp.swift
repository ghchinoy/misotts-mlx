import SwiftUI

@main
struct MisoTTSStudioApp: App {
    
    init() {
        #if os(macOS)
        NSApplication.shared.setActivationPolicy(.regular)
        #endif
    }
    
    // Core notification definitions for menu bar triggers (macos-hig-interaction Menu Bar alignment)
    static let triggerSynthesisNotification = Notification.Name("MisoTTS_TriggerSynthesis")
    static let triggerResetNotification = Notification.Name("MisoTTS_TriggerReset")
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .navigationTitle("MisoTTS Studio")
                // Subscribe to native macOS menu bar shortcuts
                .onReceive(NotificationCenter.default.publisher(for: Self.triggerSynthesisNotification)) { _ in
                    // Synthesize is bound to Cmd + Return
                    // It will trigger synthesis if the user presses Cmd + Return or clicks the menu item
                    NotificationCenter.default.post(name: Notification.Name("MisoTTS_DoSynthesis"), object: nil)
                }
                .onReceive(NotificationCenter.default.publisher(for: Self.triggerResetNotification)) { _ in
                    // Reset is bound to Cmd + Option + R
                    NotificationCenter.default.post(name: Notification.Name("MisoTTS_DoReset"), object: nil)
                }
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            // Include standard system command replacements/menus
            SidebarCommands()
            
            // File Commands Menu group
            CommandGroup(replacing: .newItem) {
                Button("Synthesize Speech...") {
                    NotificationCenter.default.post(name: Self.triggerSynthesisNotification, object: nil)
                }
                .keyboardShortcut(.return, modifiers: .command)
                
                Button("Reset Configuration Defaults") {
                    NotificationCenter.default.post(name: Self.triggerResetNotification, object: nil)
                }
                .keyboardShortcut("r", modifiers: [.command, .option])
            }
            
            // Standard App Help Commands
            CommandGroup(replacing: .help) {
                Link("MisoTTS Architecture Guide", destination: URL(string: "https://github.com/hardikpandya/stop-slop")!) // placeholders are fine or just standard link
                    .keyboardShortcut("?", modifiers: .command)
            }
        }
    }
}
