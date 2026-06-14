// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MisoTTSStudio",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MisoTTSStudio", targets: ["MisoTTSStudio"])
    ],
    targets: [
        .executableTarget(
            name: "MisoTTSStudio",
            path: "Sources"
        )
    ]
)
