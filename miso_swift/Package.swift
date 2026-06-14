// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MisoSwiftDemo",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.10.0")
    ],
    targets: [
        .executableTarget(
            name: "MisoSwiftDemo",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Sources"
        )
    ]
)
