// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ClothMetal",
    platforms: [.macOS(.v13)],
    targets: [
        // Flat layout: all sources live in the package root.
        .executableTarget(
            name: "ClothMetal",
            path: ".",
            exclude: ["build.sh", "README.md"]
        )
    ]
)
