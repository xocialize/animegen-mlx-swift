// swift-tools-version: 6.2
// animegen-mlx-swift — MLXEngine package for aidealab/AnimeGen-T2V, an anime-specialized
// text-to-video model that is a full fine-tune of Wan2.2-T2V-A14B. It reuses the Bernini-R
// dual-expert pipeline (BerniniR + wan-core) verbatim; the only differences are the fine-tuned
// expert weights (converted 1:1 into Bernini's 1095-key layout) and AIdeaLab's fixed sampler
// recipe (CFG-free · euler · 8 steps · shift 3.0), with the lightx2v 250928 Lightning LoRA
// merged offline into the checkpoint. Python oracle + conversion: DEV_ARCHIVE/bernini-r-mlx
// (scripts/convert_animegen.py, scripts/merge_animegen_lightning.py).

import PackageDescription

let package = Package(
    name: "MLXAnimeGen",
    platforms: [
        // v26 to match the MLXEngine contract (MLXToolKit) the wrapper links.
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXAnimeGen", targets: ["MLXAnimeGen"]),
    ],
    dependencies: [
        // The Bernini-R renderer/pipeline (dual-expert Wan2.2-A14B) — reused verbatim; AnimeGen
        // is a weights-swap + recipe on top, not a new pipeline. Pulls WanCore transitively.
        .package(url: "https://github.com/xocialize/bernini-r-mlx-swift", from: "0.2.0"),
        .package(url: "https://github.com/xocialize/wan-core-mlx-swift", from: "0.3.0"),
        // MLXEngine contract; ≥0.27.0 for the CAN cancellation gate.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.27.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", "0.32.3"..<"0.33.0"),  // NAX split-K fix (AB-R-0778)
    ],
    targets: [
        .target(
            name: "MLXAnimeGen",
            dependencies: [
                .product(name: "BerniniR", package: "bernini-r-mlx-swift"),
                // Reused verbatim for the frame→PNG/MP4 encoders (public free functions);
                // AnimeGen shares Bernini's exact output-encoding path.
                .product(name: "MLXBerniniR", package: "bernini-r-mlx-swift"),
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MLXAnimeGen"
        ),
        .testTarget(
            name: "MLXAnimeGenTests",
            dependencies: [
                "MLXAnimeGen",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXAnimeGenTests"
        ),
    ]
)
