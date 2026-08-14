import Foundation
import WanCore
import MLX
import MLXToolKit

import BerniniR
import MLXBerniniR  // reused frame→PNG/MP4 encoders (encodePNG / encodeMP4)

/// MLXEngine package: AnimeGen-T2V (aidealab) — an anime-specialized full fine-tune of
/// Wan2.2-T2V-A14B. It reuses the Bernini-R dual-expert renderer/pipeline verbatim (same
/// `BerniniPipeline`, `denoiseT2V`, WanVAE, umT5); AnimeGen contributes only the fine-tuned expert
/// weights (converted 1:1 into Bernini's 1095-key layout) and AIdeaLab's fixed CFG-free recipe
/// (euler · 8 steps · shift 3.0), with the lightx2v 250928 Lightning LoRA merged offline
/// (`[high 2.0, low 1.0]`). One resident renderer exposes `textToVideo` + `textToImage`
/// (t2i = single-frame t2v — useful for anime-still corpus harvesting).
///
/// Engine-owned lifecycle (C13): the engine constructs from an `AnimeGenConfiguration`, pages the
/// working set in with `load()`, drives `run(_:)`, reclaims with `unload()`. Lifecycle is isolated
/// to `InferenceActor`; the non-`Sendable` `BerniniPipeline` never crosses the boundary.
/// Cancellation (CAN gate, engine 0.27.0) is honored at run entry and every denoising step.
@InferenceActor
public final class AnimeGenPackage: ModelPackage {
    public typealias Configuration = AnimeGenConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // AnimeGen weights are Apache-2.0 (AIdeaLab / GENIAC); this port code is Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "aidealab/AnimeGen-T2V",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // Split footprint (1.14 contract): residentBytes = weights floor, peakActivationBytes
                // = transient scratch; governor budgets Σresident + max(peakActivation).
                //   int4: MEASURED 42.1 GB peak @ 480p/17f/8-step CFG-free (2026-07-13, ckpt-lightning-int4).
                //     ~27 GB resident weights + ~15 GB activation. Declared with headroom.
                //   bf16: ESTIMATED from the 30-step CFG convtest (81.4 GB peak, 3f/480p); CFG-free
                //     8-step (B=1, no uncond) is lighter. Re-measure in-app (Phase 4) before trusting.
                footprints: [
                    QuantFootprint(quant: .int4, residentBytes: 28_000_000_000,
                                   peakActivationBytes: 18_000_000_000),
                    QuantFootprint(quant: .bf16, residentBytes: 58_000_000_000,
                                   peakActivationBytes: 26_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [
                SpecialtyWeight(.anime, strength: 0.95),
                SpecialtyWeight(.general, strength: 0.3),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "animegen-t2v",
                    summary: "Anime-specialized text-to-video (Wan2.2-A14B fine-tune, MLX). "
                        + "Cel-shaded / key-visual style; 832x480 native, frames must be 4n+1. "
                        + "Fixed fast recipe: CFG-free euler, 8 steps, shift 3.0. The I2V variant "
                        + "(.i2vInt4/.i2vBf16) takes an `initImage` first frame (channel-concat, 4 steps).",
                    modes: []
                ),
                T2IContract.descriptor(
                    name: "animegen-t2i",
                    summary: "Anime stills via single-frame Wan2.2-A14B video diffusion "
                        + "(AnimeGen fine-tune, MLX). Cel-shaded key art, 832x480 native. "
                        + "Fixed fast recipe: CFG-free euler, 8 steps, shift 3.0.",
                    modes: []
                ),
            ]
        )
    }

    private let configuration: Configuration
    /// The resident pipeline (dual experts + VAE + tokenizer), paged in by `load()`.
    private var pipeline: BerniniPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Page the working set in. Idempotent when already resident. Resolution order:
    /// explicit `modelDirectory` → `ANIMEGEN_WEIGHTS_DIR` env → HF download to the local cache.
    public func load() async throws {
        guard pipeline == nil else { return }
        let directory: URL
        if let explicit = configuration.modelDirectory {
            directory = explicit
        } else if let env = ProcessInfo.processInfo.environment["ANIMEGEN_WEIGHTS_DIR"] {
            directory = URL(filePath: env)
        } else {
            directory = try await WeightLoader.snapshotDownload(repoID: configuration.repo)
        }
        pipeline = try await BerniniPipeline.fromPretrained(modelDir: directory)
    }

    public func unload() async {
        pipeline = nil
        // Bound the MLX buffer-pool ratchet on eviction (engine ≥0.21.0 posture).
        MLX.GPU.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint is the FIRST act of run() — before the notLoaded guard.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2IRequest", got: String(describing: type(of: request)))
            }
            return try runT2I(t2i, pipeline: pipeline)
        case .textToVideo:
            guard let t2v = request as? T2VRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2VRequest", got: String(describing: type(of: request)))
            }
            return try await runT2V(t2v, pipeline: pipeline)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // MARK: - Surfaces (fixed AnimeGen recipe: CFG-free euler, 8 steps, shift 3.0)

    /// AnimeGen's merged Lightning checkpoint only works few-step: always the fixed recipe.
    private static let steps = 8       // T2V
    private static let i2vSteps = 4    // I2V
    private static let shift = 3.0

    private func runT2I(_ request: T2IRequest, pipeline: BerniniPipeline) throws -> T2IResponse {
        try Task.checkCancellation()
        guard !configuration.i2v else {
            // The I2V checkpoint conditions on a video first-frame; text-only stills are the T2V variant.
            throw PackageError.unsupportedCapability(.textToImage)
        }
        let frames = try pipeline.t2i(
            prompt: request.prompt,
            negativePrompt: request.negativePrompt,
            width: request.width ?? 832,
            height: request.height ?? 480,
            steps: request.steps ?? Self.steps,
            shift: Self.shift,
            lightning: true,  // CFG-free path (guideScale unused)
            seed: request.seed
        ) { _, _, _ in try Task.checkCancellation() }  // C13: per-step cancellation
        try Task.checkCancellation()
        // The 16-ch Wan VAE decodes 4 output frames per latent frame; the still is frame 0.
        let (data, width, height) = try encodePNG(frame: frames[0, 0..., 0, 0..., 0...])
        return T2IResponse(image: Image(format: .png, data: data, width: width, height: height))
    }

    private func runT2V(_ request: T2VRequest, pipeline: BerniniPipeline) async throws -> T2VResponse {
        try Task.checkCancellation()
        let fps = request.fps ?? 16
        let width = request.width ?? 832
        let height = request.height ?? 480
        // Frames must be 4n+1 (VAE temporal stride); default ~5s clip.
        let numFrames = request.numFrames ?? 81

        let frames: MLXArray
        if let initImage = request.initImage {
            // AnimeGen-I2V: condition on a first frame (channel-concat, in_dim=36). Requires the
            // i2v checkpoint (a distinct package config).
            guard configuration.i2v else {
                throw PackageError.configurationMismatch(
                    expected: "no initImage (this is the text-only T2V variant; load .i2vInt4 / "
                        + ".i2vBf16 for image conditioning)", got: "initImage")
            }
            let pixels = try decodeReferencePixels(initImage, width: width, height: height)
            frames = try pipeline.i2v(
                prompt: request.prompt, imagePixels: pixels,
                width: width, height: height, numFrames: numFrames,
                steps: request.steps ?? Self.i2vSteps, shift: Self.shift, seed: request.seed
            ) { _, _, _ in try Task.checkCancellation() }  // C13: per-step cancellation
        } else {
            guard !configuration.i2v else {
                throw PackageError.configurationMismatch(
                    expected: "initImage (the I2V variant requires a conditioning image)",
                    got: "text-only t2v")
            }
            frames = try pipeline.t2v(
                prompt: request.prompt,
                negativePrompt: request.negativePrompt,
                width: width,
                height: height,
                numFrames: numFrames,
                steps: request.steps ?? Self.steps,
                shift: Self.shift,
                lightning: true,  // CFG-free path (guideScale unused)
                seed: request.seed
            ) { _, _, _ in try Task.checkCancellation() }  // C13: per-step cancellation
        }
        // Post-core checkpoint: streaming VAE decode bails per chunk (non-throwing).
        try Task.checkCancellation()
        let mp4 = try await encodeMP4(frames: frames, fps: fps)
        return T2VResponse(
            video: Video(format: .mp4, data: mp4,
                         durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
    }
}

extension AnimeGenPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(AnimeGenPackage.self)
    }
}
