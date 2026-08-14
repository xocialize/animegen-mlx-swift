import Foundation
import WanCore
import MLXToolKit

/// Init-time configuration for `AnimeGenPackage` (C9): which published variant and where the flat
/// checkpoint lives. Per-request prompt/size/frames ride the canonical `T2VRequest` / `T2IRequest`.
///
/// AnimeGen ships as a *merged* Lightning checkpoint (the lightx2v 250928 4-step LoRA fused into
/// the fine-tuned experts at conversion time, `[high 2.0, low 1.0]`), so — like Bernini-R's
/// `.lightning` config — the package always runs the fixed CFG-free few-step sampler. AnimeGen's
/// recipe differs from Bernini's preset: **euler · 8 steps · shift 3.0** (AIdeaLab's card),
/// vs Bernini's 4-step/shift-5.0. The merged weights only work few-step.
///
/// Checkpoint resolution at `load()`:
///   1. `modelDirectory` (a resolved flat dir: `{high,low}_noise_model/vae/t5_encoder.safetensors`
///      + `config.json`, exactly Bernini's 1095-key layout)
///   2. `ANIMEGEN_WEIGHTS_DIR` env override
///   3. HF download of `repo` into the local cache
public struct AnimeGenConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Published variant repo id (also the provenance source).
    public var repo: String
    public var revision: String?
    /// Backbone quant of the chosen variant (bf16 or int4) — selection metadata; the loader
    /// auto-detects the actual quantization from the checkpoint's config.json.
    public var quant: Quant
    /// True for the AnimeGen-I2V checkpoint (Wan2.2-I2V-A14B, in_dim=36): the `textToVideo`
    /// surface then REQUIRES `initImage` (channel-concat conditioning). A distinct checkpoint,
    /// so it's a config (which package loads), not a request mode.
    public var i2v: Bool
    /// Resolved local checkpoint folder. Environment-specific → excluded from `Codable`.
    public var modelDirectory: URL?
    /// Engine-chosen models root (future auto-materialization target). Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/AnimeGen-T2V-A14B-Lightning-int4",
        revision: String? = nil,
        quant: Quant = .int4,
        i2v: Bool = false,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.i2v = i2v
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The int4 consumer variant (~27 GB on disk; ~42 GB peak measured at 480p/17f — the anime
    /// go-to tier). CFG-free 8-step: light and fast.
    public static var int4: AnimeGenConfiguration {
        AnimeGenConfiguration(repo: "mlx-community/AnimeGen-T2V-A14B-Lightning-int4", quant: .int4)
    }

    /// The bf16 variant (~64 GB on disk). Higher fidelity, heavier.
    public static var bf16: AnimeGenConfiguration {
        AnimeGenConfiguration(repo: "mlx-community/AnimeGen-T2V-A14B-Lightning-bf16", quant: .bf16)
    }

    /// AnimeGen-**I2V** int4 (Wan2.2-I2V-A14B fine-tune). `textToVideo` requires `initImage`.
    public static var i2vInt4: AnimeGenConfiguration {
        AnimeGenConfiguration(
            repo: "mlx-community/AnimeGen-I2V-A14B-Lightning-int4", quant: .int4, i2v: true)
    }

    /// AnimeGen-**I2V** bf16.
    public static var i2vBf16: AnimeGenConfiguration {
        AnimeGenConfiguration(
            repo: "mlx-community/AnimeGen-I2V-A14B-Lightning-bf16", quant: .bf16, i2v: true)
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant, i2v
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the resolved checkpoint into the OS file cache
/// before `load()`'s GPU evals, mirroring Bernini. Nil when the HF-download path is used.
extension AnimeGenConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] { [modelDirectory].compactMap { $0 } }
}

extension Specialty {
    /// Anime / cel-shaded stylized generation. AnimeGen's whole reason for existing.
    public static let anime: Specialty = "anime"
}
