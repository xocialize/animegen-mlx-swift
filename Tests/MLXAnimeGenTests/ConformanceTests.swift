// ConformanceTests.swift — AnimeGen through the engine's offline gates (no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before the notLoaded
// guard or weights). CAN-3 documents the checkpoint cadence: t2v/t2i thread a throwing `onStep`
// (one `try Task.checkCancellation()` per denoising step, AnimeGenPackage.swift) and the wan-core
// streaming VAE decode bails per temporal chunk, with the wrapper's post-core checkpoint
// rethrowing unchanged. Offline checks: two-layer license gate + manifest surfaces + footprints.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXAnimeGen

final class AnimeGenConformanceTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        let package = AnimeGenPackage(configuration: AnimeGenConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2VRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (document of record)

    func testCANCadenceDeclaration() {
        // textToVideo is a long-run capability — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: AnimeGenPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: AnimeGenPackage.manifest,
            posture: .cadence([
                .init(phase: .denoise, unit: .step),   // throwing onStep, every pipeline call
                .init(phase: .decode, unit: .chunk),    // wan-core decodeStreaming per chunk
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Offline manifest / license conformance (C7/C8, surfaces, footprints)

    func testManifestLicenseAndSurfaces() {
        let m = AnimeGenPackage.manifest
        // C7/C8: both layers Apache-2.0 → admissible with no per-run acceptance.
        XCTAssertEqual(m.license.weightLicense, .apache2)
        XCTAssertEqual(m.license.portCodeLicense, .apache2)
        // Provenance points at the upstream fine-tune, tier 1.
        XCTAssertEqual(m.provenance.sourceRepo, "aidealab/AnimeGen-T2V")
        // Both surfaces present.
        let names = Set(m.surfaces.map(\.name))
        XCTAssertTrue(names.contains("animegen-t2v"))
        XCTAssertTrue(names.contains("animegen-t2i"))
        // Split footprints declared for both quants (1.14 contract).
        let quants = Set(m.requirements.footprints.map(\.quant))
        XCTAssertTrue(quants.contains(.int4))
        XCTAssertTrue(quants.contains(.bf16))
        for f in m.requirements.footprints {
            XCTAssertGreaterThan(f.residentBytes, 0)
            XCTAssertGreaterThan(f.peakActivationBytes, 0)
        }
    }

    func testConfigurationCodableRoundTrip() throws {
        let cfg = AnimeGenConfiguration.int4
        let data = try JSONEncoder().encode(cfg)
        let back = try JSONDecoder().decode(AnimeGenConfiguration.self, from: data)
        XCTAssertEqual(back.repo, cfg.repo)
        XCTAssertEqual(back.quant, cfg.quant)
    }
}
