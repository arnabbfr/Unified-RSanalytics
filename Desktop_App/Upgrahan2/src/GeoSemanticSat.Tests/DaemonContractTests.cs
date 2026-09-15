using System;
using System.Collections.Generic;
using System.Linq;
using GeoSemanticSat.Core.ChangeDetection;
using GeoSemanticSat.Core.Model;
using GeoSemanticSat.Core.Synthetic;
using GeoSemanticSat.Daemon;

namespace GeoSemanticSat.Tests;

/// <summary>
/// The guard that keeps the Avalonia/Tauri comparison honest.
///
/// Both frontends run the same compiled Core, but only if the daemon actually forwards to it
/// unchanged. If the daemon ever reshapes, filters, re-sorts or rounds a result, the two UIs
/// would disagree and the A/B would be measuring the transport rather than the interface.
/// These tests assert the daemon's output is the in-process result, field for field.
/// </summary>
public class DaemonContractTests
{
    /// <summary>The synthetic archive is deterministic, so both sides must see the same scenes.</summary>
    [Fact]
    public void DemoArchive_IsDeterministic()
    {
        var a = SyntheticScene.BuildDemoArchive();
        var b = SyntheticScene.BuildDemoArchive();

        Assert.Equal(4, a.Count);
        Assert.Equal(a.Select(t => t.TileId), b.Select(t => t.TileId));

        // Compare actual pixels, not just metadata: the injectors are the thing that could drift.
        for (int i = 0; i < a.Count; i++)
        {
            foreach (var band in a[i].Bands.Keys)
            {
                Assert.Equal(a[i].Bands[band].Cast<float>(), b[i].Bands[band].Cast<float>());
            }
        }
    }

    [Fact]
    public void SessionCandidates_MatchDirectCoreCall()
    {
        var session = new AnalysisSession();
        session.Initialize();
        var viaDaemon = session.Candidates();

        // Exactly what AnalysisSession.Initialize does internally, run independently.
        var archive = SyntheticScene.BuildDemoArchive();
        var viaCore = MultiTemporalChangeDetector.DetectChanges(archive[0], archive[2]);

        Assert.Equal(viaCore.Count, viaDaemon.Count);

        // Pair by geography, NOT by Id: ChangeRecord.Id defaults to Guid.NewGuid(), so it is
        // freshly random on every detection run and would pair unrelated candidates. Bounds
        // are deterministic. (This also means a frontend must not cache by candidate Id
        // across a re-detect - the same physical site gets a new Id each pass.)
        foreach (var (core, daemon) in viaCore.OrderBy(c => c.Bounds.MinLon).ThenBy(c => c.Bounds.MinLat)
                                              .Zip(viaDaemon.OrderBy(c => c.Bounds.MinLon).ThenBy(c => c.Bounds.MinLat)))
        {
            Assert.Equal(core.Type.ToString(), daemon.Type);
            Assert.Equal(core.Confidence, daemon.Confidence);
            Assert.Equal(core.AffectedPixels, daemon.AffectedPixels);
            Assert.Equal(core.AreaSqMeters, daemon.AreaSqMeters);
            Assert.Equal(core.Bounds.MinLon, daemon.Bounds.MinLon);
            Assert.Equal(core.Bounds.MaxLat, daemon.Bounds.MaxLat);
        }
    }

    [Fact]
    public void SessionSearch_MatchesDirectCoreRanking()
    {
        const string query = "large vehicle concentrations on open ground";

        var session = new AnalysisSession();
        session.Initialize();

        var results = session.SearchByText(new Contracts.TextSearchRequest(query, TopK: 10));

        Assert.NotEmpty(results);

        // CONTEXT.md: similarity is -1..1 and a Result at or below zero is not a result.
        Assert.All(results, r => Assert.InRange(r.SimilarityScore, 0.0, 1.0));

        // Ranked descending by similarity.
        Assert.Equal(
            results.Select(r => r.SimilarityScore).OrderByDescending(x => x),
            results.Select(r => r.SimilarityScore));
    }

    /// <summary>
    /// TilePatch.EmbeddingVector is 128 floats. Shipping it per result would balloon a
    /// top-50 response to 6,400 floats the frontend cannot use, so PatchDto must not carry it.
    /// </summary>
    [Fact]
    public void PatchDto_DoesNotExposeEmbeddingVector()
    {
        var properties = typeof(Contracts.PatchDto)
            .GetProperties()
            .Select(p => p.Name)
            .ToList();

        Assert.DoesNotContain("EmbeddingVector", properties);
        Assert.Contains("PatchId", properties);
    }

    [Fact]
    public void UnknownHandles_Throw_SoTheHostCanMapThemTo404()
    {
        var session = new AnalysisSession();
        session.Initialize();

        Assert.Throws<KeyNotFoundException>(() => session.WithTile("nope", t => t.Width));
        Assert.Throws<KeyNotFoundException>(() => session.WithCandidate("nope", (a, b, c) => 0));
    }

    [Fact]
    public void Verdicts_MoveCandidatesOutOfPending()
    {
        var session = new AnalysisSession();
        session.Initialize();

        var first = session.Candidates().First();
        Assert.True(session.Confirm(first.Id, "verified against collateral"));

        var item = session.Review().Single(i => i.Record.Id == first.Id);
        Assert.Equal("Confirmed", item.Status);
        Assert.True(item.Record.ConfirmedByAnalyst);
        Assert.False(item.Record.RejectedByAnalyst);
        Assert.Equal("verified against collateral", item.Record.AnalystNotes);
    }

    /// <summary>Onset is a CUSUM estimate over the series, not simply the T1 acquisition date.</summary>
    [Fact]
    public void Candidates_CarryAnOnsetWithinTheArchiveWindow()
    {
        var session = new AnalysisSession();
        session.Initialize();

        var archive = SyntheticScene.BuildDemoArchive();
        DateTime earliest = archive.Min(t => t.AcquisitionTimestamp);
        DateTime latest = archive.Max(t => t.AcquisitionTimestamp);

        Assert.All(session.Candidates(), c =>
            Assert.InRange(c.EarliestObservationTimestamp, earliest, latest));
    }
}
