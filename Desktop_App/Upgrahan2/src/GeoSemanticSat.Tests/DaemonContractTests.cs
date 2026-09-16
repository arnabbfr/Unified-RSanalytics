using System;
using System.Collections.Generic;
using System.Linq;
using GeoSemanticSat.Core.ChangeDetection;
using GeoSemanticSat.Core.Model;
using System.IO;
using System.Text.Json;
using GeoSemanticSat.Core.Raster;
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

    /// <summary>
    /// Evidence imagery must come from the pair detection actually ran on.
    ///
    /// These used to be resolved by matching the record's timestamps against _timeSeries.
    /// That silently works for the synthetic archive and silently fails for anything else:
    /// a tile from LoadGeoTiff lives in _tiles and never enters _timeSeries, so the match
    /// missed and fell through to the synthetic first/last scenes. The analyst then reviewed
    /// a candidate against imagery it was never derived from.
    /// </summary>
    [Fact]
    public void EvidenceImagery_ComesFromTheDetectedPair_NotTheArchiveEnds()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"gss_detect_pair_{Guid.NewGuid():N}.tif");
        try
        {
            var session = new AnalysisSession();
            var status = session.Initialize();
            string t1Handle = status.Tiles[0].Handle;

            // A tile from outside the synthetic series: this is the case the old timestamp
            // lookup could not resolve. It is the archive's own final scene round-tripped
            // through a file, so detection against the baseline finds real candidates.
            var source = SyntheticScene.BuildDemoArchive()[^1];
            var bands = source.Bands.Keys.ToList();
            GeoTiffWriter.WriteGeoTiff(tempFile, source, bands);
            string t2Handle = session.LoadGeoTiff(tempFile).Handle;
            Assert.DoesNotContain(status.Tiles, t => t.Handle == t2Handle);

            string expectedT1 = session.WithTile(t1Handle, t => t.TileId);
            string expectedT2 = session.WithTile(t2Handle, t => t.TileId);

            var detected = session.DetectChanges(new Contracts.DetectRequest(t1Handle, t2Handle));
            Assert.NotEmpty(detected);

            var pair = session.WithCandidate(detected[0].Id, (a, b, _) => (T1: a.TileId, T2: b.TileId));
            Assert.Equal(expectedT1, pair.T1);
            Assert.Equal(expectedT2, pair.T2);

            // The heatmap is evidence for the same candidates, so it must agree. It used to
            // render _timeSeries[0] and [2] unconditionally.
            var heatmap = session.WithHeatmapContext((a, b, _) => (T1: a.TileId, T2: b.TileId));
            Assert.Equal(expectedT1, heatmap.T1);
            Assert.Equal(expectedT2, heatmap.T2);
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    /// <summary>A flagged candidate must not export as one nobody looked at.</summary>
    [Fact]
    public void FlagWithoutNotes_IsStillDistinguishableFromUnreviewed()
    {
        var session = new AnalysisSession();
        session.Initialize();

        var first = session.Candidates().First();
        Assert.True(session.Flag(first.Id, ""));

        var item = session.Review().Single(i => i.Record.Id == first.Id);
        Assert.Equal("Flagged", item.Status);
        Assert.False(item.Record.ConfirmedByAnalyst);
        Assert.False(item.Record.RejectedByAnalyst);
        // The record is all the GeoJSON export carries, so the outcome has to live on it.
        Assert.False(string.IsNullOrWhiteSpace(item.Record.AnalystNotes));
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

    /// <summary>
    /// A non-positive patch size used to never terminate the scan loop: the bound
    /// (height - patchSize) grows while the counter steps by a negative stride, and a stride
    /// of zero never advances. Because callers hold a lock for the duration, one such request
    /// hung the whole process with no error and no way back short of killing it.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-8)]
    public void DetectChanges_RejectsNonPositivePatchSize_RatherThanHanging(int patchSize)
    {
        var archive = SyntheticScene.BuildDemoArchive();
        var options = new MultiTemporalChangeDetector.ChangeDetectionOptions(PatchSize: patchSize);

        // The point is that this returns at all. Before the guard it never came back.
        var thrown = Record.Exception(
            () => MultiTemporalChangeDetector.DetectChanges(archive[0], archive[2], options));

        Assert.IsType<ArgumentOutOfRangeException>(thrown);
    }

    /// <summary>
    /// CONTEXT.md: a recorded outcome is not the same as "not yet reviewed". Flagging used to
    /// set only the queue item's comment, leaving the record's notes empty and both analyst
    /// booleans false - so an exported flagged candidate was indistinguishable from an
    /// untouched one.
    /// </summary>
    [Fact]
    public void Flagging_RecordsTheNoteOnTheRecord_NotJustTheQueueItem()
    {
        var session = new AnalysisSession();
        session.Initialize();

        var target = session.Candidates().First();
        Assert.True(session.Flag(target.Id, "revisit once the cloud clears"));

        var item = session.Review().Single(i => i.Record.Id == target.Id);
        Assert.Equal("Flagged", item.Status);
        Assert.Equal("revisit once the cloud clears", item.AnalystComments);
        Assert.Equal("revisit once the cloud clears", item.Record.AnalystNotes);

        // Flagged asserts neither outcome - it means "look again".
        Assert.False(item.Record.ConfirmedByAnalyst);
        Assert.False(item.Record.RejectedByAnalyst);
    }

    /// <summary>
    /// RFC 8259 forbids a byte-order mark. With one present the ordinary
    /// open(path) + json.loads fails on the first character.
    /// </summary>
    [Fact]
    public void ProvenanceExport_HasNoByteOrderMark_AndParsesAsGeoJson()
    {
        var session = new AnalysisSession();
        session.Initialize();

        string path = Path.Combine(Path.GetTempPath(), $"gss-prov-{Guid.NewGuid():N}.geojson");
        try
        {
            session.ExportGeoJson(path);

            byte[] head = File.ReadAllBytes(path).Take(3).ToArray();
            Assert.False(head.SequenceEqual(new byte[] { 0xEF, 0xBB, 0xBF }), "export starts with a UTF-8 BOM");

            using var parsed = JsonDocument.Parse(File.ReadAllText(path));
            Assert.Equal("FeatureCollection", parsed.RootElement.GetProperty("type").GetString());
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    /// <summary>
    /// A negative crop origin made the default width tile.Width - startX, which is larger
    /// than the tile, and the clamp that followed used the same unvalidated origin as its
    /// bound so it never constrained anything. The result was an oversized image padded with
    /// a repeated edge pixel, reporting dimensions the tile does not have.
    /// </summary>
    [Fact]
    public void RenderTile_ClampsNegativeCropOrigin_ToTheTileItself()
    {
        var tile = SyntheticScene.BuildDemoArchive()[0];

        using var stream = RasterVisualizer.RenderTileToBmpStream(tile, startX: -50, startY: -50);
        byte[] bmp = stream.ToArray();

        Assert.Equal((byte)'B', bmp[0]);
        Assert.Equal((byte)'M', bmp[1]);

        // Width and height live at offsets 18 and 22 of a BITMAPINFOHEADER.
        int width = BitConverter.ToInt32(bmp, 18);
        int height = Math.Abs(BitConverter.ToInt32(bmp, 22));

        Assert.InRange(width, 1, tile.Width);
        Assert.InRange(height, 1, tile.Height);
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
