using System;
using System.Collections.Generic;
using System.Linq;
using GeoSemanticSat.Core.ChangeDetection;
using GeoSemanticSat.Core.Clustering;
using GeoSemanticSat.Core.Model;
using GeoSemanticSat.Core.VectorIndex;
using GeoSemanticSat.Core.Workflow;

namespace GeoSemanticSat.Daemon;

/// <summary>
/// Wire shapes for the daemon.
///
/// These are deliberately NOT the Core model types. Two reasons:
/// - TilePatch.EmbeddingVector is 128 floats per patch. A top-50 search would ship 6,400
///   floats the frontend cannot use. It is dropped here.
/// - SatelliteTile holds Dictionary&lt;SpectralBand, float[,]&gt; - the entire multi-band raster.
///   It never crosses the wire at all; the daemon keeps it and hands out a string handle.
///
/// Vocabulary follows CONTEXT.md: a search returns Results, change detection produces
/// Candidates, and only a human decision makes something Verified or Rejected.
/// </summary>
public static class Contracts
{
    public record BboxDto(double MinLon, double MinLat, double MaxLon, double MaxLat)
    {
        public static BboxDto From(BoundingBox b) => new(b.MinLon, b.MinLat, b.MaxLon, b.MaxLat);
        public BoundingBox ToCore() => new(MinLon, MinLat, MaxLon, MaxLat);
    }

    public record CoordDto(double Latitude, double Longitude)
    {
        public static CoordDto From(GeoCoordinate c) => new(c.Latitude, c.Longitude);
        public GeoCoordinate ToCore() => new(Latitude, Longitude);
    }

    public record TileInfoDto(
        string Handle,
        string TileId,
        string Platform,
        DateTime AcquisitionTimestamp,
        BboxDto Bounds,
        int Width,
        int Height,
        double GroundSamplingDistanceMeters,
        double CloudCoverPercentage,
        bool RequiresVisibleBandProxies,
        IReadOnlyList<string> Bands)
    {
        public static TileInfoDto From(string handle, SatelliteTile t) => new(
            handle,
            t.TileId,
            t.Platform.ToString(),
            t.AcquisitionTimestamp,
            BboxDto.From(t.Bounds),
            t.Width,
            t.Height,
            t.GroundSamplingDistanceMeters,
            t.CloudCoverPercentage,
            t.RequiresVisibleBandProxies,
            t.Bands.Keys.Select(b => b.ToString()).ToList());
    }

    /// <summary>A Patch, minus its embedding. See CONTEXT.md - not analyst-facing vocabulary.</summary>
    public record PatchDto(
        string PatchId,
        string ParentTileId,
        string Platform,
        DateTime Timestamp,
        BboxDto Bounds,
        int PixelX,
        int PixelY,
        int PatchWidth,
        int PatchHeight,
        double QualityScore,
        bool HasCloudOrShadow)
    {
        public static PatchDto From(TilePatch p) => new(
            p.PatchId, p.ParentTileId, p.Platform.ToString(), p.Timestamp,
            BboxDto.From(p.Bounds), p.PixelX, p.PixelY, p.PatchWidth, p.PatchHeight,
            p.QualityScore, p.HasCloudOrShadow);
    }

    /// <summary>
    /// A search Result. SimilarityScore is -1..1 and is NOT a probability or a confidence.
    /// CONTEXT.md forbids displaying results at or below zero.
    /// </summary>
    public record SearchResultDto(PatchDto Patch, double SimilarityScore, double Distance)
    {
        public static SearchResultDto From(SearchResult r) =>
            new(PatchDto.From(r.Patch), r.SimilarityScore, r.Distance);
    }

    /// <summary>
    /// A change Candidate. Confidence is an evidence score in 0..1 that ranks candidates
    /// against each other - explicitly not a calibrated probability (CONTEXT.md).
    /// </summary>
    public record ChangeRecordDto(
        string Id,
        string TileId,
        BboxDto Bounds,
        CoordDto Center,
        DateTime TimestampT1,
        DateTime TimestampT2,
        DateTime EarliestObservationTimestamp,
        string Type,
        double Confidence,
        int AffectedPixels,
        double AreaSqMeters,
        IReadOnlyDictionary<string, double> Metrics,
        string ProcessingNotes,
        bool ConfirmedByAnalyst,
        bool RejectedByAnalyst,
        string AnalystNotes)
    {
        public static ChangeRecordDto From(ChangeRecord c) => new(
            c.Id, c.TileId, BboxDto.From(c.Bounds), CoordDto.From(c.Center),
            c.TimestampT1, c.TimestampT2, c.EarliestObservationTimestamp,
            c.Type.ToString(), c.Confidence, c.AffectedPixels, c.AreaSqMeters,
            c.Metrics, c.ProcessingNotes, c.ConfirmedByAnalyst, c.RejectedByAnalyst, c.AnalystNotes);
    }

    public record ChangeSearchResultDto(ChangeRecordDto Record, double DistanceKm, double RelevanceScore)
    {
        public static ChangeSearchResultDto From(ChangeSearchResult r) =>
            new(ChangeRecordDto.From(r.Record), r.DistanceKm, r.RelevanceScore);
    }

    /// <summary>
    /// ClusterGroup.Centroid is the mean embedding vector, not a location, so it is not sent.
    /// The map needs a geographic centre, which is the centre of the enclosing bounds.
    /// </summary>
    public record ClusterDto(
        int ClusterId,
        string Label,
        CoordDto Centre,
        BboxDto EnclosingBounds,
        int MemberCount,
        double CohesionScore,
        IReadOnlyList<PatchDto> Members)
    {
        public static ClusterDto From(ClusterGroup g) => new(
            g.ClusterId,
            g.Label,
            CoordDto.From(g.EnclosingBounds.Center),
            BboxDto.From(g.EnclosingBounds),
            g.Members.Count,
            g.CohesionScore,
            g.Members.Select(PatchDto.From).ToList());
    }

    public record ReviewItemDto(
        ChangeRecordDto Record,
        DateTime AddedTimestamp,
        string Status,
        string AnalystComments,
        DateTime? DecisionTimestamp)
    {
        public static ReviewItemDto From(ReviewItem i) => new(
            ChangeRecordDto.From(i.Record), i.AddedTimestamp, i.Status,
            i.AnalystComments, i.DecisionTimestamp);
    }

    // ---- Request bodies ----

    public record SearchFilterDto(
        BboxDto? Bounds = null,
        CoordDto? Center = null,
        double? RadiusKm = null,
        DateTime? StartDate = null,
        DateTime? EndDate = null,
        string? Platform = null,
        double MinQuality = 0.3)
    {
        public SearchFilter ToCore() => new(
            Bounds?.ToCore(),
            Center?.ToCore(),
            RadiusKm,
            StartDate,
            EndDate,
            Platform is null ? null : Enum.Parse<SensorPlatform>(Platform, ignoreCase: true),
            MinQuality);
    }

    public record TextSearchRequest(string Query, int TopK = 10, SearchFilterDto? Filter = null);

    public record SimilarSearchRequest(string PatchId, int TopK = 10);

    public record FeedbackSearchRequest(
        string Query,
        IReadOnlyList<string> RelevantPatchIds,
        IReadOnlyList<string> IrrelevantPatchIds,
        int TopK = 10);

    public record DetectRequest(
        string T1Handle,
        string T2Handle,
        int PatchSize = 16,
        double MinConfidence = 0.65,
        bool EnableRadiometricNormalization = true,
        bool EnableJitterSuppression = true,
        bool EnableQualityMasking = true,
        double MinChangeMagnitude = 0.05)
    {
        public MultiTemporalChangeDetector.ChangeDetectionOptions ToOptions() => new(
            PatchSize, MinConfidence, EnableRadiometricNormalization,
            EnableJitterSuppression, EnableQualityMasking, MinChangeMagnitude);
    }

    public record ChangeSearchRequest(
        CoordDto? Center = null,
        double? RadiusKm = null,
        BboxDto? Bounds = null,
        DateTime? StartDate = null,
        DateTime? EndDate = null,
        string? TargetChangeType = null,
        double MinConfidence = 0.50,
        string? Keyword = null,
        int TopK = 50)
    {
        public ChangeSearchCriteria ToCore() => new(
            Center?.ToCore(),
            RadiusKm,
            Bounds?.ToCore(),
            StartDate,
            EndDate,
            TargetChangeType is null ? null : Enum.Parse<ChangeType>(TargetChangeType, ignoreCase: true),
            MinConfidence,
            Keyword);
    }

    public record OnsetRequest(string ChangeId);

    public record ClusterRequest(double EpsilonCosine = 0.35, int MinPoints = 3, double MaxDistanceKm = 2.0);

    public record VerdictRequest(string Notes = "");

    public record ExportRequest(string Path);

    public record GenerateArchiveRequest(double Latitude, double Longitude);

    public record BenchmarkRequest(string OutputDirectory);

    public record SessionStatusDto(
        int IndexedPatches,
        int Candidates,
        int HighConfidenceCandidates,
        int ReviewQueueSize,
        IReadOnlyList<TileInfoDto> Tiles);
}
