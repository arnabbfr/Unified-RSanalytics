using System;
using System.Collections.Generic;
using System.Linq;
using GeoSemanticSat.Core.ChangeDetection;
using GeoSemanticSat.Core.Clustering;
using GeoSemanticSat.Core.Model;
using GeoSemanticSat.Core.Raster;
using GeoSemanticSat.Core.Synthetic;
using GeoSemanticSat.Core.VectorIndex;
using GeoSemanticSat.Core.Workflow;
using GeoSemanticSat.Engine.Embeddings;
using GeoSemanticSat.Engine.Retrieval;
using VectorIndexStore = GeoSemanticSat.Core.VectorIndex.VectorIndex;

namespace GeoSemanticSat.Daemon;

/// <summary>
/// Owns everything that used to live in MainWindow's private fields: the vector index, the
/// search/change/review engines, and the loaded rasters.
///
/// SatelliteTile holds the whole multi-band raster, so it never leaves this process. Callers
/// get a string handle and pass it back. This is the reason the IPC boundary stays narrow:
/// the only large object in the system is the one thing that never has to cross it.
///
/// One session per daemon process. Every method takes the lock - the HTTP host is concurrent
/// and Core's engines are not uniformly thread-safe.
/// </summary>
public sealed class AnalysisSession
{
    private readonly object _lock = new();

    // Not readonly: neither VectorIndex nor ChangeSearchEngine exposes a Clear(), so a
    // re-initialise replaces them rather than adding mutating API to Core.
    private VectorIndexStore _index = new(SemanticEmbeddingLayout.Dimension);
    private ChangeSearchEngine _changeSearch = new();
    private readonly ReviewQueue _reviewQueue = new();
    private readonly Dictionary<string, SatelliteTile> _tiles = new(StringComparer.Ordinal);

    private SemanticSearchEngine? _searchEngine;
    private List<SatelliteTile> _timeSeries = new();
    private List<ChangeRecord> _candidates = new();

    /// <summary>Matches the UI's threshold for calling a candidate high-confidence.</summary>
    public const double HighConfidenceThreshold = 0.85;

    private SemanticSearchEngine Engine =>
        _searchEngine ??= new SemanticSearchEngine(_index);

    /// <summary>
    /// Loads the four-scene demo archive, indexes T1 and T3, and runs the default change
    /// pass so the workflow has something to show. Idempotent: calling it twice rebuilds.
    /// </summary>
    public Contracts.SessionStatusDto Initialize()
    {
        lock (_lock)
        {
            _tiles.Clear();
            _reviewQueue.Clear();
            _index = new VectorIndexStore(SemanticEmbeddingLayout.Dimension);
            _changeSearch = new ChangeSearchEngine();
            _searchEngine = null;

            _timeSeries = SyntheticScene.BuildDemoArchive();
            foreach (var tile in _timeSeries) _tiles[tile.TileId] = tile;

            var t1 = _timeSeries[0];
            var t3 = _timeSeries[2];

            Engine.IngestTile(t1, patchSize: 32);
            Engine.IngestTile(t3, patchSize: 32);

            _candidates = MultiTemporalChangeDetector.DetectChanges(t1, t3);
            ResolveOnsets(t1, t3);

            _changeSearch.AddRange(_candidates);
            foreach (var candidate in _candidates) _reviewQueue.Enqueue(candidate);

            return BuildStatus();
        }
    }

    /// <summary>Loads a GeoTIFF from disk and indexes it. Returns its handle.</summary>
    public Contracts.TileInfoDto LoadGeoTiff(string path)
    {
        lock (_lock)
        {
            var tile = GeoTiffReader.Read(path);
            _tiles[tile.TileId] = tile;
            Engine.IngestTile(tile, patchSize: 32);
            return Contracts.TileInfoDto.From(tile.TileId, tile);
        }
    }

    public Contracts.SessionStatusDto Status()
    {
        lock (_lock) return BuildStatus();
    }

    // ---- Search ----

    public List<Contracts.SearchResultDto> SearchByText(Contracts.TextSearchRequest req)
    {
        lock (_lock)
        {
            return Engine.SearchByText(req.Query, req.TopK, req.Filter?.ToCore())
                         .Select(Contracts.SearchResultDto.From)
                         .ToList();
        }
    }

    public List<Contracts.SearchResultDto> SearchSimilar(Contracts.SimilarSearchRequest req)
    {
        lock (_lock)
        {
            var patch = _index.GetAllPatches().FirstOrDefault(p => p.PatchId == req.PatchId)
                ?? throw new KeyNotFoundException($"Unknown patch '{req.PatchId}'.");

            return Engine.SearchByImagePatch(patch, req.TopK)
                         .Select(Contracts.SearchResultDto.From)
                         .ToList();
        }
    }

    /// <summary>Rocchio relevance feedback: Q' = aQ + (b/|Dr|)SUM(Dr) - (g/|Dnr|)SUM(Dnr).</summary>
    public List<Contracts.SearchResultDto> SearchWithFeedback(Contracts.FeedbackSearchRequest req)
    {
        lock (_lock)
        {
            var byId = _index.GetAllPatches().ToDictionary(p => p.PatchId, StringComparer.Ordinal);

            List<TilePatch> Patches(IReadOnlyList<string> ids) => ids
                .Where(byId.ContainsKey)
                .Select(id => byId[id])
                .ToList();

            return Engine.SearchWithFeedback(
                             req.Query,
                             Patches(req.RelevantPatchIds),
                             Patches(req.IrrelevantPatchIds),
                             req.TopK)
                         .Select(Contracts.SearchResultDto.From)
                         .ToList();
        }
    }

    public static string ExplainQuery(string query) => TextQueryEncoder.ExplainQuery(query);

    // ---- Change detection ----

    public List<Contracts.ChangeRecordDto> DetectChanges(Contracts.DetectRequest req)
    {
        lock (_lock)
        {
            var t1 = RequireTile(req.T1Handle);
            var t2 = RequireTile(req.T2Handle);

            _candidates = MultiTemporalChangeDetector.DetectChanges(t1, t2, req.ToOptions());
            ResolveOnsets(t1, t2);

            _changeSearch = new ChangeSearchEngine();
            _changeSearch.AddRange(_candidates);

            _reviewQueue.Clear();
            foreach (var candidate in _candidates) _reviewQueue.Enqueue(candidate);

            return _candidates.Select(Contracts.ChangeRecordDto.From).ToList();
        }
    }

    public List<Contracts.ChangeRecordDto> Candidates()
    {
        lock (_lock) return _candidates.Select(Contracts.ChangeRecordDto.From).ToList();
    }

    public List<Contracts.ChangeSearchResultDto> SearchChanges(Contracts.ChangeSearchRequest req)
    {
        lock (_lock)
        {
            return _changeSearch.Search(req.ToCore(), req.TopK)
                                .Select(Contracts.ChangeSearchResultDto.From)
                                .ToList();
        }
    }

    // ---- Clustering ----

    public List<Contracts.ClusterDto> Cluster(Contracts.ClusterRequest req)
    {
        lock (_lock)
        {
            return SpatialSemanticClusterer
                .ClusterSites(_index.GetAllPatches(), req.EpsilonCosine, req.MinPoints, req.MaxDistanceKm)
                .Select(Contracts.ClusterDto.From)
                .ToList();
        }
    }

    // ---- Review ----

    public List<Contracts.ReviewItemDto> Review()
    {
        lock (_lock) return _reviewQueue.GetAll().Select(Contracts.ReviewItemDto.From).ToList();
    }

    public bool Confirm(string id, string notes)
    {
        lock (_lock) { Annotate(id, notes, confirmed: true); return _reviewQueue.Confirm(id, notes); }
    }

    public bool Reject(string id, string notes)
    {
        lock (_lock) { Annotate(id, notes, confirmed: false); return _reviewQueue.Reject(id, notes); }
    }

    public bool Flag(string id, string notes)
    {
        lock (_lock) return _reviewQueue.FlagForReview(id, notes);
    }

    public void ExportGeoJson(string path)
    {
        lock (_lock) ProvenanceAuditTrail.SaveGeoJson(path, _candidates);
    }

    // ---- Rasters (images are produced by the caller under the lock) ----

    public T WithTile<T>(string handle, Func<SatelliteTile, T> fn)
    {
        lock (_lock) return fn(RequireTile(handle));
    }

    public T WithCandidate<T>(string changeId, Func<SatelliteTile, SatelliteTile, ChangeRecord, T> fn)
    {
        lock (_lock)
        {
            var record = _candidates.FirstOrDefault(c => c.Id == changeId)
                ?? throw new KeyNotFoundException($"Unknown candidate '{changeId}'.");

            // The pair the candidate was actually derived from, by timestamp.
            var t1 = _timeSeries.FirstOrDefault(t => t.AcquisitionTimestamp == record.TimestampT1)
                     ?? _timeSeries.First();
            var t2 = _timeSeries.FirstOrDefault(t => t.AcquisitionTimestamp == record.TimestampT2)
                     ?? _timeSeries.Last();

            return fn(t1, t2, record);
        }
    }

    public T WithHeatmapContext<T>(Func<SatelliteTile, SatelliteTile, IReadOnlyList<ChangeRecord>, T> fn)
    {
        lock (_lock)
        {
            var t1 = _timeSeries.Count > 0 ? _timeSeries[0] : throw new InvalidOperationException("No archive loaded.");
            var t2 = _timeSeries.Count > 2 ? _timeSeries[2] : _timeSeries[^1];
            return fn(t1, t2, _candidates);
        }
    }

    // ---- internals ----

    private SatelliteTile RequireTile(string handle) =>
        _tiles.TryGetValue(handle, out var tile)
            ? tile
            : throw new KeyNotFoundException($"Unknown tile handle '{handle}'.");

    /// <summary>
    /// CUSUM onset over the chronological series. Without this every candidate reports its
    /// T1 as the onset, which is the acquisition date, not the date the change began.
    /// </summary>
    private void ResolveOnsets(SatelliteTile t1, SatelliteTile t2)
    {
        foreach (var candidate in _candidates)
        {
            candidate.EarliestObservationTimestamp = _timeSeries.Count >= 2
                ? OnsetEstimator.EstimateEarliestObservation(_timeSeries, candidate.Bounds, candidate.Type)
                : t1.AcquisitionTimestamp;
        }
    }

    private void Annotate(string id, string notes, bool confirmed)
    {
        var record = _candidates.FirstOrDefault(c => c.Id == id);
        if (record is null) return;
        record.ConfirmedByAnalyst = confirmed;
        record.RejectedByAnalyst = !confirmed;
        record.AnalystNotes = notes;
    }

    private Contracts.SessionStatusDto BuildStatus() => new(
        _index.Count,
        _candidates.Count,
        _candidates.Count(c => c.Confidence >= HighConfidenceThreshold),
        _reviewQueue.Count,
        _tiles.Select(kv => Contracts.TileInfoDto.From(kv.Key, kv.Value)).ToList());
}
