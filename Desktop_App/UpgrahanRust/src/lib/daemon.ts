/**
 * Typed client for GeoSemanticSat.Daemon.
 *
 * Every shape here mirrors Desktop_App/Upgrahan2/src/GeoSemanticSat.Daemon/Contracts.cs.
 * Terminology follows CONTEXT.md and must not drift:
 *   - Result    : a Patch returned by a search. Says "this looks like what you described".
 *                 Makes NO claim that anything changed. Similarity is -1..1, never a
 *                 probability, and a Result at or below zero is not shown.
 *   - Candidate : a place the system believes physically changed, which no human has
 *                 checked yet. Always carries a change type.
 *   - Verified  : a Candidate a human confirmed. The only state presentable as fact.
 *   - Rejected  : a Candidate a human dismissed. A real, recorded outcome.
 */

import { invoke } from "@tauri-apps/api/core";

export type ChangeType =
  | "NoChange"
  | "Construction"
  | "Clearance"
  | "WaterExtentVariation"
  | "RoadDevelopment"
  | "ActivityConcentration";

export type SensorPlatform =
  | "Sentinel2_Optical"
  | "Sentinel1_SAR"
  | "Landsat8_9"
  | "ISRO_Bhuvan";

export type VisualRenderMode =
  | "TrueColorRGB"
  | "FalseColorInfrared"
  | "SWIR_GeologicalMoisture"
  | "NDVI_Heatmap"
  | "NDWI_WaterMap"
  | "NDBI_BuiltUpUrban"
  | "SAR_MicrowaveSimulation"
  | "ThermalRadiance"
  | "ChangeOverlay";

export type ReviewStatus = "Pending" | "Confirmed" | "Rejected" | "Flagged";

export interface Bbox {
  minLon: number;
  minLat: number;
  maxLon: number;
  maxLat: number;
}

export interface Coord {
  latitude: number;
  longitude: number;
}

export interface TileInfo {
  handle: string;
  tileId: string;
  platform: SensorPlatform;
  acquisitionTimestamp: string;
  bounds: Bbox;
  width: number;
  height: number;
  groundSamplingDistanceMeters: number;
  cloudCoverPercentage: number;
  /** True when the scene lacks NIR/SWIR, so indices fall back to visible-band proxies. */
  requiresVisibleBandProxies: boolean;
  bands: string[];
}

export interface Patch {
  patchId: string;
  parentTileId: string;
  platform: SensorPlatform;
  timestamp: string;
  bounds: Bbox;
  pixelX: number;
  pixelY: number;
  patchWidth: number;
  patchHeight: number;
  qualityScore: number;
  hasCloudOrShadow: boolean;
}

/** A search Result. `similarityScore` is -1..1 and is not a confidence. */
export interface SearchResult {
  patch: Patch;
  similarityScore: number;
  distance: number;
}

/**
 * A change Candidate. `confidence` is an evidence score in 0..1 that ranks candidates
 * against each other - explicitly NOT a calibrated probability.
 *
 * Note: `id` is a fresh GUID on every detection run, so do not persist selection by id
 * across a re-detect.
 */
export interface ChangeRecord {
  id: string;
  tileId: string;
  bounds: Bbox;
  center: Coord;
  timestampT1: string;
  timestampT2: string;
  earliestObservationTimestamp: string;
  type: ChangeType;
  confidence: number;
  affectedPixels: number;
  areaSqMeters: number;
  metrics: Record<string, number>;
  processingNotes: string;
  confirmedByAnalyst: boolean;
  rejectedByAnalyst: boolean;
  analystNotes: string;
}

export interface ChangeSearchResult {
  record: ChangeRecord;
  distanceKm: number;
  relevanceScore: number;
}

export interface Cluster {
  clusterId: number;
  label: string;
  centre: Coord;
  enclosingBounds: Bbox;
  memberCount: number;
  cohesionScore: number;
  members: Patch[];
}

export interface ReviewItem {
  record: ChangeRecord;
  addedTimestamp: string;
  status: ReviewStatus;
  analystComments: string;
  decisionTimestamp: string | null;
}

export interface SessionStatus {
  indexedPatches: number;
  candidates: number;
  highConfidenceCandidates: number;
  reviewQueueSize: number;
  tiles: TileInfo[];
}

export interface FocusedInspection {
  /** base64 BMP */
  t1: string;
  t2: string;
  overlay: string;
  cropX: number;
  cropY: number;
  cropWidth: number;
  cropHeight: number;
  peakMagnitude: number;
  meanMagnitude: number;
}

export interface IndexHeatmaps {
  ndvi: string;
  ndbi: string;
  ndwi: string;
  bsi: string;
  cropX: number;
  cropY: number;
  cropWidth: number;
  cropHeight: number;
}

interface Endpoint {
  baseUrl: string;
  token: string;
}

let endpoint: Endpoint | null = null;

const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));

/**
 * Called after the client has had to start a replacement daemon and re-seed its session.
 * The app sets this so it can reload what it is showing and tell the analyst, since any
 * verdicts recorded against the previous process are gone.
 */
let onDaemonRestarted: (() => void) | undefined;

export function setDaemonRestartHandler(handler: () => void | Promise<void>) {
  onDaemonRestarted = handler;
}

/**
 * In-flight connection, so concurrent callers share one start_daemon invocation.
 *
 * Caching only the resolved endpoint is not enough: boot() leaves the titlebar live, so
 * Benchmark or Load GeoTIFF can call in while the first connect is still awaiting. Each
 * would then invoke start_daemon, and the supervisor has no whole-start lock - which can
 * orphan one daemon and leave the client talking to the other.
 */
let connecting: Promise<Endpoint> | null = null;

/** Starts the daemon (idempotent) and caches where to reach it. */
export function connect(): Promise<Endpoint> {
  if (endpoint) return Promise.resolve(endpoint);
  connecting ??= openConnection().finally(() => {
    connecting = null;
  });
  return connecting;
}

async function openConnection(): Promise<Endpoint> {
  const resolved = await invoke<Endpoint>("start_daemon");

  // Fail loudly on a malformed endpoint. Without this a missing baseUrl produces a
  // relative fetch that quietly returns the app's own index.html, surfacing as an
  // "Unexpected token '<'" JSON error a long way from the actual cause.
  if (!resolved?.baseUrl || !resolved.token) {
    throw new Error(
      `The analysis daemon returned an unusable endpoint: ${JSON.stringify(resolved)}`,
    );
  }

  endpoint = resolved;
  return endpoint;
}

/**
 * One attempt against the current endpoint. Separated from `request` so a transport
 * failure can be retried against a freshly started daemon.
 */
async function attempt(path: string, init?: RequestInit): Promise<Response> {
  const ep = await connect();

  return fetch(`${ep.baseUrl}${path}`, {
    ...init,
    headers: {
      "X-GSS-Token": ep.token,
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await attempt(path, init);
  } catch {
    // fetch rejects (rather than returning a status) when nothing answered at all -
    // the daemon exited, so the cached endpoint points at a closed port. Drop it and
    // try once more; start_daemon notices the child is gone and starts a new one.
    endpoint = null;

    try {
      // A replacement daemon is a FRESH session: no archive, no candidates, no review
      // queue. Reconnecting alone leaves every call succeeding against an empty index,
      // which reads as "nothing matched" rather than "the engine restarted". Re-seed it
      // before retrying, and let the app know so it can refresh what it is showing.
      await attempt("/session/init", { method: "POST" });
      // Awaited: the handler reloads the session, and the retry below would otherwise
      // return data the UI has not reconciled with the freshly initialized daemon.
      await onDaemonRestarted?.();

      response = await attempt(path, init);
    } catch (e) {
      throw new Error(
        `The analysis engine stopped and could not be restarted. ${messageOf(e)}`,
      );
    }
  }

  if (!response.ok) {
    // The daemon returns {"error": "..."} for the cases it maps deliberately.
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.error ?? `${init?.method ?? "GET"} ${path} failed (${response.status})`);
  }

  // An empty body is not always a 204. The verdict endpoints answer 200 with
  // Content-Length: 0, and calling .json() on that throws "Unexpected end of JSON input"
  // a long way from the cause. Treat "no content" as the condition rather than one status
  // code, so this holds for any endpoint that legitimately returns nothing.
  if (response.status === 204 || response.headers.get("content-length") === "0") {
    return undefined as T;
  }

  const body = await response.text();
  return (body ? JSON.parse(body) : undefined) as T;
}

const post = <T>(path: string, body?: unknown): Promise<T> =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

/** URL for an image endpoint, usable directly as an <img src>. */
export async function imageUrl(path: string): Promise<string> {
  const ep = await connect();
  const separator = path.includes("?") ? "&" : "?";
  return `${ep.baseUrl}${path}${separator}token=${encodeURIComponent(ep.token)}`;
}

export const api = {
  // session
  init: () => post<SessionStatus>("/session/init"),
  status: () => request<SessionStatus>("/session/status"),
  loadGeoTiff: (path: string) => post<TileInfo>("/session/load", { path }),
  /** Synthesizes a fresh archive centred on a location that has no loaded coverage. */
  generateArchiveAt: (latitude: number, longitude: number) =>
    post<SessionStatus>("/session/generate", { latitude, longitude }),

  // search -> Results
  searchText: (query: string, topK = 12) =>
    post<SearchResult[]>("/search/text", { query, topK }),
  searchSimilar: (patchId: string, topK = 12) =>
    post<SearchResult[]>("/search/similar", { patchId, topK }),
  searchFeedback: (query: string, relevantPatchIds: string[], irrelevantPatchIds: string[], topK = 12) =>
    post<SearchResult[]>("/search/feedback", { query, relevantPatchIds, irrelevantPatchIds, topK }),
  explain: (q: string) =>
    request<{ explanation: string }>(`/search/explain?q=${encodeURIComponent(q)}`),

  // change -> Candidates
  detect: (body: Record<string, unknown>) => post<ChangeRecord[]>("/change/detect", body),
  candidates: () => request<ChangeRecord[]>("/change/candidates"),
  searchChanges: (body: Record<string, unknown>) =>
    post<ChangeSearchResult[]>("/change/search", body),

  // clustering
  // 0.25 is Avalonia's value (MainWindow.axaml.cs OnRunClusteringClicked). It sets DBSCAN
  // membership, so a different default here means the two frontends group the same archive
  // differently - exactly the divergence this project exists to rule out.
  cluster: (epsilonCosine = 0.25, minPoints = 2, maxDistanceKm = 50) =>
    post<Cluster[]>("/cluster", { epsilonCosine, minPoints, maxDistanceKm }),

  // review
  review: () => request<ReviewItem[]>("/review"),
  confirm: (id: string, notes = "") => post<void>(`/review/${id}/confirm`, { notes }),
  reject: (id: string, notes = "") => post<void>(`/review/${id}/reject`, { notes }),
  flag: (id: string, notes = "") => post<void>(`/review/${id}/flag`, { notes }),

  /**
   * Runs the evaluation suite and writes a report. Slow - tens of seconds - and it holds the
   * daemon's session lock while it runs, so the UI must show progress rather than appear hung.
   */
  benchmark: (outputDirectory: string) =>
    post<{ outputDirectory: string }>("/benchmark", { outputDirectory }),

  // export
  exportGeoJson: (path: string) => post<{ path: string }>("/export/geojson", { path }),

  // imagery
  tileUrl: (handle: string, mode: VisualRenderMode = "TrueColorRGB") =>
    imageUrl(`/image/tile?handle=${encodeURIComponent(handle)}&mode=${mode}`),
  heatmapUrl: (changeId?: string) =>
    imageUrl(`/image/heatmap${changeId ? `?changeId=${encodeURIComponent(changeId)}` : ""}`),
  spectralUrl: (changeId: string) => imageUrl(`/image/spectral/${encodeURIComponent(changeId)}`),
  focused: (changeId: string, mode: VisualRenderMode = "TrueColorRGB", heatmap = "CVA") =>
    request<FocusedInspection>(
      `/image/focused/${encodeURIComponent(changeId)}?mode=${mode}&heatmap=${heatmap}`,
    ),
  indices: (changeId: string) =>
    request<IndexHeatmaps>(`/image/indices/${encodeURIComponent(changeId)}`),
};

/** base64 BMP from the daemon -> a data URL an <img> can render. */
export const bmpDataUrl = (base64: string) => `data:image/bmp;base64,${base64}`;
