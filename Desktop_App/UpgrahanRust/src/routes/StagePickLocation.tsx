import { createEffect, createMemo, createResource, createSignal, For, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { notify, messageOf } from "~/lib/notify";
import IconCheck from "~icons/lucide/check";
import IconDownloadCloud from "~icons/lucide/download-cloud";
import IconImport from "~icons/lucide/import";
import IconRefreshCw from "~icons/lucide/refresh-cw";
import IconRotateCcw from "~icons/lucide/rotate-ccw";
import IconSparkles from "~icons/lucide/sparkles";
import IconX from "~icons/lucide/x";
import { Button } from "~/components/Button";
import { Select } from "~/components/Select";
import { MapCanvas } from "~/components/MapCanvas";
import {
  ChangeTypeChip,
  EmptyState,
  ErrorNote,
  EvidenceScore,
  Field,
  formatArea,
  formatCoord,
  formatDate,
  Input,
  Metric,
  Skeleton,
} from "~/components/ui";
import {
  api,
  bmpDataUrl,
  type ChangeSearchResult,
  type ChangeType,
  type Bbox,
} from "~/lib/daemon";
import { useApp } from "~/lib/store";

const CHANGE_TYPE_LABELS: Record<Exclude<ChangeType, "NoChange">, string> = {
  Construction: "Construction",
  Clearance: "Clearance",
  WaterExtentVariation: "Water extent variation",
  RoadDevelopment: "Road development",
  ActivityConcentration: "Activity concentration",
};

const CHANGE_TYPES = Object.keys(CHANGE_TYPE_LABELS) as Array<keyof typeof CHANGE_TYPE_LABELS>;

/** Formats a plain `YYYY-MM-DD` input value without the local-timezone off-by-one `new Date(iso)` risks. */
function formatDateOnly(isoDate: string): string {
  const [y, m, d] = isoDate.split("-").map(Number);
  if (!y || !m || !d) return isoDate;
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

function toIsoDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

type DateRange = { start: number; end: number };
type DatePresetKey = "custom" | "full" | "q1_2024" | "preOnset" | "onsetWindow" | "postDisturbance";

const DATE_PRESET_OPTIONS: { value: DatePresetKey; label: string }[] = [
  { value: "custom", label: "Custom" },
  { value: "full", label: "Full archive" },
  { value: "q1_2024", label: "Q1 2024" },
  { value: "preOnset", label: "Pre-onset baseline" },
  { value: "onsetWindow", label: "Active onset window" },
  { value: "postDisturbance", label: "Post-disturbance" },
];

const Q1_2024: DateRange = { start: Date.parse("2024-01-01T00:00:00Z"), end: Date.parse("2024-03-31T00:00:00Z") };

// Three fixed offsets inside the archive's bounding box (as fractions of its width/height),
// so each sector lands inside loaded coverage regardless of the archive's actual shape.
const SECTORS = [
  { label: "Sector Alpha", fracLon: 0.22, fracLat: 0.22 },
  { label: "Sector Bravo", fracLon: 0.5, fracLat: 0.5 },
  { label: "Sector Charlie", fracLon: 0.78, fracLat: 0.78 },
] as const;

interface ArchiveBounds {
  minLon: number;
  minLat: number;
  maxLon: number;
  maxLat: number;
}

/** Bounding-box radius in km, from degree spans - same approximation MapCanvas's circle uses. */
function kmSpan(bounds: ArchiveBounds, atLat: number) {
  const latKm = (bounds.maxLat - bounds.minLat) * 110.574;
  const lonKm = (bounds.maxLon - bounds.minLon) * 111.32 * Math.cos((atLat * Math.PI) / 180);
  return { latKm, lonKm };
}

// --- Slippy-map tile maths (standard Web Mercator), for the offline precache walk. ---
function lonToTileX(lon: number, z: number): number {
  return Math.floor(((lon + 180) / 360) * 2 ** z);
}
function latToTileY(lat: number, z: number): number {
  const rad = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * 2 ** z);
}

interface TileProviderInfo {
  id: string;
  urlTemplate: string;
}

const DEFAULT_PROVIDER_ID = "EsriSatellite";
const PRECACHE_ZOOMS = [11, 12, 13, 14];
const PRECACHE_CONCURRENCY = 4;

/** Docked before/after provenance panel for the selected candidate, with deep-verify/verdict actions. */
function ProvenancePanel(props: { result: ChangeSearchResult }) {
  const [state, actions] = useApp();
  const record = () => props.result.record;

  const [focused] = createResource(() => record().id, (id) => api.focused(id));

  const platform = createMemo(() => {
    const tileId = record().tileId;
    const match = state.session?.tiles.find((t) => t.tileId === tileId)?.platform;
    return match ? match.replace(/_/g, " ") : "Unknown sensor";
  });

  return (
    <div class="flex shrink-0 flex-col gap-3 border-t border-ed-line bg-ed-card-2 p-3">
      <div class="flex items-start justify-between gap-3">
        <div class="flex min-w-0 flex-col gap-1.5">
          <div class="flex items-center gap-2">
            <ChangeTypeChip type={record().type} />
            <span class="font-mono text-[11px] text-ed-text-3">{record().id.slice(0, 8)}</span>
          </div>
          <div class="grid grid-cols-3 gap-x-4 gap-y-1.5">
            <Metric label="Centre" value={formatCoord(record().center)} mono />
            <Metric label="Area" value={formatArea(record().areaSqMeters)} />
            <Metric label="Sensor" value={platform()} />
            <Metric label="T1" value={formatDate(record().timestampT1)} />
            <Metric label="T2" value={formatDate(record().timestampT2)} />
            <Metric label="Onset (estimated)" value={formatDate(record().earliestObservationTimestamp)} />
          </div>
        </div>

        <div class="flex shrink-0 flex-col gap-1.5">
          <Button
            variant="blue"
            size="sm"
            title="Open Stage 3 with this candidate already selected."
            onClick={() => actions.setStage(3)}
          >
            Deep verify
          </Button>
          <Button variant="verify" size="sm" onClick={() => void actions.verdict(record().id, "confirm", "")}>
            <IconCheck class="size-3.5" />
            Confirm
          </Button>
          <Button variant="destructive" size="sm" onClick={() => void actions.verdict(record().id, "reject", "")}>
            <IconX class="size-3.5" />
            Reject
          </Button>
        </div>
      </div>

      <div class="grid grid-cols-2 gap-2">
        <For each={["t1", "t2"] as const}>
          {(key) => (
            <div class="flex flex-col gap-1">
              <span class="text-[10px] uppercase tracking-wide text-ed-text-3">
                {key === "t1" ? "Before (T1)" : "After (T2)"}
              </span>
              <div class="flex h-28 items-center justify-center overflow-hidden rounded-lg border border-ed-line bg-ed-stage">
                <Show when={focused()} fallback={<Skeleton class="h-full w-full" />}>
                  {(f) => (
                    <img
                      src={bmpDataUrl(f()[key])}
                      alt={key === "t1" ? "Before" : "After"}
                      class="h-full w-full object-contain"
                    />
                  )}
                </Show>
              </div>
            </div>
          )}
        </For>
      </div>
    </div>
  );
}

/** Shown instead of a bare empty state when an area search returns zero candidates. */
function NoCoverageRecovery(props: {
  requested: { latitude: number; longitude: number };
  coverage: ArchiveBounds | null;
  onResetToCoverage: () => void;
  onGenerateHere: () => void;
  onLoadGeoTiff: () => void;
}) {
  return (
    <EmptyState
      title="No candidates in this area"
      hint="This search landed outside the coverage of the loaded archive."
    >
      <div class="flex w-full flex-col gap-3">
        <div class="grid grid-cols-2 gap-3 rounded-lg border border-ed-line bg-ed-card-2 p-2.5 text-left">
          <Metric label="Requested" value={formatCoord(props.requested)} mono />
          <Metric
            label="Loaded coverage"
            value={
              props.coverage
                ? `${props.coverage.minLat.toFixed(2)}°…${props.coverage.maxLat.toFixed(2)}°, ${props.coverage.minLon.toFixed(2)}°…${props.coverage.maxLon.toFixed(2)}°`
                : "No archive loaded"
            }
            mono
          />
        </div>
        <div class="flex flex-wrap justify-center gap-1.5">
          <Button variant="gray" size="sm" disabled={!props.coverage} onClick={props.onResetToCoverage}>
            <IconRefreshCw class="size-3.5" />
            Reset to loaded coverage
          </Button>
          <Button variant="gray" size="sm" onClick={props.onLoadGeoTiff}>
            <IconImport class="size-3.5" />
            Load a GeoTIFF
          </Button>
          <Button variant="gray" size="sm" onClick={props.onGenerateHere}>
            <IconSparkles class="size-3.5" />
            Generate scenes here
          </Button>
        </div>
      </div>
    </EmptyState>
  );
}

export function StagePickLocation() {
  const [state, actions] = useApp();

  // A location handed off from Stage 1 ("Target here") wins over the archive-centre default.
  // Captured once, synchronously, since this component remounts fresh every time Stage 2 opens.
  // Seed from any pending target so the first paint already shows the right place;
  // the createEffect below handles every subsequent "Target here".
  const pendingTarget = state.pendingTarget;

  const firstTile = state.session?.tiles[0];
  const initial = pendingTarget
    ? { latitude: pendingTarget.latitude, longitude: pendingTarget.longitude }
    : firstTile
      ? {
          latitude: (firstTile.bounds.minLat + firstTile.bounds.maxLat) / 2,
          longitude: (firstTile.bounds.minLon + firstTile.bounds.maxLon) / 2,
        }
      : { latitude: 28.61, longitude: 77.2 };

  const [latitude, setLatitude] = createSignal(initial.latitude);
  const [longitude, setLongitude] = createSignal(initial.longitude);
  const [radiusKm, setRadiusKm] = createSignal(5);
  // "any" rather than "": Kobalte treats an empty-string option value as no-selection and
  // falls back to the placeholder, so the control rendered "Select..." instead of "All".
  const [changeType, setChangeType] = createSignal<Exclude<ChangeType, "NoChange"> | "any">("any");
  const [startDate, setStartDate] = createSignal("");
  const [endDate, setEndDate] = createSignal("");
  const [minConfidence, setMinConfidence] = createSignal(0.5);
  const [datePreset, setDatePreset] = createSignal<DatePresetKey>("custom");

  // Local to this stage - deliberately not in the global store, per the task spec.
  const [candidates, setCandidates] = createSignal<ChangeSearchResult[]>([]);
  const [searching, setSearching] = createSignal(false);
  const [hasSearched, setHasSearched] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  const [caching, setCaching] = createSignal(false);
  const [cacheProgress, setCacheProgress] = createSignal<{ done: number; total: number } | null>(null);

  async function search() {
    setSearching(true);
    setError(null);
    try {
      const body: Record<string, unknown> = {
        center: { latitude: latitude(), longitude: longitude() },
        radiusKm: radiusKm(),
        minConfidence: minConfidence(),
        topK: 50,
      };
      const type = changeType();
      // "any" means no filter. It is a truthy string, so this has to compare against the
      // sentinel rather than test truthiness - otherwise it reaches the daemon and
      // Enum.Parse<ChangeType>("any") throws.
      if (type !== "any") body.targetChangeType = type;
      if (startDate()) body.startDate = startDate();
      if (endDate()) body.endDate = endDate();

      const found = await api.searchChanges(body);
      setCandidates(found);

      // This stage's work happens here rather than in the store, so it reports its own
      // engagement. Only a search the analyst ran counts - not the boot-time pass.
      if (found.length > 0) actions.markTouched(2);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setSearching(false);
      setHasSearched(true);
    }
  }

  // React to pendingTarget rather than reading it once at creation. Reading it at creation
  // only works if this screen is torn down between visits; if it stays mounted, a second
  // "Target here" from stage 1 silently kept the first one's coordinates.
  createEffect(() => {
    const target = state.pendingTarget;
    if (!target) return;

    setLatitude(target.latitude);
    setLongitude(target.longitude);

    // Move the viewport too. Updating the coordinate fields alone left the map sitting
    // wherever it was, so "Target here" appeared to do nothing when the target was
    // off-screen. The box is a rough degree-padding around the point; fitBounds then
    // picks a sensible zoom.
    const pad = 0.01;
    setFlyToBounds({
      minLat: target.latitude - pad,
      maxLat: target.latitude + pad,
      minLon: target.longitude - pad,
      maxLon: target.longitude + pad,
    });

    actions.clearPendingTarget();
    void search();
  });

  function selectCandidate(r: ChangeSearchResult) {
    actions.select(r.record.id);
  }

  const mapMarkers = createMemo(() =>
    candidates().map((r) => ({
      id: r.record.id,
      lat: r.record.center.latitude,
      lon: r.record.center.longitude,
      type: r.record.type,
      selected: r.record.id === state.selectedCandidateId,
    })),
  );

  /**
   * Where the map should move next. A signal rather than a memo because two things drive
   * it - selecting a candidate, and a "Target here" handoff from stage 1 - and whichever
   * happened most recently should win. A memo over selection alone could not express that.
   */
  const [flyToBounds, setFlyToBounds] = createSignal<Bbox | undefined>();

  createEffect(() => {
    const selected = candidates().find((r) => r.record.id === state.selectedCandidateId);
    if (selected) setFlyToBounds(selected.record.bounds);
  });

  const selectedResult = createMemo(
    () => candidates().find((r) => r.record.id === state.selectedCandidateId) ?? null,
  );

  // --- Feature 1/2/3: date period presets, quick-date chips, reset. ---

  const archiveDateRange = createMemo<DateRange>(() => {
    const timestamps = (state.session?.tiles ?? [])
      .map((t) => Date.parse(t.acquisitionTimestamp))
      .filter((ms) => !Number.isNaN(ms))
      .sort((a, b) => a - b);
    if (timestamps.length === 0) return Q1_2024;
    return { start: timestamps[0]!, end: timestamps[timestamps.length - 1]! };
  });

  const datePresetRanges = createMemo<Record<Exclude<DatePresetKey, "custom">, DateRange>>(() => {
    const { start, end } = archiveDateRange();
    const third = (end - start) / 3;
    return {
      full: { start, end },
      q1_2024: Q1_2024,
      preOnset: { start, end: start + third },
      onsetWindow: { start: start + third, end: start + third * 2 },
      postDisturbance: { start: start + third * 2, end },
    };
  });

  function applyDatePreset(key: Exclude<DatePresetKey, "custom">) {
    const range = datePresetRanges()[key];
    setStartDate(toIsoDate(range.start));
    setEndDate(toIsoDate(range.end));
    setDatePreset(key);
    void search();
  }

  function handlePresetSelect(key: DatePresetKey) {
    if (key === "custom") {
      setDatePreset("custom");
      return;
    }
    applyDatePreset(key);
  }

  function editStartDate(value: string) {
    setStartDate(value);
    setDatePreset("custom");
  }
  function editEndDate(value: string) {
    setEndDate(value);
    setDatePreset("custom");
  }

  // --- Feature 4: preset sector buttons. ---

  const archiveBounds = createMemo<ArchiveBounds | null>(() => {
    const tiles = state.session?.tiles ?? [];
    if (tiles.length === 0) return null;
    let minLon = Infinity;
    let minLat = Infinity;
    let maxLon = -Infinity;
    let maxLat = -Infinity;
    for (const t of tiles) {
      minLon = Math.min(minLon, t.bounds.minLon);
      minLat = Math.min(minLat, t.bounds.minLat);
      maxLon = Math.max(maxLon, t.bounds.maxLon);
      maxLat = Math.max(maxLat, t.bounds.maxLat);
    }
    return { minLon, minLat, maxLon, maxLat };
  });

  function jumpToSector(sector: (typeof SECTORS)[number]) {
    const bounds = archiveBounds() ?? {
      minLon: initial.longitude - 0.05,
      maxLon: initial.longitude + 0.05,
      minLat: initial.latitude - 0.05,
      maxLat: initial.latitude + 0.05,
    };
    const lat = bounds.minLat + (bounds.maxLat - bounds.minLat) * sector.fracLat;
    const lon = bounds.minLon + (bounds.maxLon - bounds.minLon) * sector.fracLon;
    const { latKm, lonKm } = kmSpan(bounds, lat);
    const radius = Math.min(Math.max(Math.min(latKm, lonKm) / 6, 1), 25);

    setLatitude(lat);
    setLongitude(lon);
    setRadiusKm(Number(radius.toFixed(2)));
    void search();
  }

  // --- Feature 5: active-criteria summary. ---

  const summary = createMemo(() => {
    const type = changeType();
    const bits = [
      `Within ${radiusKm()} km of ${latitude().toFixed(4)}°, ${longitude().toFixed(4)}°`,
      type === "any" ? "All change types" : CHANGE_TYPE_LABELS[type],
    ];
    const s = startDate();
    const e = endDate();
    if (s && e) bits.push(`${formatDateOnly(s)} – ${formatDateOnly(e)}`);
    else if (s) bits.push(`From ${formatDateOnly(s)}`);
    else if (e) bits.push(`Until ${formatDateOnly(e)}`);
    else bits.push("Full archive");
    bits.push(`evidence ≥ ${minConfidence().toFixed(2)}`);
    return bits.join(" · ");
  });

  // --- Feature 7: offline precache of the current search area. ---

  async function precacheArea() {
    let providers: TileProviderInfo[] = [];
    try {
      providers = await invoke<TileProviderInfo[]>("tile_providers");
    } catch (e) {
      notify.error(messageOf(e));
      return;
    }
    // ponytail: targets the app's default basemap, not whatever MapCanvas's own selector is
    // currently showing - that selection is internal to MapCanvas and not exposed upward.
    const provider = providers.find((p) => p.id === DEFAULT_PROVIDER_ID) ?? providers[0];
    if (!provider) {
      notify.error("No basemap provider is available to cache.");
      return;
    }

    const lat = latitude();
    const lon = longitude();
    const radius = radiusKm();
    const degLat = radius / 110.574;
    const degLon = radius / Math.max(111.32 * Math.cos((lat * Math.PI) / 180), 0.0001);
    const minLat = lat - degLat;
    const maxLat = lat + degLat;
    const minLon = lon - degLon;
    const maxLon = lon + degLon;

    const tiles: { z: number; x: number; y: number }[] = [];
    for (const z of PRECACHE_ZOOMS) {
      const xMin = lonToTileX(minLon, z);
      const xMax = lonToTileX(maxLon, z);
      const yMin = latToTileY(maxLat, z);
      const yMax = latToTileY(minLat, z);
      for (let x = xMin; x <= xMax; x++) {
        for (let y = yMin; y <= yMax; y++) tiles.push({ z, x, y });
      }
    }

    setCaching(true);
    setCacheProgress({ done: 0, total: tiles.length });
    let done = 0;
    let nextIndex = 0;

    async function worker() {
      for (;;) {
        const i = nextIndex++;
        const tile = tiles[i];
        if (!tile) return;
        try {
          const existing = await invoke<number[] | null>("read_cached_tile", {
            provider: provider!.id,
            z: tile.z,
            x: tile.x,
            y: tile.y,
          });
          if (!existing) {
            const url = provider!.urlTemplate
              .replace("{z}", String(tile.z))
              .replace("{x}", String(tile.x))
              .replace("{y}", String(tile.y));
            const response = await fetch(url);
            if (response.ok) {
              const bytes = Array.from(new Uint8Array(await response.arrayBuffer()));
              await invoke("write_cached_tile", { provider: provider!.id, z: tile.z, x: tile.x, y: tile.y, bytes });
            }
          }
        } catch {
          // ponytail: per-tile failure is skipped, not fatal - one bad tile shouldn't abort the run.
        }
        done++;
        setCacheProgress({ done, total: tiles.length });
      }
    }

    await Promise.all(Array.from({ length: PRECACHE_CONCURRENCY }, () => worker()));
    setCaching(false);
    notify.success(`Cached ${tiles.length} tile${tiles.length === 1 ? "" : "s"} for offline use.`);
  }

  // --- Feature 8: no-coverage recovery actions. ---

  function resetToCoverage() {
    const bounds = archiveBounds();
    if (!bounds) return;
    const lat = (bounds.minLat + bounds.maxLat) / 2;
    const lon = (bounds.minLon + bounds.maxLon) / 2;
    const { latKm, lonKm } = kmSpan(bounds, lat);
    const radius = Math.max(Math.max(latKm, lonKm) / 2, 1);

    setLatitude(lat);
    setLongitude(lon);
    setRadiusKm(Number(radius.toFixed(2)));
    void search();
  }

  async function generateHere() {
    await actions.generateArchiveAt(latitude(), longitude());
    void search();
  }

  async function loadGeoTiffThenSearch() {
    try {
      const chosen = await open({ multiple: false, filters: [{ name: "GeoTIFF", extensions: ["tif", "tiff"] }] });
      if (typeof chosen === "string") {
        await actions.loadGeoTiff(chosen);
        void search();
      }
    } catch (e) {
      setError(messageOf(e));
    }
  }

  return (
    <div class="flex h-full min-h-0">
      <div class="flex w-80 shrink-0 flex-col border-r border-ed-line bg-ed-card-2">
        <div class="flex shrink-0 flex-col gap-3 overflow-y-auto border-b border-ed-line p-4">
          <div class="grid grid-cols-2 gap-2">
            <Field label="Latitude">
              <Input
                type="number"
                step="0.0001"
                value={latitude()}
                onInput={(e) => setLatitude(e.currentTarget.valueAsNumber || 0)}
              />
            </Field>
            <Field label="Longitude">
              <Input
                type="number"
                step="0.0001"
                value={longitude()}
                onInput={(e) => setLongitude(e.currentTarget.valueAsNumber || 0)}
              />
            </Field>
          </div>

          <Field label="Radius (km)">
            <Input
              type="number"
              min="0.1"
              step="0.5"
              value={radiusKm()}
              onInput={(e) => setRadiusKm(e.currentTarget.valueAsNumber || 0)}
            />
          </Field>

          <div class="flex flex-wrap gap-1.5">
            <For each={SECTORS}>
              {(sector) => (
                <Button
                  variant="gray"
                  size="sm"
                  title="Jump to this sector of the loaded archive and search it."
                  onClick={() => jumpToSector(sector)}
                >
                  {sector.label}
                </Button>
              )}
            </For>
          </div>

          <Field label="Change type">
            <Select
              value={changeType()}
              options={[
                { value: "any" as const, label: "All change types" },
                ...CHANGE_TYPES.map((t) => ({ value: t, label: CHANGE_TYPE_LABELS[t] })),
              ]}
              onChange={(v) => setChangeType(v as Exclude<ChangeType, "NoChange"> | "any")}
            />
          </Field>

          <Field label="Date period">
            <Select value={datePreset()} options={DATE_PRESET_OPTIONS} onChange={handlePresetSelect} />
          </Field>

          <div class="grid grid-cols-2 gap-2">
            <Field label="Start date">
              <Input type="date" value={startDate()} onInput={(e) => editStartDate(e.currentTarget.value)} />
            </Field>
            <Field label="End date">
              <Input type="date" value={endDate()} onInput={(e) => editEndDate(e.currentTarget.value)} />
            </Field>
          </div>

          <div class="flex flex-wrap gap-1.5">
            <Button variant="gray" size="sm" onClick={() => applyDatePreset("full")}>
              Full archive
            </Button>
            <Button variant="gray" size="sm" onClick={() => applyDatePreset("preOnset")}>
              Pre-onset
            </Button>
            <Button variant="gray" size="sm" onClick={() => applyDatePreset("onsetWindow")}>
              Onset window
            </Button>
            <Button variant="gray" size="sm" onClick={() => applyDatePreset("postDisturbance")}>
              Post-disturbance
            </Button>
            <Button
              variant="gray"
              size="sm"
              title="Reset dates to the full archive span"
              onClick={() => applyDatePreset("full")}
            >
              <IconRotateCcw class="size-3.5" />
              Reset
            </Button>
          </div>

          <Field label="Minimum evidence score">
            <Input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={minConfidence()}
              onInput={(e) => setMinConfidence(e.currentTarget.valueAsNumber || 0)}
            />
          </Field>

          <Button variant="blue" onClick={() => void search()} disabled={searching()}>
            {searching() ? "Searching…" : "Search this area"}
          </Button>
          <p class="text-[11px] text-ed-text-3">{summary()}</p>
        </div>

        <div class="min-h-0 flex-1 overflow-y-auto p-3">
          <Show when={error()}>{(message) => <ErrorNote message={message()} />}</Show>

          <Show
            when={candidates().length > 0}
            fallback={
              <Show when={!searching()}>
                <Show
                  when={hasSearched()}
                  fallback={
                    <EmptyState
                      title="No candidates yet"
                      hint="Set a location and radius, then search this area to find unreviewed candidates."
                    />
                  }
                >
                  <NoCoverageRecovery
                    requested={{ latitude: latitude(), longitude: longitude() }}
                    coverage={archiveBounds()}
                    onResetToCoverage={resetToCoverage}
                    onGenerateHere={() => void generateHere()}
                    onLoadGeoTiff={() => void loadGeoTiffThenSearch()}
                  />
                </Show>
              </Show>
            }
          >
            <p class="mb-2 text-[10px] uppercase tracking-wide text-ed-text-3">
              {candidates().length} candidate{candidates().length === 1 ? "" : "s"}
            </p>
            <div class="flex flex-col gap-2">
              <For each={candidates()}>
                {(r) => (
                  <button
                    type="button"
                    onClick={() => selectCandidate(r)}
                    class={
                      "flex flex-col gap-1.5 rounded-xl border p-3 text-left transition-colors " +
                      (r.record.id === state.selectedCandidateId
                        ? "border-ed-accent bg-ed-accent/10"
                        : "border-ed-line bg-ed-card hover:bg-ed-ctl-hover")
                    }
                  >
                    <div class="flex items-center justify-between gap-2">
                      <ChangeTypeChip type={r.record.type} />
                      <span class="text-[11px] text-ed-text-3">{r.distanceKm.toFixed(2)} km away</span>
                    </div>
                    <EvidenceScore value={r.record.confidence} />
                    <div class="flex items-center justify-between text-[11px] text-ed-text-3">
                      <span>{formatArea(r.record.areaSqMeters)}</span>
                      <span>
                        {formatDate(r.record.timestampT1)} → {formatDate(r.record.timestampT2)}
                      </span>
                    </div>
                  </button>
                )}
              </For>
            </div>
          </Show>
        </div>

        <div class="shrink-0 border-t border-ed-line p-3">
          <Button
            variant="verify"
            class="w-full"
            disabled={candidates().length === 0}
            // Gating is derived from the data now, so stage 3 is already open once
            // candidates exist. This button's job is simply to go there.
            onClick={() => actions.setStage(3)}
          >
            Inspect these candidates
          </Button>
        </div>
      </div>

      <div class="flex min-w-0 flex-1 flex-col">
        <div class="relative min-h-0 flex-1">
          <MapCanvas
            markers={mapMarkers()}
            centre={{ latitude: latitude(), longitude: longitude() }}
            radiusKm={radiusKm()}
            bounds={flyToBounds() ?? archiveBounds() ?? undefined}
            onMarkerClick={(id) => actions.select(id)}
          />

          <div class="absolute bottom-3 right-3 flex flex-col items-end gap-1.5">
            {/*
              Frosted glass: this control sits directly on satellite imagery, which can be any
              brightness, so a translucent fill alone left the label unreadable over pale
              ground. Blurring and desaturating what is behind it restores contrast while
              keeping the see-through look.
            */}
            <Button
              variant="gray"
              size="sm"
              disabled={caching()}
              class="border-white/10 bg-ed-card/60 backdrop-blur-md backdrop-saturate-150 shadow-ed-pop hover:bg-ed-card/80"
              title="Downloads basemap tiles for this search area so the map keeps working offline."
              onClick={() => void precacheArea()}
            >
              <IconDownloadCloud class="size-3.5" />
              {caching() ? "Caching…" : "Cache this area for offline use"}
            </Button>
            <Show when={cacheProgress()}>
              {(p) => (
                <p class="rounded-md border border-white/10 bg-ed-card/60 px-2 py-1 text-[11px] text-ed-text-2 shadow-ed-pop backdrop-blur-md backdrop-saturate-150">
                  Cached {p().done} of {p().total} tiles
                </p>
              )}
            </Show>
          </div>
        </div>

        <Show when={selectedResult()}>{(r) => <ProvenancePanel result={r()} />}</Show>
      </div>
    </div>
  );
}
