import { createResource, createSignal, For, Show } from "solid-js";
import IconSearch from "~icons/lucide/search";
import IconSparkles from "~icons/lucide/sparkles";
import IconCrosshair from "~icons/lucide/crosshair";
import IconScanSearch from "~icons/lucide/scan-search";
import { Button } from "~/components/Button";
import { Select } from "~/components/Select";
import {
  Badge,
  Card,
  EmptyState,
  formatCoord,
  formatDate,
  Skeleton,
  SkeletonRows,
  SimilarityScore,
} from "~/components/ui";
import { api, type SearchResult, type VisualRenderMode } from "~/lib/daemon";
import { useApp } from "~/lib/store";

const RENDER_MODE_LABELS: Record<VisualRenderMode, string> = {
  TrueColorRGB: "True colour",
  FalseColorInfrared: "False colour (infrared)",
  SWIR_GeologicalMoisture: "SWIR (moisture)",
  NDVI_Heatmap: "NDVI (vegetation)",
  NDWI_WaterMap: "NDWI (water)",
  NDBI_BuiltUpUrban: "NDBI (built-up)",
  SAR_MicrowaveSimulation: "SAR (microwave)",
  ThermalRadiance: "Thermal",
  ChangeOverlay: "Change overlay",
};

const RENDER_MODES = Object.keys(RENDER_MODE_LABELS) as VisualRenderMode[];

const SENSORS = [
  "Sentinel2_Optical",
  "Sentinel1_SAR",
  "Landsat8_9",
  "ISRO_Bhuvan",
] as const;

const SENSOR_LABELS: Record<(typeof SENSORS)[number], string> = {
  Sentinel2_Optical: "Sentinel-2 (optical)",
  Sentinel1_SAR: "Sentinel-1 (radar)",
  Landsat8_9: "Landsat 8/9",
  ISRO_Bhuvan: "ISRO Bhuvan",
};

const PRESET_QUERIES = [
  "structures near a river",
  "deforestation or cleared land",
  "large vehicle concentrations on open ground",
  "new construction",
  "runway or airfield",
];

/** Centre point of a patch's bounds, for a compact coordinate readout. */
function patchCentre(bounds: SearchResult["patch"]["bounds"]) {
  return {
    latitude: (bounds.minLat + bounds.maxLat) / 2,
    longitude: (bounds.minLon + bounds.maxLon) / 2,
  };
}

function ResultThumbnail(props: { result: SearchResult; renderMode: VisualRenderMode }) {
  // Renders the whole parent tile - there is no per-patch crop endpoint yet, so the
  // thumbnail is wider than the actual result. A cropped endpoint would be a better fit.
  const [url] = createResource(
    () => [props.result.patch.parentTileId, props.renderMode] as const,
    ([handle, mode]) => api.tileUrl(handle, mode),
  );

  return (
    <Show when={url()} fallback={<Skeleton class="size-14 shrink-0 rounded-lg" />}>
      {(src) => (
        <img
          src={src()}
          alt=""
          class="size-14 shrink-0 rounded-lg border border-ed-line object-cover"
        />
      )}
    </Show>
  );
}

function ResultRow(props: { result: SearchResult }) {
  const [state, actions] = useApp();
  const centre = () => patchCentre(props.result.patch.bounds);
  const canInspect = () => actions.canEnter(3);

  return (
    <Card class="flex items-center gap-3 p-3 transition-colors hover:bg-ed-ctl-hover">
      <ResultThumbnail result={props.result} renderMode={state.renderMode} />

      <div class="flex min-w-0 flex-1 flex-col gap-0.5">
        <div class="flex items-center gap-2">
          <span class="truncate text-[13px] text-ed-text-1">{props.result.patch.patchId}</span>
          <Show when={props.result.patch.hasCloudOrShadow}>
            <Badge tone="caution">Cloud/shadow</Badge>
          </Show>
        </div>
        <p class="truncate text-[11px] text-ed-text-3">
          {props.result.patch.platform} · tile {props.result.patch.parentTileId} ·{" "}
          {formatDate(props.result.patch.timestamp)} · {formatCoord(centre())}
        </p>
        <p class="text-[11px] text-ed-text-3">
          quality {props.result.patch.qualityScore.toFixed(2)}
        </p>
      </div>

      <div class="flex shrink-0 flex-col items-end gap-2">
        <SimilarityScore value={props.result.similarityScore} />
        <div class="flex items-center gap-1.5">
          <Button
            variant="gray"
            size="sm"
            onClick={() => actions.findSimilar(props.result.patch.patchId)}
          >
            Find similar
          </Button>
          <Button
            variant="gray"
            size="sm"
            title="Switch to Stage 2 with this Result's location pre-filled."
            onClick={() => actions.targetLocation(centre())}
          >
            <IconCrosshair class="size-3.5" />
            Target here
          </Button>
          <Button
            variant="gray"
            size="sm"
            disabled={!canInspect()}
            title={canInspect() ? undefined : "Change detection has to run first - Stage 3 isn't reachable yet."}
            onClick={() => canInspect() && actions.setStage(3)}
          >
            <IconScanSearch class="size-3.5" />
            Inspect changes
          </Button>
        </div>
      </div>
    </Card>
  );
}

export function StageFindImages() {
  const [state, actions] = useApp();
  const [draft, setDraft] = createSignal(state.query);

  const runSearch = (q: string) => {
    setDraft(q);
    void actions.search(q);
  };

  // Results at or below zero similarity are not Results at all (CONTEXT.md) - filter here
  // so the list itself can never show one, regardless of what the daemon returns.
  // Sensor and order are applied client-side: both are properties of Results already in
  // hand, so re-querying the daemon for them would be a round trip for nothing.
  const visibleResults = () => {
    const kept = state.results.filter(
      (r) =>
        r.similarityScore > 0 &&
        (state.sensorFilter === "All" || r.patch.platform === state.sensorFilter),
    );

    const time = (iso: string) => new Date(iso).getTime();

    switch (state.sortOrder) {
      case "newest":
        return [...kept].sort((a, b) => time(b.patch.timestamp) - time(a.patch.timestamp));
      case "oldest":
        return [...kept].sort((a, b) => time(a.patch.timestamp) - time(b.patch.timestamp));
      case "quality":
        return [...kept].sort((a, b) => b.patch.qualityScore - a.patch.qualityScore);
      case "area":
        // patchWidth * patchHeight is pixel area, not true ground area - GSD varies by
        // platform and tile, so this is a proxy, not a physical measurement.
        return [...kept].sort(
          (a, b) =>
            b.patch.patchWidth * b.patch.patchHeight - a.patch.patchWidth * a.patch.patchHeight,
        );
      default:
        return kept; // the daemon already returns these ranked by similarity
    }
  };

  return (
    <div class="flex h-full min-h-0 flex-col">
      <div class="shrink-0 border-b border-ed-line bg-ed-card-2 px-4 py-3">
        <div class="flex items-center gap-2">
          <div class="relative flex-1">
            <IconSearch class="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-ed-text-3" />
            <input
              data-search-input
              value={draft()}
              onInput={(e) => setDraft(e.currentTarget.value)}
              onKeyDown={(e) => e.key === "Enter" && runSearch(draft())}
              placeholder="Describe what you're looking for…"
              class="h-8 w-full rounded-lg border border-ed-line bg-ed-ctl pl-8 pr-2.5 text-[13px] text-ed-text-1 outline-none transition-[background-color,border-color,box-shadow] duration-200 placeholder:text-ed-text-3 focus-visible:border-ed-accent focus-visible:ring-2 focus-visible:ring-ed-accent/30"
            />
          </div>
          <Button variant="blue" onClick={() => runSearch(draft())}>
            Search
          </Button>
          <Select
            value={state.renderMode}
            title="How the imagery is rendered"
            options={RENDER_MODES.map((m) => ({ value: m, label: RENDER_MODE_LABELS[m] }))}
            onChange={actions.setRenderMode}
          />
        </div>

        <div class="mt-2 flex items-center gap-2">
          <Select
            value={state.sensorFilter}
            title="Show only Results from one sensor"
            options={[
              { value: "All" as const, label: "All sensors" },
              ...SENSORS.map((x) => ({ value: x, label: SENSOR_LABELS[x] })),
            ]}
            onChange={actions.setSensorFilter}
          />

          <Select
            value={state.sortOrder}
            title="Order the Results"
            options={[
              { value: "similarity" as const, label: "Best match first" },
              { value: "newest" as const, label: "Newest first" },
              { value: "oldest" as const, label: "Oldest first" },
              { value: "quality" as const, label: "Best image quality first" },
              { value: "area" as const, label: "Largest area first" },
            ]}
            onChange={actions.setSortOrder}
          />

          <span class="text-[11px] text-ed-text-3">
            {visibleResults().length} shown
          </span>
        </div>

        <Show when={state.explanation}>
          <p class="mt-2 text-[11px] text-ed-text-3">
            <span class="text-ed-text-3">Interpreted as: </span>
            {state.explanation}
          </p>
        </Show>

        <div class="mt-2.5 flex flex-wrap items-center gap-1.5">
          <IconSparkles class="size-3.5 text-ed-text-3" />
          <For each={PRESET_QUERIES}>
            {(q) => (
              <Button variant="gray" size="sm" onClick={() => runSearch(q)}>
                {q}
              </Button>
            )}
          </For>
        </div>
      </div>

      <div class="min-h-0 flex-1 overflow-y-auto p-4">
        <Show when={!state.searching} fallback={<SkeletonRows rows={5} />}>
          <Show
            when={visibleResults().length > 0}
            fallback={
              <EmptyState
                title="No results yet"
                hint="Describe a place, or run one of the preset searches above, to find images that match."
              />
            }
          >
            <div class="flex flex-col gap-2">
              <For each={visibleResults()}>{(r) => <ResultRow result={r} />}</For>
            </div>
          </Show>
        </Show>
      </div>
    </div>
  );
}
