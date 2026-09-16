import { createEffect, createMemo, createResource, createSignal, For, onCleanup, onMount, Show } from "solid-js";
import IconActivity from "~icons/lucide/activity";
import IconChartLine from "~icons/lucide/chart-line";
import IconCheck from "~icons/lucide/check";
import IconChevronLeft from "~icons/lucide/chevron-left";
import IconChevronRight from "~icons/lucide/chevron-right";
import IconEye from "~icons/lucide/eye";
import IconFlag from "~icons/lucide/flag";
import IconLayers from "~icons/lucide/layers";
import IconLoaderCircle from "~icons/lucide/loader-circle";
import IconX from "~icons/lucide/x";

import { Button } from "~/components/Button";
import { Select, type SelectOption } from "~/components/Select";
import {
  Card,
  changeTypeLabel,
  ChangeTypeChip,
  EmptyState,
  ErrorNote,
  EvidenceScore,
  formatArea,
  formatAreaWithHectares,
  formatCoord,
  formatDate,
  Metric,
  Skeleton,
  StatusBadge,
} from "~/components/ui";
import { cn } from "~/lib/cn";
import {
  api,
  bmpDataUrl,
  type ChangeRecord,
  type ReviewItem,
  type ReviewStatus,
  type TileInfo,
  type VisualRenderMode,
} from "~/lib/daemon";
import { useApp } from "~/lib/store";

// ponytail: no UI exposed for these two knobs - fixed defaults. Add controls if analysts ask.
// The values are Avalonia's (MainWindow.axaml.cs OnRerunDetectionClicked), not a local
// choice: patch size sets the detection grid and min confidence sets what is admitted as a
// candidate, so differing here would give the two frontends different candidate sets from
// the same archive.
const PATCH_SIZE = 16;
const MIN_CONFIDENCE = 0.65;

const BLINK_INTERVAL_MS = 600;

type MainKey = "t1" | "t2" | "overlay" | "ndvi" | "ndbi" | "ndwi" | "bsi";

const MAIN_LABELS: Record<MainKey, string> = {
  t1: "Before (T1)",
  t2: "After (T2)",
  overlay: "Change overlay",
  ndvi: "Vegetation (NDVI)",
  ndbi: "Built-up (NDBI)",
  ndwi: "Water (NDWI)",
  bsi: "Bare soil (BSI)",
};

/** Base-imagery render modes that make sense in the triptych (excludes ChangeOverlay - that's its own panel). */
const RENDER_MODE_OPTIONS: readonly SelectOption<VisualRenderMode>[] = [
  { value: "TrueColorRGB", label: "True colour" },
  { value: "FalseColorInfrared", label: "False colour (infrared)" },
  { value: "SWIR_GeologicalMoisture", label: "SWIR (moisture)" },
  { value: "NDVI_Heatmap", label: "NDVI (heatmap)" },
  { value: "NDWI_WaterMap", label: "NDWI (water)" },
  { value: "NDBI_BuiltUpUrban", label: "NDBI (built-up)" },
  { value: "SAR_MicrowaveSimulation", label: "SAR (microwave sim)" },
  { value: "ThermalRadiance", label: "Thermal radiance" },
];

type IndexKey = "ndvi" | "ndbi" | "ndwi" | "bsi";

/** Which `record.metrics` key backs each index card's delta badge. */
const DELTA_METRIC_KEY: Record<IndexKey, string> = {
  ndvi: "DeltaNDVI",
  ndbi: "DeltaNDBI",
  ndwi: "DeltaNDWI",
  bsi: "DeltaBSI",
};

const DELTA_AXIS_CAPTION: Record<IndexKey, string> = {
  ndvi: "Loss ◀ 0 ▶ Growth",
  ndbi: "◀ 0 ▶ Built-up",
  ndwi: "Drying ◀ 0 ▶ Water",
  bsi: "◀ 0 ▶ Soil",
};

const formatDelta = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;

// Below this magnitude a spectral index delta is noise, not evidence - it does not clear
// the bar to be worth a sentence in the evidence panel.
const EVIDENCE_SENTENCE_THRESHOLD = 0.05;

/** Human-readable bullets over the four spectral deltas, for the evidence panel. */
function evidenceSentences(metrics: Record<string, number>): string[] {
  const lines: string[] = [];

  const ndvi = metrics.DeltaNDVI;
  if (ndvi !== undefined && Math.abs(ndvi) >= EVIDENCE_SENTENCE_THRESHOLD) {
    lines.push(
      ndvi < 0
        ? `Vegetation loss: NDVI fell by ${Math.abs(ndvi).toFixed(3)}`
        : `Vegetation gain: NDVI rose by ${ndvi.toFixed(3)}`,
    );
  }

  const ndbi = metrics.DeltaNDBI;
  if (ndbi !== undefined && ndbi >= EVIDENCE_SENTENCE_THRESHOLD) {
    lines.push(`Built-up signal increased by ${ndbi.toFixed(3)}`);
  }

  const bsi = metrics.DeltaBSI;
  if (bsi !== undefined && bsi >= EVIDENCE_SENTENCE_THRESHOLD) {
    lines.push(`Bare soil exposure increased by ${bsi.toFixed(3)}`);
  }

  const ndwi = metrics.DeltaNDWI;
  if (ndwi !== undefined && Math.abs(ndwi) >= EVIDENCE_SENTENCE_THRESHOLD) {
    lines.push(
      ndwi < 0
        ? `Water signal fell by ${Math.abs(ndwi).toFixed(3)}`
        : `Water signal increased by ${ndwi.toFixed(3)}`,
    );
  }

  return lines;
}

/**
 * Horizontal strip of the 4-scene archive time series (`state.session.tiles`). Clicking a
 * pass re-runs detection with that tile as T2, comparing it against the fixed T1 baseline.
 */
function PassTimeline(props: {
  tiles: TileInfo[];
  t2Handle: string | undefined;
  onSelect: (handle: string) => void;
}) {
  const t1Handle = () => props.tiles[0]?.handle;

  return (
    <div class="flex flex-col gap-1.5">
      <span class="text-[10px] uppercase tracking-wide text-ed-text-3">Archive passes</span>
      <div class="flex flex-wrap gap-1.5">
        <For each={props.tiles}>
          {(tile) => {
            const isBaseline = () => tile.handle === t1Handle();
            const isSelectedT2 = () => tile.handle === props.t2Handle;
            return (
              <button
                type="button"
                // Selecting the baseline would re-run detection with T1 as its own T2: a
                // guaranteed no-change pass that wipes the candidate set. It stays visible
                // because the analyst needs to see WHICH pass is the baseline.
                disabled={isBaseline()}
                onClick={() => props.onSelect(tile.handle)}
                title={
                  isBaseline()
                    ? "Baseline (T1) - every other pass is compared against this one"
                    : "Compare the baseline against this pass as T2"
                }
                class={cn(
                  "rounded-full border px-2.5 py-1 font-mono text-[11px] tabular-nums transition-colors duration-200",
                  isSelectedT2()
                    ? "border-ed-accent bg-ed-accent text-white"
                    : isBaseline()
                      ? "cursor-default border-ed-accent/40 text-ed-accent"
                      : "border-ed-line text-ed-text-2 hover:bg-ed-ctl-hover",
                )}
              >
                {formatDate(tile.acquisitionTimestamp)}
                <Show when={isBaseline()}> · T1</Show>
              </button>
            );
          }}
        </For>
      </div>
      <p class="text-[11px] text-ed-text-3">
        Selecting a pass re-runs detection, comparing the baseline (T1) against that date.
      </p>
    </div>
  );
}

/**
 * The analyst's verdict on a candidate.
 *
 * The review queue is authoritative, not the record's two booleans: Flagged deliberately
 * asserts neither confirmed nor rejected, so deriving from the booleans alone rendered a
 * freshly flagged candidate as "Pending" - indistinguishable from one nobody had looked at.
 * The booleans are the fallback for a record that predates the queue entry.
 */
function candidateStatus(record: ChangeRecord, review: ReviewItem[]): ReviewStatus {
  const item = review.find((i) => i.record.id === record.id);
  if (item) return item.status;
  if (record.confirmedByAnalyst) return "Confirmed";
  if (record.rejectedByAnalyst) return "Rejected";
  return "Pending";
}

function CandidateRow(props: {
  record: ChangeRecord;
  review: ReviewItem[];
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={props.onSelect}
      class={cn(
        "flex w-full flex-col gap-1.5 rounded-lg px-2.5 py-2 text-left transition-colors duration-200 hover:bg-ed-ctl-hover",
        props.selected && "bg-ed-ctl-active",
      )}
    >
      <div class="flex items-center justify-between gap-2">
        <ChangeTypeChip type={props.record.type} />
        <StatusBadge status={candidateStatus(props.record, props.review)} />
      </div>
      <EvidenceScore value={props.record.confidence} />
      <p class="text-[11px] text-ed-text-3">
        {formatArea(props.record.areaSqMeters)} · {formatCoord(props.record.center)}
      </p>
    </button>
  );
}

function DetectToggle(props: {
  label: string;
  title: string;
  active: boolean;
  onToggle: () => void;
}) {
  return (
    <Button
      variant="gray"
      size="sm"
      data-selected={props.active}
      title={props.title}
      onClick={props.onToggle}
    >
      <Show when={props.active}>
        <IconCheck class="size-3.5" />
      </Show>
      {props.label}
    </Button>
  );
}

/** Before/after/overlay triptych, spectral indices, blink compare and the re-run controls. */
function CandidateInspector(props: { record: ChangeRecord }) {
  const [state, actions] = useApp();

  const [enableRadiometricNormalization, setEnableRadiometricNormalization] = createSignal(true);
  const [enableQualityMasking, setEnableQualityMasking] = createSignal(true);
  const [enableJitterSuppression, setEnableJitterSuppression] = createSignal(true);

  const [mainKey, setMainKey] = createSignal<MainKey>("overlay");
  const [blinking, setBlinking] = createSignal(false);
  const [blinkFrame, setBlinkFrame] = createSignal<"t1" | "t2">("t1");
  const [renderMode, setRenderMode] = createSignal<VisualRenderMode>("TrueColorRGB");

  // Which loaded tile stands in as T2 for the multi-pass timeline; defaults to the store's
  // own default (3rd tile) so the timeline starts on whatever detect() ran with initially.
  const [t2Handle, setT2Handle] = createSignal<string | undefined>(state.session?.tiles[2]?.handle);

  const [focused, { refetch: refetchFocused }] = createResource(
    () => [props.record.id, renderMode()] as const,
    ([id, mode]) => api.focused(id, mode),
  );
  const [indices, { refetch: refetchIndices }] = createResource(
    () => props.record.id,
    (id) => api.indices(id),
  );
  const [spectral, { refetch: refetchSpectral }] = createResource(
    () => props.record.id,
    (id) => api.spectralUrl(id),
  );
  // The chart URL resolves even when the daemon fails to render it (imageUrl only builds a
  // string); the actual failure only shows up once the <img> tries to load it.
  const [spectralLoadFailed, setSpectralLoadFailed] = createSignal(false);
  createEffect(() => {
    props.record.id;
    setSpectralLoadFailed(false);
  });

  const selectPass = (handle: string) => {
    setT2Handle(handle);
    void actions.detect({ t2Handle: handle });
  };

  // Reset the viewer whenever the selected candidate changes.
  createEffect(() => {
    props.record.id;
    setMainKey("overlay");
    setBlinking(false);
  });

  // Blink compare: alternate T1/T2 every ~600ms while enabled. Cleaned up on toggle-off and unmount.
  createEffect(() => {
    if (!blinking()) return;
    const timer = setInterval(
      () => setBlinkFrame((f) => (f === "t1" ? "t2" : "t1")),
      BLINK_INTERVAL_MS,
    );
    onCleanup(() => clearInterval(timer));
  });

  const sources = createMemo<Record<MainKey, string | undefined>>(() => {
    const f = focused();
    const idx = indices();
    return {
      t1: f && bmpDataUrl(f.t1),
      t2: f && bmpDataUrl(f.t2),
      overlay: f && bmpDataUrl(f.overlay),
      ndvi: idx && bmpDataUrl(idx.ndvi),
      ndbi: idx && bmpDataUrl(idx.ndbi),
      ndwi: idx && bmpDataUrl(idx.ndwi),
      bsi: idx && bmpDataUrl(idx.bsi),
    };
  });

  const activeKey = () => (blinking() ? blinkFrame() : mainKey());
  const mainSrc = () => sources()[activeKey()];

  const pickMain = (key: MainKey) => {
    setBlinking(false);
    setMainKey(key);
  };

  const rerun = () =>
    actions.detect({
      // Without this the store falls back to tiles[2], so changing a detection option after
      // picking a pass quietly re-ran against a different T2 than the timeline showed.
      t2Handle: t2Handle(),
      enableRadiometricNormalization: enableRadiometricNormalization(),
      enableQualityMasking: enableQualityMasking(),
      enableJitterSuppression: enableJitterSuppression(),
      patchSize: PATCH_SIZE,
      minConfidence: MIN_CONFIDENCE,
    });

  return (
    <div class="flex flex-col gap-4">
      <PassTimeline tiles={state.session?.tiles ?? []} t2Handle={t2Handle()} onSelect={selectPass} />

      <div class="flex flex-wrap items-center gap-2">
        <div class="flex flex-col gap-1">
          <span class="text-[10px] uppercase tracking-wide text-ed-text-3">Base imagery</span>
          <Select
            value={renderMode()}
            options={RENDER_MODE_OPTIONS}
            onChange={setRenderMode}
            title="Spectral render mode for the before/after/overlay triptych"
          />
        </div>

        <DetectToggle
          label="Calibrate brightness"
          title="PIF/IRLS radiometric normalisation"
          active={enableRadiometricNormalization()}
          onToggle={() => setEnableRadiometricNormalization((v) => !v)}
        />
        <DetectToggle
          label="Ignore cloudy pixels"
          title="Quality masking"
          active={enableQualityMasking()}
          onToggle={() => setEnableQualityMasking((v) => !v)}
        />
        <DetectToggle
          label="Suppress alignment jitter"
          title="Sub-pixel registration jitter suppression"
          active={enableJitterSuppression()}
          onToggle={() => setEnableJitterSuppression((v) => !v)}
        />
        <Button variant="blue" size="sm" disabled={state.detecting} onClick={rerun}>
          <Show when={state.detecting} fallback={<IconActivity class="size-3.5" />}>
            <IconLoaderCircle class="size-3.5 animate-spin" />
          </Show>
          Re-run detection
        </Button>

        <Button
          variant="gray"
          size="sm"
          class="ml-auto"
          data-selected={blinking()}
          title="Alternate the before/after images every 600ms to spot the change."
          onClick={() => setBlinking((v) => !v)}
        >
          <IconEye class="size-3.5" />
          Blink compare
        </Button>
      </div>

      <Show
        when={!focused.error}
        fallback={
          <ErrorNote
            message={focused.error instanceof Error ? focused.error.message : String(focused.error)}
            onRetry={() => refetchFocused()}
          />
        }
      >
        <div class="flex flex-col gap-1.5">
          <div class="flex h-64 items-center justify-center overflow-hidden rounded-xl border border-ed-line bg-ed-stage">
            <Show when={mainSrc()} fallback={<Skeleton class="h-full w-full" />}>
              {(src) => (
                <img src={src()} alt={MAIN_LABELS[activeKey()]} class="h-full w-full object-contain" />
              )}
            </Show>
          </div>
          <p class="text-[11px] text-ed-text-3">
            {MAIN_LABELS[activeKey()]}
            <Show when={blinking()}> · blinking</Show>
          </p>
        </div>

        <div class="grid grid-cols-3 gap-2">
          <For each={["t1", "t2", "overlay"] as const}>
            {(key) => (
              <button
                type="button"
                onClick={() => pickMain(key)}
                class={cn(
                  "flex flex-col gap-1 overflow-hidden rounded-lg border p-1 transition-colors duration-200",
                  !blinking() && mainKey() === key
                    ? "border-ed-accent"
                    : "border-ed-line hover:border-ed-line-strong",
                )}
              >
                <div class="flex h-16 items-center justify-center overflow-hidden rounded-md bg-ed-stage">
                  <Show when={sources()[key]} fallback={<Skeleton class="h-full w-full" />}>
                    {(src) => <img src={src()} alt={MAIN_LABELS[key]} class="h-full w-full object-cover" />}
                  </Show>
                </div>
                <span class="truncate text-[10px] uppercase tracking-wide text-ed-text-3">
                  {MAIN_LABELS[key]}
                </span>
              </button>
            )}
          </For>
        </div>
      </Show>

      <div class="flex flex-col gap-2">
        <div class="flex items-center gap-1.5 text-[11px] font-medium text-ed-text-2">
          <IconLayers class="size-3.5" />
          Spectral indices
        </div>
        <Show
          when={!indices.error}
          fallback={
            <ErrorNote
              message={indices.error instanceof Error ? indices.error.message : String(indices.error)}
              onRetry={() => refetchIndices()}
            />
          }
        >
          <div class="grid grid-cols-4 gap-2">
            <For each={["ndvi", "ndbi", "ndwi", "bsi"] as const}>
              {(key) => {
                const delta = () => props.record.metrics[DELTA_METRIC_KEY[key]];
                return (
                  <button
                    type="button"
                    onClick={() => pickMain(key)}
                    class={cn(
                      "flex flex-col gap-1 overflow-hidden rounded-lg border p-1 transition-colors duration-200",
                      !blinking() && mainKey() === key
                        ? "border-ed-accent"
                        : "border-ed-line hover:border-ed-line-strong",
                    )}
                  >
                    <div class="relative flex h-14 items-center justify-center overflow-hidden rounded-md bg-ed-stage">
                      <Show when={sources()[key]} fallback={<Skeleton class="h-full w-full" />}>
                        {(src) => <img src={src()} alt={MAIN_LABELS[key]} class="h-full w-full object-cover" />}
                      </Show>
                      <Show when={delta() !== undefined}>
                        <span class="absolute bottom-1 right-1 rounded bg-ed-card/90 px-1 py-0.5 font-mono text-[10px] tabular-nums text-ed-text-1">
                          {formatDelta(delta()!)}
                        </span>
                      </Show>
                    </div>
                    <span class="truncate text-[10px] uppercase tracking-wide text-ed-text-3">
                      {MAIN_LABELS[key]}
                    </span>
                    <span class="truncate text-[9px] text-ed-text-3">{DELTA_AXIS_CAPTION[key]}</span>
                  </button>
                );
              }}
            </For>
          </div>
        </Show>
      </div>

      <div class="flex flex-col gap-2 border-t border-ed-line pt-3">
        <div class="flex items-center gap-1.5 text-[11px] font-medium text-ed-text-2">
          <IconChartLine class="size-3.5" />
          Spectral profile
        </div>
        <p class="text-[11px] text-ed-text-3">
          Per-band reflectance for this site, before (T1) vs after (T2).
        </p>
        <Show
          when={!spectral.error && !spectralLoadFailed()}
          fallback={
            <ErrorNote
              message={
                spectral.error instanceof Error
                  ? spectral.error.message
                  : "Could not load the spectral profile chart."
              }
              onRetry={() => {
                setSpectralLoadFailed(false);
                refetchSpectral();
              }}
            />
          }
        >
          <Show when={spectral()} fallback={<Skeleton class="h-40 w-full rounded-lg" />}>
            {(src) => (
              <img
                src={src()}
                alt="Per-band reflectance, before vs after"
                class="w-full rounded-lg border border-ed-line bg-ed-stage object-contain"
                onError={() => setSpectralLoadFailed(true)}
              />
            )}
          </Show>
        </Show>
      </div>
    </div>
  );
}

/** Metrics, detector notes and the confirm/reject/flag verdict bar for the selected candidate. */
function EvidencePanel(props: { record: ChangeRecord; onVerdictSubmitted: () => void }) {
  const [, actions] = useApp();
  const [notes, setNotes] = createSignal("");

  createEffect(() => {
    props.record.id;
    setNotes("");
  });

  const metricEntries = createMemo(() => Object.entries(props.record.metrics));

  // Belief, not fact: a Candidate is unverified until an analyst confirms it (CONTEXT.md),
  // so this reads "Likely", never "Confirmed".
  const headline = createMemo(
    () =>
      `Likely ${changeTypeLabel(props.record.type).toLowerCase()}, covering ${formatArea(props.record.areaSqMeters)}, first visible ${formatDate(props.record.earliestObservationTimestamp)}.`,
  );

  const sentences = createMemo(() => evidenceSentences(props.record.metrics));

  const submit = (verdict: "confirm" | "reject" | "flag") => {
    void actions.verdict(props.record.id, verdict, notes());
    // Confirm/Reject are done with this candidate, so move on. Flag means "come back to
    // this one later" - auto-advancing away from it would defeat the point of flagging.
    if (verdict !== "flag") props.onVerdictSubmitted();
  };

  return (
    <div class="flex h-full min-h-0 flex-col gap-4 overflow-y-auto">
      <div>
        <p class="mb-2 text-[10px] uppercase tracking-wide text-ed-text-3">Evidence</p>
        <p class="mb-3 text-[13px] text-ed-text-1">{headline()}</p>
        <div class="grid grid-cols-2 gap-3">
          <Metric label="Type" value={props.record.type} />
          <Metric label="Evidence score" value={props.record.confidence.toFixed(2)} mono />
          <Metric label="Area" value={formatAreaWithHectares(props.record.areaSqMeters)} />
          <Metric label="Affected pixels" value={props.record.affectedPixels.toLocaleString()} mono />
          <Metric label="T1 date" value={formatDate(props.record.timestampT1)} />
          <Metric label="T2 date" value={formatDate(props.record.timestampT2)} />
          <Metric label="Onset (estimated)" value={formatDate(props.record.earliestObservationTimestamp)} />
          <Metric label="Centre" value={formatCoord(props.record.center)} mono />
        </div>
      </div>

      <div class="flex flex-col gap-1.5 border-t border-ed-line pt-3">
        <span class="text-[10px] uppercase tracking-wide text-ed-text-3">What the indices show</span>
        <Show
          when={sentences().length > 0}
          fallback={
            <div class="flex items-start gap-1.5 text-[11px] text-ed-text-2">
              <span class="mt-1 size-1.5 shrink-0 rounded-full bg-ed-text-3" />
              <span>The change was detected on combined spectral magnitude rather than any single index.</span>
            </div>
          }
        >
          <For each={sentences()}>
            {(line) => (
              <div class="flex items-start gap-1.5 text-[11px] text-ed-text-2">
                <span class="mt-1 size-1.5 shrink-0 rounded-full bg-ed-accent" />
                <span>{line}</span>
              </div>
            )}
          </For>
        </Show>
      </div>

      <Show when={metricEntries().length > 0}>
        <div class="flex flex-col gap-1 border-t border-ed-line pt-3">
          <span class="text-[10px] uppercase tracking-wide text-ed-text-3">Detector metrics</span>
          <For each={metricEntries()}>
            {([key, value]) => (
              <div class="flex items-center justify-between text-[11px] font-mono text-ed-text-2">
                <span class="truncate">{key}</span>
                <span class="tabular-nums">{value.toFixed(3)}</span>
              </div>
            )}
          </For>
        </div>
      </Show>

      <Show when={props.record.processingNotes}>
        <p class="border-t border-ed-line pt-3 text-[11px] leading-relaxed text-ed-text-3">
          {props.record.processingNotes}
        </p>
      </Show>

      <div class="mt-auto flex flex-col gap-2 border-t border-ed-line pt-3">
        <textarea
          value={notes()}
          onInput={(e) => setNotes(e.currentTarget.value)}
          placeholder="Analyst notes…"
          rows={3}
          class="resize-none rounded-lg border border-ed-line bg-ed-ctl px-2.5 py-2 text-[12px] text-ed-text-1 outline-none transition-colors duration-200 placeholder:text-ed-text-3 focus-visible:border-ed-accent focus-visible:ring-2 focus-visible:ring-ed-accent/30"
        />
        <div class="flex items-center gap-2">
          <Button variant="verify" size="sm" class="flex-1" onClick={() => submit("confirm")}>
            <IconCheck class="size-3.5" />
            Confirm
          </Button>
          <Button variant="destructive" size="sm" class="flex-1" onClick={() => submit("reject")}>
            <IconX class="size-3.5" />
            Reject
          </Button>
          <Button variant="gray" size="sm" class="flex-1" onClick={() => submit("flag")}>
            <IconFlag class="size-3.5" />
            Flag
          </Button>
        </div>
      </div>
    </div>
  );
}

/** "Candidate N of M", type + centre, and Prev/Next - sits above the inspector. */
function CandidateNavHeader(props: {
  record: ChangeRecord;
  index: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  return (
    <div class="mb-3 flex items-center justify-between gap-3">
      <div class="flex min-w-0 items-center gap-2">
        <span class="shrink-0 text-[11px] text-ed-text-3">
          Candidate {props.index + 1} of {props.total}
        </span>
        <ChangeTypeChip type={props.record.type} />
        <span class="truncate font-mono text-[11px] text-ed-text-3">
          {formatCoord(props.record.center)}
        </span>
      </div>
      <div class="flex shrink-0 items-center gap-1.5">
        <Button
          variant="gray"
          size="xs"
          title="Previous candidate (P)"
          disabled={props.index <= 0}
          onClick={props.onPrev}
        >
          <IconChevronLeft class="size-3.5" />
          Prev
        </Button>
        <Button
          variant="gray"
          size="xs"
          title="Next candidate (N)"
          disabled={props.index < 0 || props.index >= props.total - 1}
          onClick={props.onNext}
        >
          Next
          <IconChevronRight class="size-3.5" />
        </Button>
      </div>
    </div>
  );
}

/**
 * Stage 3. Three-region layout: candidate list, spectral inspector, evidence + verdict panel.
 * Keyboard: C confirm, R reject, N next candidate, P previous candidate - suppressed while a
 * text field is focused.
 */
export function StageCheckChanges() {
  const [state, actions] = useApp();
  const record = createMemo(() => actions.selectedCandidate());
  const candidateIndex = createMemo(() =>
    state.candidates.findIndex((c) => c.id === state.selectedCandidateId),
  );

  const nextCandidateId = (afterId: string): string | null => {
    const idx = state.candidates.findIndex((c) => c.id === afterId);
    return state.candidates[idx + 1]?.id ?? null;
  };

  const prevCandidateId = (beforeId: string): string | null => {
    const idx = state.candidates.findIndex((c) => c.id === beforeId);
    return idx > 0 ? (state.candidates[idx - 1]?.id ?? null) : null;
  };

  const goToNext = () => {
    const current = state.selectedCandidateId;
    if (!current) return;
    const next = nextCandidateId(current);
    if (next) actions.select(next);
  };

  const goToPrev = () => {
    const current = state.selectedCandidateId;
    if (!current) return;
    const prev = prevCandidateId(current);
    if (prev) actions.select(prev);
  };

  const handleKeydown = (e: KeyboardEvent) => {
    const target = e.target as HTMLElement | null;
    if (target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT")) return;

    const current = record();
    if (e.key === "c" || e.key === "C") {
      if (current) {
        void actions.verdict(current.id, "confirm", "");
        goToNext();
      }
    } else if (e.key === "r" || e.key === "R") {
      if (current) {
        void actions.verdict(current.id, "reject", "");
        goToNext();
      }
    } else if (e.key === "n" || e.key === "N") {
      goToNext();
    } else if (e.key === "p" || e.key === "P") {
      goToPrev();
    }
  };

  onMount(() => window.addEventListener("keydown", handleKeydown));
  onCleanup(() => window.removeEventListener("keydown", handleKeydown));

  return (
    <div class="flex h-full min-h-0">
      <aside class="flex w-[280px] shrink-0 flex-col gap-1 overflow-y-auto border-r border-ed-line p-2">
        <Show
          when={state.candidates.length > 0}
          fallback={
            <div class="p-2">
              <EmptyState
                title="No candidates yet"
                hint="Run detection from the previous stage to see candidates here."
              />
            </div>
          }
        >
          <For each={state.candidates}>
            {(c) => (
              <CandidateRow
                record={c}
                review={state.review}
                selected={c.id === state.selectedCandidateId}
                onSelect={() => actions.select(c.id)}
              />
            )}
          </For>
        </Show>
      </aside>

      <section class="min-w-0 flex-1 overflow-y-auto p-4">
        <Show
          when={record()}
          fallback={
            <EmptyState
              title="Select a candidate"
              hint="Pick a candidate on the left to inspect its spectral evidence."
            />
          }
        >
          {(r) => (
            <>
              <CandidateNavHeader
                record={r()}
                index={candidateIndex()}
                total={state.candidates.length}
                onPrev={goToPrev}
                onNext={goToNext}
              />
              <CandidateInspector record={r()} />
            </>
          )}
        </Show>
      </section>

      <aside class="w-[300px] shrink-0 border-l border-ed-line p-4">
        <Show when={record()} fallback={<Card padded class="h-full" />}>
          {(r) => <EvidencePanel record={r()} onVerdictSubmitted={goToNext} />}
        </Show>
      </aside>
    </div>
  );
}
