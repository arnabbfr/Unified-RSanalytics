import { createEffect, createMemo, createResource, createSignal, For, onCleanup, onMount, Show } from "solid-js";
import IconActivity from "~icons/lucide/activity";
import IconCheck from "~icons/lucide/check";
import IconEye from "~icons/lucide/eye";
import IconFlag from "~icons/lucide/flag";
import IconLayers from "~icons/lucide/layers";
import IconLoaderCircle from "~icons/lucide/loader-circle";
import IconX from "~icons/lucide/x";

import { Button } from "~/components/Button";
import {
  Card,
  ChangeTypeChip,
  EmptyState,
  ErrorNote,
  EvidenceScore,
  formatArea,
  formatCoord,
  formatDate,
  Metric,
  Skeleton,
  StatusBadge,
} from "~/components/ui";
import { cn } from "~/lib/cn";
import { api, bmpDataUrl, type ChangeRecord, type ReviewStatus } from "~/lib/daemon";
import { useApp } from "~/lib/store";

// ponytail: no UI exposed for these two knobs - fixed, sane defaults. Add controls if analysts ask.
const PATCH_SIZE = 64;
const MIN_CONFIDENCE = 0.35;

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

/** Confirmed/Rejected/Pending, derived from the two flags a ChangeRecord actually carries. */
function candidateStatus(record: ChangeRecord): ReviewStatus {
  if (record.confirmedByAnalyst) return "Confirmed";
  if (record.rejectedByAnalyst) return "Rejected";
  return "Pending";
}

function CandidateRow(props: { record: ChangeRecord; selected: boolean; onSelect: () => void }) {
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
        <StatusBadge status={candidateStatus(props.record)} />
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

  const [focused, { refetch: refetchFocused }] = createResource(
    () => props.record.id,
    (id) => api.focused(id),
  );
  const [indices, { refetch: refetchIndices }] = createResource(
    () => props.record.id,
    (id) => api.indices(id),
  );

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
      enableRadiometricNormalization: enableRadiometricNormalization(),
      enableQualityMasking: enableQualityMasking(),
      enableJitterSuppression: enableJitterSuppression(),
      patchSize: PATCH_SIZE,
      minConfidence: MIN_CONFIDENCE,
    });

  return (
    <div class="flex flex-col gap-4">
      <div class="flex flex-wrap items-center gap-2">
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
                  <div class="flex h-14 items-center justify-center overflow-hidden rounded-md bg-ed-stage">
                    <Show when={sources()[key]} fallback={<Skeleton class="h-full w-full" />}>
                      {(src) => <img src={src()} alt={MAIN_LABELS[key]} class="h-full w-full object-cover" />}
                    </Show>
                  </div>
                  <span class="truncate text-[10px] uppercase tracking-wide text-ed-text-3">
                    {MAIN_LABELS[key]}
                  </span>
                  <span class="text-[10px] uppercase tracking-wide text-ed-text-3">{key}</span>
                </button>
              )}
            </For>
          </div>
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

  const submit = (verdict: "confirm" | "reject" | "flag") => {
    void actions.verdict(props.record.id, verdict, notes());
    props.onVerdictSubmitted();
  };

  return (
    <div class="flex h-full min-h-0 flex-col gap-4 overflow-y-auto">
      <div>
        <p class="mb-2 text-[10px] uppercase tracking-wide text-ed-text-3">Evidence</p>
        <div class="grid grid-cols-2 gap-3">
          <Metric label="Type" value={props.record.type} />
          <Metric label="Evidence score" value={props.record.confidence.toFixed(2)} mono />
          <Metric label="Area" value={formatArea(props.record.areaSqMeters)} />
          <Metric label="Affected pixels" value={props.record.affectedPixels.toLocaleString()} mono />
          <Metric label="T1 date" value={formatDate(props.record.timestampT1)} />
          <Metric label="T2 date" value={formatDate(props.record.timestampT2)} />
          <Metric label="Onset (estimated)" value={formatDate(props.record.earliestObservationTimestamp)} />
          <Metric label="Centre" value={formatCoord(props.record.center)} mono />
        </div>
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

/**
 * Stage 3. Three-region layout: candidate list, spectral inspector, evidence + verdict panel.
 * Keyboard: C confirm, R reject, N next candidate - suppressed while a text field is focused.
 */
export function StageCheckChanges() {
  const [state, actions] = useApp();
  const record = createMemo(() => actions.selectedCandidate());

  const nextCandidateId = (afterId: string): string | null => {
    const idx = state.candidates.findIndex((c) => c.id === afterId);
    return state.candidates[idx + 1]?.id ?? null;
  };

  const goToNext = () => {
    const current = state.selectedCandidateId;
    if (!current) return;
    const next = nextCandidateId(current);
    if (next) actions.select(next);
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
          {(r) => <CandidateInspector record={r()} />}
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
