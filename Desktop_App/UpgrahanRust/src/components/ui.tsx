import { cva, type VariantProps } from "cva";
import { For, Show, splitProps, type ComponentProps, type JSX, type ParentProps } from "solid-js";
import { cn } from "~/lib/cn";
import type { ChangeType, ReviewStatus } from "~/lib/daemon";

/**
 * Shared surface vocabulary, following Cap's settings building blocks
 * (apps/desktop/src/routes/(window-chrome)/settings/Setting.tsx): rounded cards on the
 * --ed-card surface, hairline-divided rows, dense 11-13px type.
 */

export function Card(props: ParentProps<{ class?: string; padded?: boolean }>) {
  return (
    <div
      class={cn(
        "overflow-hidden rounded-xl border border-ed-line bg-ed-card shadow-ed-card",
        props.padded && "p-4",
        props.class,
      )}
    >
      {props.children}
    </div>
  );
}

/** A card whose children are hairline-divided rows. Cap's SectionRows. */
export function CardRows(props: ParentProps<{ class?: string }>) {
  return (
    <Card class={cn("divide-y divide-ed-line", props.class)}>{props.children}</Card>
  );
}

export function Row(
  props: ParentProps<{ label: string; description?: string; class?: string }>,
) {
  return (
    <div class={cn("flex flex-row items-center justify-between gap-4 px-4 py-3.5", props.class)}>
      <div class="flex min-w-0 flex-1 flex-col gap-0.5">
        <p class="text-[13px] text-ed-text-1">{props.label}</p>
        <Show when={props.description}>
          <p class="text-xs leading-snug text-ed-text-3">{props.description}</p>
        </Show>
      </div>
      <div class="flex shrink-0 items-center gap-2">{props.children}</div>
    </div>
  );
}

export function SectionHeader(props: { title: string; hint?: string; children?: JSX.Element }) {
  return (
    <div class="mb-2.5 flex items-end justify-between gap-3">
      <div class="flex flex-col gap-0.5">
        <h2 class="text-[13px] font-medium text-ed-text-1">{props.title}</h2>
        <Show when={props.hint}>
          <p class="text-[11px] leading-snug text-ed-text-3">{props.hint}</p>
        </Show>
      </div>
      <Show when={props.children}>
        <div class="flex shrink-0 items-center gap-1.5">{props.children}</div>
      </Show>
    </div>
  );
}

const badgeStyles = cva(
  "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
  {
    defaultVariants: { tone: "neutral" },
    variants: {
      tone: {
        neutral: "bg-ed-ctl text-ed-text-2",
        accent: "bg-ed-accent/15 text-ed-accent-2",
        verified: "bg-verdict-verified/15 text-verdict-verified",
        rejected: "bg-verdict-rejected/15 text-verdict-rejected",
        caution: "bg-yellow-9/15 text-yellow-11",
      },
    },
  },
);

export function Badge(props: ComponentProps<"span"> & VariantProps<typeof badgeStyles>) {
  const [local, rest] = splitProps(props, ["tone", "class"]);
  return <span class={badgeStyles({ tone: local.tone, class: local.class })} {...rest} />;
}

/**
 * Change-type chip. Categorical colour only - a type is a classification, never a severity
 * ranking. Strength lives in the evidence score, shown separately.
 */
const TYPE_STYLE: Record<ChangeType, { label: string; class: string }> = {
  Construction: { label: "Construction", class: "text-type-construction bg-type-construction/12" },
  Clearance: { label: "Clearance", class: "text-type-clearance bg-type-clearance/12" },
  WaterExtentVariation: { label: "Water extent", class: "text-type-water bg-type-water/12" },
  RoadDevelopment: { label: "Road", class: "text-type-road bg-type-road/12" },
  ActivityConcentration: { label: "Activity", class: "text-type-activity bg-type-activity/12" },
  NoChange: { label: "No change", class: "text-type-nochange bg-ed-ctl" },
};

export function ChangeTypeChip(props: { type: ChangeType; class?: string }) {
  return (
    <span
      class={cn(
        "inline-flex items-center rounded-md px-1.5 py-0.5 text-[10px] font-medium",
        TYPE_STYLE[props.type].class,
        props.class,
      )}
    >
      {TYPE_STYLE[props.type].label}
    </span>
  );
}

/** The same label ChangeTypeChip renders, for spots that need the word rather than the chip. */
export const changeTypeLabel = (type: ChangeType) => TYPE_STYLE[type].label;

const STATUS_TONE: Record<ReviewStatus, VariantProps<typeof badgeStyles>["tone"]> = {
  Pending: "neutral",
  Confirmed: "verified",
  Rejected: "rejected",
  Flagged: "caution",
};

/** Verified / Rejected / Flagged / Pending, in CONTEXT.md's vocabulary. */
export function StatusBadge(props: { status: ReviewStatus }) {
  return <Badge tone={STATUS_TONE[props.status]}>{props.status}</Badge>;
}

/**
 * Evidence score, 0..1. Deliberately rendered as a bar plus a 2-dp number and never as
 * "N% likely" - CONTEXT.md forbids presenting it as a calibrated probability.
 */
export function EvidenceScore(props: { value: number; class?: string }) {
  return (
    <div class={cn("flex items-center gap-2", props.class)} title="Evidence score, 0 to 1. Ranks candidates against each other; not a probability.">
      <div class="h-1 w-14 overflow-hidden rounded-full bg-ed-ctl-active">
        <div
          class="h-full rounded-full bg-ed-accent transition-[width] duration-200"
          style={{ width: `${Math.round(Math.max(0, Math.min(1, props.value)) * 100)}%` }}
        />
      </div>
      <span class="font-mono text-[11px] tabular-nums text-ed-text-2">
        {props.value.toFixed(2)}
      </span>
    </div>
  );
}

/**
 * Similarity, -1..1. Shown only for values above zero: a negative score means the opposite
 * of the query and is not a result (CONTEXT.md).
 */
export function SimilarityScore(props: { value: number }) {
  return (
    <span
      class="font-mono text-[11px] tabular-nums text-ed-text-2"
      title="Similarity to your query, -1 to 1. Not a confidence."
    >
      {props.value.toFixed(3)}
    </span>
  );
}

export function Field(props: ParentProps<{ label: string; class?: string }>) {
  return (
    <label class={cn("flex flex-col gap-1", props.class)}>
      <span class="text-[11px] font-medium text-ed-text-2">{props.label}</span>
      {props.children}
    </label>
  );
}

export function Input(props: ComponentProps<"input">) {
  const [local, rest] = splitProps(props, ["class"]);
  return (
    <input
      class={cn(
        "h-8 rounded-lg border border-ed-line bg-ed-ctl px-2.5 text-[13px] text-ed-text-1",
        "placeholder:text-ed-text-3 outline-none transition-[background-color,border-color,box-shadow] duration-200",
        "focus-visible:border-ed-accent focus-visible:ring-2 focus-visible:ring-ed-accent/30",
        local.class,
      )}
      {...rest}
    />
  );
}

/** Skeletons mirror the real layout, as Cap's do, so loading never shifts the page. */
export function Skeleton(props: { class?: string }) {
  return <div class={cn("animate-pulse rounded-md bg-ed-ctl-active", props.class)} />;
}

export function SkeletonRows(props: { rows?: number }) {
  return (
    <div class="flex flex-col gap-2">
      <For each={Array.from({ length: props.rows ?? 4 })}>
        {() => (
          <div class="flex items-center gap-3 rounded-xl border border-ed-line bg-ed-card p-3">
            <Skeleton class="size-12 shrink-0 rounded-lg" />
            <div class="flex flex-1 flex-col gap-1.5">
              <Skeleton class="h-3 w-1/3" />
              <Skeleton class="h-2.5 w-1/2" />
            </div>
            <Skeleton class="h-2.5 w-10" />
          </div>
        )}
      </For>
    </div>
  );
}

export function EmptyState(props: { title: string; hint?: string; children?: JSX.Element }) {
  return (
    <div class="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-ed-line-strong px-6 py-10 text-center">
      <p class="text-[13px] text-ed-text-2">{props.title}</p>
      <Show when={props.hint}>
        <p class="max-w-sm text-[11px] leading-relaxed text-ed-text-3">{props.hint}</p>
      </Show>
      <Show when={props.children}>
        <div class="mt-1.5 flex items-center gap-2">{props.children}</div>
      </Show>
    </div>
  );
}

export function ErrorNote(props: { message: string; onRetry?: () => void }) {
  return (
    <div class="flex items-start gap-3 rounded-xl border border-red-6 bg-red-3/40 px-4 py-3">
      <div class="flex-1">
        <p class="text-[13px] text-red-11">Something went wrong</p>
        <p class="mt-0.5 font-mono text-[11px] leading-relaxed text-ed-text-2">{props.message}</p>
      </div>
      <Show when={props.onRetry}>
        <button
          type="button"
          class="shrink-0 rounded-md px-2 py-1 text-[11px] text-ed-text-2 transition-colors hover:bg-ed-ctl-hover hover:text-ed-text-1"
          onClick={() => props.onRetry?.()}
        >
          Retry
        </button>
      </Show>
    </div>
  );
}

/** Small key/value readout used across the inspector panels. */
export function Metric(props: { label: string; value: string; mono?: boolean }) {
  return (
    <div class="flex flex-col gap-0.5">
      <span class="text-[10px] uppercase tracking-wide text-ed-text-3">{props.label}</span>
      <span class={cn("text-[12px] text-ed-text-1", props.mono && "font-mono tabular-nums")}>
        {props.value}
      </span>
    </div>
  );
}

export const formatArea = (sqm: number) =>
  sqm >= 1_000_000 ? `${(sqm / 1_000_000).toFixed(2)} km²` : `${Math.round(sqm).toLocaleString()} m²`;

/** formatArea plus hectares (1 ha = 10,000 m²), for panels that want the analyst-familiar unit alongside it. */
export const formatAreaWithHectares = (sqm: number) => `${formatArea(sqm)} · ${(sqm / 10_000).toFixed(2)} ha`;

export const formatDate = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });

export const formatCoord = (c: { latitude: number; longitude: number }) =>
  `${c.latitude.toFixed(4)}°, ${c.longitude.toFixed(4)}°`;
