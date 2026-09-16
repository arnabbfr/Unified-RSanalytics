import { createEffect, createMemo, createSignal, For, Show } from "solid-js";
import IconChevronRight from "~icons/lucide/chevron-right";
import { Button } from "~/components/Button";
import { EmptyState, formatCoord, Skeleton } from "~/components/ui";
import { MapCanvas } from "~/components/MapCanvas";
import { cn } from "~/lib/cn";
import type { Bbox, Cluster } from "~/lib/daemon";
import { useApp } from "~/lib/store";

const MEMBER_PREVIEW_COUNT = 6;

/** Union of every cluster's enclosing bounds, for the initial map fit. */
function unionBounds(clusters: Cluster[]): Bbox | undefined {
  if (clusters.length === 0) return undefined;
  let minLon = Infinity;
  let minLat = Infinity;
  let maxLon = -Infinity;
  let maxLat = -Infinity;
  for (const c of clusters) {
    minLon = Math.min(minLon, c.enclosingBounds.minLon);
    minLat = Math.min(minLat, c.enclosingBounds.minLat);
    maxLon = Math.max(maxLon, c.enclosingBounds.maxLon);
    maxLat = Math.max(maxLat, c.enclosingBounds.maxLat);
  }
  return { minLon, minLat, maxLon, maxLat };
}

/**
 * Cohesion bar, written inline rather than reusing <EvidenceScore>: cohesion measures how
 * alike a group's members are, not evidence for a change, so it deliberately doesn't share
 * that component's semantics or tooltip copy.
 */
function CohesionBar(props: { value: number }) {
  const pct = () => Math.round(Math.max(0, Math.min(1, props.value)) * 100);
  return (
    <div
      class="flex items-center gap-2"
      title="Cohesion: how alike the members of this group look to each other, 0 to 1."
    >
      <div class="h-1 w-14 overflow-hidden rounded-full bg-ed-ctl-active">
        <div
          class="h-full rounded-full bg-ed-accent transition-[width] duration-200"
          style={{ width: `${pct()}%` }}
        />
      </div>
      <span class="font-mono text-[11px] tabular-nums text-ed-text-2">
        {props.value.toFixed(2)}
      </span>
    </div>
  );
}

function ClusterCard(props: {
  cluster: Cluster;
  selected: boolean;
  onSelect: () => void;
  ref: (el: HTMLDivElement) => void;
}) {
  const shown = createMemo(() => props.cluster.members.slice(0, MEMBER_PREVIEW_COUNT));
  const remaining = createMemo(() => props.cluster.members.length - shown().length);

  return (
    <div
      ref={props.ref}
      onClick={props.onSelect}
      class={cn(
        "flex cursor-pointer flex-col gap-3 overflow-hidden rounded-xl border bg-ed-card p-4 shadow-ed-card transition-colors",
        props.selected
          ? "border-ed-accent ring-1 ring-ed-accent"
          : "border-ed-line hover:bg-ed-ctl-hover",
      )}
    >
      <div class="flex items-start justify-between gap-3">
        <div class="flex flex-col gap-0.5">
          <p class="text-[13px] text-ed-text-1">{props.cluster.label}</p>
          <p class="text-[11px] text-ed-text-3">
            {props.cluster.memberCount} member{props.cluster.memberCount === 1 ? "" : "s"}
          </p>
        </div>
        <CohesionBar value={props.cluster.cohesionScore} />
      </div>

      <div class="grid grid-cols-2 gap-2 text-[11px] text-ed-text-3">
        <div class="flex flex-col gap-0.5">
          <span class="text-[10px] uppercase tracking-wide text-ed-text-3">Centre</span>
          <span class="font-mono tabular-nums text-ed-text-2">
            {formatCoord(props.cluster.centre)}
          </span>
        </div>
        <div class="flex flex-col gap-0.5">
          <span class="text-[10px] uppercase tracking-wide text-ed-text-3">Bounds</span>
          <span class="font-mono tabular-nums text-ed-text-2">
            {props.cluster.enclosingBounds.minLat.toFixed(2)},{" "}
            {props.cluster.enclosingBounds.minLon.toFixed(2)} →{" "}
            {props.cluster.enclosingBounds.maxLat.toFixed(2)},{" "}
            {props.cluster.enclosingBounds.maxLon.toFixed(2)}
          </span>
        </div>
      </div>

      <div class="flex flex-col gap-1 border-t border-ed-line pt-2.5">
        <span class="text-[10px] uppercase tracking-wide text-ed-text-3">Members</span>
        <div class="flex flex-col gap-0.5 font-mono text-[11px] text-ed-text-2">
          <For each={shown()}>{(patch) => <span class="truncate">{patch.patchId}</span>}</For>
        </div>
        <Show when={remaining() > 0}>
          <span class="text-[11px] text-ed-text-3">+{remaining()} more</span>
        </Show>
      </div>
    </div>
  );
}

function ClusterCardSkeleton() {
  return (
    <div class="flex flex-col gap-3 rounded-xl border border-ed-line bg-ed-card p-4">
      <div class="flex items-center justify-between gap-3">
        <Skeleton class="h-3.5 w-2/5" />
        <Skeleton class="h-2.5 w-14" />
      </div>
      <Skeleton class="h-8 w-full" />
      <Skeleton class="h-12 w-full" />
    </div>
  );
}

export function StageGroupPlaces() {
  const [state, actions] = useApp();

  const [selectedId, setSelectedId] = createSignal<number | null>(null);
  const [focusBounds, setFocusBounds] = createSignal<Bbox | undefined>(undefined);
  const cardRefs = new Map<number, HTMLDivElement>();

  // Refit to the union of every cluster's bounds whenever a fresh set of clusters lands
  // (first grouping pass, or a re-group), clearing whatever was individually selected.
  let fittedClusters: Cluster[] | null = null;
  createEffect(() => {
    if (state.clusters.length > 0 && state.clusters !== fittedClusters) {
      fittedClusters = state.clusters;
      setFocusBounds(unionBounds(state.clusters));
      setSelectedId(null);
    }
  });

  const markers = createMemo(() =>
    state.clusters.map((c) => ({
      id: String(c.clusterId),
      lat: c.centre.latitude,
      lon: c.centre.longitude,
      selected: c.clusterId === selectedId(),
    })),
  );

  function selectCard(clusterId: number) {
    setSelectedId(clusterId);
    const cluster = state.clusters.find((c) => c.clusterId === clusterId);
    if (cluster) setFocusBounds(cluster.enclosingBounds);
  }

  function selectMarker(id: string) {
    const clusterId = Number(id);
    setSelectedId(clusterId);
    cardRefs.get(clusterId)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  return (
    <div class="flex h-full min-h-0 flex-col">
      <div class="shrink-0 border-b border-ed-line bg-ed-card-2 px-4 py-3">
        <div class="flex items-center gap-3">
          <Button variant="blue" disabled={state.clustering} onClick={() => actions.cluster()}>
            Group similar places
          </Button>
          <p
            class="text-[11px] text-ed-text-3"
            title="Groups are formed with a density-based clustering pass over cosine similarity of image embeddings plus geographic distance."
          >
            Groups sites that look similar to each other and sit close together on the ground.
          </p>
        </div>
      </div>

      <Show
        when={!state.clustering}
        fallback={
          <div class="min-h-0 flex-1 overflow-y-auto p-4">
            <div class="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
              <For each={Array.from({ length: 6 })}>{() => <ClusterCardSkeleton />}</For>
            </div>
          </div>
        }
      >
        <Show
          when={state.clusters.length > 0}
          fallback={
            <div class="min-h-0 flex-1 overflow-y-auto p-4">
              <EmptyState
                title="No groups yet"
                hint="Run grouping to see which sites cluster together by appearance and location."
              >
                <Button variant="gray" size="sm" onClick={() => actions.cluster()}>
                  <IconChevronRight class="size-3.5" />
                  Group similar places
                </Button>
              </EmptyState>
            </div>
          }
        >
          {/*
           * Two-pane: map fills the main area, cards sit in a fixed 340px sidebar. At 1440px
           * wide this reads better than a bottom strip - the map is the primary tool for
           * judging how clusters are spread across the ground, so it should keep full height;
           * a bottom strip would compress exactly the dimension that matters here, and cards
           * read fine as a scrollable list at 340px.
           */}
          <div class="flex min-h-0 flex-1">
            <MapCanvas
              class="flex-1"
              markers={markers()}
              bounds={focusBounds()}
              onMarkerClick={selectMarker}
            />
            <div class="w-[340px] shrink-0 overflow-y-auto border-l border-ed-line bg-ed-stage p-3">
              <div class="flex flex-col gap-3">
                <For each={state.clusters}>
                  {(cluster) => (
                    <ClusterCard
                      cluster={cluster}
                      selected={cluster.clusterId === selectedId()}
                      onSelect={() => selectCard(cluster.clusterId)}
                      ref={(el) => cardRefs.set(cluster.clusterId, el)}
                    />
                  )}
                </For>
              </div>
            </div>
          </div>
        </Show>
      </Show>
    </div>
  );
}
