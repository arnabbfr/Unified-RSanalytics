import { save } from "@tauri-apps/plugin-dialog";
import { createMemo, createSignal, For, Show } from "solid-js";
import toast from "solid-toast";
import IconDownload from "~icons/lucide/download";
import IconLoaderCircle from "~icons/lucide/loader-circle";

import { Button } from "~/components/Button";
import {
  Card,
  ChangeTypeChip,
  EmptyState,
  EvidenceScore,
  formatArea,
  formatDate,
  StatusBadge,
} from "~/components/ui";
import { cn } from "~/lib/cn";
import { api, type ReviewItem, type ReviewStatus } from "~/lib/daemon";
import { useApp } from "~/lib/store";

const SUMMARY_TONES: Record<ReviewStatus, string> = {
  Confirmed: "text-verdict-verified",
  Rejected: "text-verdict-rejected",
  Flagged: "text-yellow-11",
  Pending: "text-verdict-pending",
};

const SUMMARY_LABELS: Record<ReviewStatus, string> = {
  Confirmed: "Verified",
  Rejected: "Rejected",
  Flagged: "Flagged",
  Pending: "Pending",
};

function SummaryCard(props: { status: ReviewStatus; count: number }) {
  return (
    <Card class="flex flex-1 flex-col gap-0.5 p-3">
      <span class={cn("text-xl font-medium tabular-nums", SUMMARY_TONES[props.status])}>
        {props.count}
      </span>
      <span class="text-[10px] uppercase tracking-wide text-ed-text-3">
        {SUMMARY_LABELS[props.status]}
      </span>
    </Card>
  );
}

function ReviewRow(props: { item: ReviewItem }) {
  return (
    <Card class="flex flex-col gap-2 p-3">
      <div class="flex items-center justify-between gap-3">
        <div class="flex items-center gap-2">
          <StatusBadge status={props.item.status} />
          <ChangeTypeChip type={props.item.record.type} />
        </div>
        <EvidenceScore value={props.item.record.confidence} />
      </div>

      <p class="text-[11px] text-ed-text-3">
        {formatArea(props.item.record.areaSqMeters)} · T1 {formatDate(props.item.record.timestampT1)} · T2{" "}
        {formatDate(props.item.record.timestampT2)} · added {formatDate(props.item.addedTimestamp)}
        <Show when={props.item.decisionTimestamp}>
          {(ts) => <> · decided {formatDate(ts())}</>}
        </Show>
      </p>

      <Show when={props.item.analystComments}>
        <p class="text-[11px] leading-relaxed text-ed-text-2">{props.item.analystComments}</p>
      </Show>
    </Card>
  );
}

/** Stage 5. Verdict summary, the review queue, and a W3C PROV-O audit trail export. */
export function StageReviewExport() {
  const [state, actions] = useApp();
  const [exporting, setExporting] = createSignal(false);
  const [refreshing, setRefreshing] = createSignal(false);

  const counts = createMemo(() => {
    const c: Record<ReviewStatus, number> = { Confirmed: 0, Rejected: 0, Flagged: 0, Pending: 0 };
    for (const item of state.review) c[item.status]++;
    return c;
  });

  // Stable sort: Pending items surface first, everything else keeps its existing order.
  const sortedReview = createMemo(() =>
    [...state.review].sort((a, b) => Number(b.status === "Pending") - Number(a.status === "Pending")),
  );

  const handleExport = async () => {
    const path = await save({
      defaultPath: "analyst_review_audit.geojson",
      filters: [{ name: "GeoJSON", extensions: ["geojson"] }],
    });
    if (!path) return;

    setExporting(true);
    try {
      await api.exportGeoJson(path);
      toast.success("Audit trail exported");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await actions.refreshReview();
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div class="flex h-full min-h-0 flex-col gap-4 overflow-y-auto p-4">
      <div class="flex shrink-0 gap-3">
        <For each={["Confirmed", "Rejected", "Flagged", "Pending"] as const}>
          {(status) => <SummaryCard status={status} count={counts()[status]} />}
        </For>
      </div>

      <div class="flex shrink-0 flex-col gap-1.5">
        <div class="flex items-center gap-2">
          <Button variant="blue" disabled={exporting()} onClick={() => void handleExport()}>
            <Show when={exporting()} fallback={<IconDownload class="size-3.5" />}>
              <IconLoaderCircle class="size-3.5 animate-spin" />
            </Show>
            Export audit trail (GeoJSON)
          </Button>
          <Button variant="ghost" size="sm" disabled={refreshing()} onClick={() => void handleRefresh()}>
            <IconLoaderCircle class={cn("size-3.5", refreshing() && "animate-spin")} />
            Refresh
          </Button>
        </div>
        <p class="text-[11px] text-ed-text-3">
          The export is a W3C PROV-O provenance record: every verdict, its author and its timestamp.
        </p>
      </div>

      <Show
        when={sortedReview().length > 0}
        fallback={
          <EmptyState
            title="Nothing to review yet"
            hint="Confirm, reject or flag a candidate in Check Changes to populate the queue."
          />
        }
      >
        <div class="flex flex-col gap-2">
          <For each={sortedReview()}>{(item) => <ReviewRow item={item} />}</For>
        </div>
      </Show>
    </div>
  );
}
