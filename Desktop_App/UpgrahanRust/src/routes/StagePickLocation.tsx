import { createMemo, createSignal, For, Show } from "solid-js";
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
  formatDate,
  Input,
} from "~/components/ui";
import { api, type ChangeSearchResult, type ChangeType } from "~/lib/daemon";
import { useApp } from "~/lib/store";

const CHANGE_TYPE_LABELS: Record<Exclude<ChangeType, "NoChange">, string> = {
  Construction: "Construction",
  Clearance: "Clearance",
  WaterExtentVariation: "Water extent variation",
  RoadDevelopment: "Road development",
  ActivityConcentration: "Activity concentration",
};

const CHANGE_TYPES = Object.keys(CHANGE_TYPE_LABELS) as Array<keyof typeof CHANGE_TYPE_LABELS>;

export function StagePickLocation() {
  const [state, actions] = useApp();

  // Default to the loaded archive's centre, falling back to New Delhi.
  const firstTile = state.session?.tiles[0];
  const initial = firstTile
    ? {
        latitude: (firstTile.bounds.minLat + firstTile.bounds.maxLat) / 2,
        longitude: (firstTile.bounds.minLon + firstTile.bounds.maxLon) / 2,
      }
    : { latitude: 28.61, longitude: 77.2 };

  const [latitude, setLatitude] = createSignal(initial.latitude);
  const [longitude, setLongitude] = createSignal(initial.longitude);
  const [radiusKm, setRadiusKm] = createSignal(5);
  const [changeType, setChangeType] = createSignal<ChangeType | "">("");
  const [startDate, setStartDate] = createSignal("");
  const [endDate, setEndDate] = createSignal("");
  const [minConfidence, setMinConfidence] = createSignal(0);

  // Local to this stage - deliberately not in the global store, per the task spec.
  const [candidates, setCandidates] = createSignal<ChangeSearchResult[]>([]);
  const [searching, setSearching] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

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
      if (type) body.targetChangeType = type;
      if (startDate()) body.startDate = startDate();
      if (endDate()) body.endDate = endDate();

      setCandidates(await api.searchChanges(body));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSearching(false);
    }
  }

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

  const flyToBounds = createMemo(() => {
    const selected = candidates().find((r) => r.record.id === state.selectedCandidateId);
    return selected?.record.bounds;
  });

  return (
    <div class="flex h-full min-h-0">
      <div class="flex w-80 shrink-0 flex-col border-r border-ed-line bg-ed-card-2">
        <div class="flex shrink-0 flex-col gap-3 border-b border-ed-line p-4">
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

          <Field label="Change type">
            <Select
              value={changeType()}
              options={[
                { value: "" as const, label: "All" },
                ...CHANGE_TYPES.map((t) => ({ value: t, label: CHANGE_TYPE_LABELS[t] })),
              ]}
              onChange={(v) => setChangeType(v as ChangeType | "")}
            />
          </Field>

          <div class="grid grid-cols-2 gap-2">
            <Field label="Start date">
              <Input
                type="date"
                value={startDate()}
                onInput={(e) => setStartDate(e.currentTarget.value)}
              />
            </Field>
            <Field label="End date">
              <Input type="date" value={endDate()} onInput={(e) => setEndDate(e.currentTarget.value)} />
            </Field>
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
        </div>

        <div class="min-h-0 flex-1 overflow-y-auto p-3">
          <Show when={error()}>
            {(message) => <ErrorNote message={message()} />}
          </Show>

          <Show
            when={candidates().length > 0}
            fallback={
              <Show when={!searching()}>
                <EmptyState
                  title="No candidates yet"
                  hint="Set a location and radius, then search this area to find unreviewed candidates."
                />
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
            onClick={() => actions.markComplete(2)}
          >
            Use these candidates
          </Button>
        </div>
      </div>

      <div class="min-w-0 flex-1">
        <MapCanvas
          markers={mapMarkers()}
          centre={{ latitude: latitude(), longitude: longitude() }}
          radiusKm={radiusKm()}
          bounds={flyToBounds()}
          onMarkerClick={(id) => actions.select(id)}
        />
      </div>
    </div>
  );
}
