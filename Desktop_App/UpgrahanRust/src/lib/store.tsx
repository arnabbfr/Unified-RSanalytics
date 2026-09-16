import { createContext, useContext, type ParentProps } from "solid-js";
import { createStore } from "solid-js/store";
import {
  api,
  type ChangeRecord,
  type Coord,
  type SensorPlatform,
  type Cluster,
  type ReviewItem,
  type SearchResult,
  type SessionStatus,
  type VisualRenderMode,
} from "~/lib/daemon";

/**
 * Frontend state.
 *
 * Deliberately thin. Everything heavy - the vector index, the engines, the rasters - lives
 * in the daemon behind handles. What is here is selection, view mode and stage gating:
 * roughly the dozen scalars the Avalonia app keeps in MainWindow private fields, minus the
 * things that were only there because there was nowhere else to put them.
 */

export const STAGES = [
  { id: 1, name: "Find Images", caption: "Search the archive for places that look like your description." },
  { id: 2, name: "Pick Location", caption: "Narrow to an area and a time window." },
  { id: 3, name: "Check Changes", caption: "Inspect the spectral evidence for each candidate." },
  { id: 4, name: "Group Places", caption: "Cluster similar sites across the area." },
  { id: 5, name: "Review & Export", caption: "Record verdicts and export the audit trail." },
] as const;

export type StageId = (typeof STAGES)[number]["id"];

interface AppState {
  stage: StageId;
  connecting: boolean;
  error: string | null;

  session: SessionStatus | null;

  // Stage 1
  query: string;
  explanation: string;
  results: SearchResult[];
  renderMode: VisualRenderMode;
  sensorFilter: SensorPlatform | "All";
  sortOrder: "similarity" | "newest" | "oldest" | "quality" | "area";
  searching: boolean;

  // Stage 2 / 3
  candidates: ChangeRecord[];
  selectedCandidateId: string | null;
  detecting: boolean;
  /**
   * A location handed off from a Stage 1 Result via "Target here". Stage 2 (StagePickLocation)
   * should read this on mount to pre-fill its target and then call `clearPendingTarget()` -
   * this store only carries the handoff, it does not consume it.
   */
  pendingTarget: Coord | null;

  // Stage 4
  clusters: Cluster[];
  clustering: boolean;

  // Stage 5
  review: ReviewItem[];

}

const initial: AppState = {
  stage: 1,
  connecting: true,
  error: null,
  session: null,
  query: "large vehicle concentrations on open ground",
  explanation: "",
  results: [],
  renderMode: "TrueColorRGB",
  sensorFilter: "All",
  sortOrder: "similarity",
  searching: false,
  candidates: [],
  selectedCandidateId: null,
  detecting: false,
  pendingTarget: null,
  clusters: [],
  clustering: false,
  review: [],
};

function createAppStore() {
  const [state, setState] = createStore<AppState>({ ...initial });

  const fail = (e: unknown) => setState("error", e instanceof Error ? e.message : String(e));

  const actions = {
    async boot() {
      setState({ connecting: true, error: null });
      try {
        const session = await api.init();
        const candidates = await api.candidates();
        setState({
          session,
          candidates,
          connecting: false,
          selectedCandidateId: candidates[0]?.id ?? null,
        });
        await actions.refreshReview();

        // Open with content rather than an empty state, matching the Avalonia app, which
        // runs a default query AND a default clustering pass as part of initialising the
        // archive. Without the clustering pass stage 4 opens empty and its map never mounts.
        await actions.search(state.query);
        await actions.cluster();
      } catch (e) {
        setState("connecting", false);
        fail(e);
      }
    },

    async search(query: string) {
      setState({ query, searching: true, error: null });
      try {
        const [results, explained] = await Promise.all([
          api.searchText(query),
          api.explain(query),
        ]);
        setState({
          results,
          explanation: explained.explanation,
          searching: false,
        });
      } catch (e) {
        setState("searching", false);
        fail(e);
      }
    },

    async findSimilar(patchId: string) {
      setState({ searching: true, error: null });
      try {
        setState({ results: await api.searchSimilar(patchId), searching: false });
      } catch (e) {
        setState("searching", false);
        fail(e);
      }
    },

    /**
     * `t2Handle` overrides which loaded tile stands in as the "after" pass - used to
     * re-compare the baseline against a different acquisition from the multi-pass timeline.
     * Defaults to the 3rd loaded tile, matching the original T1-vs-T2 behaviour.
     */
    async detect(options: Record<string, unknown> & { t2Handle?: string }) {
      const tiles = state.session?.tiles ?? [];
      if (tiles.length < 3) return;

      setState({ detecting: true, error: null });
      try {
        const candidates = await api.detect({
          t1Handle: tiles[0]!.handle,
          t2Handle: tiles[2]!.handle,
          ...options,
        });
        setState({
          candidates,
          detecting: false,
          // Ids are regenerated on every detect, so the old selection cannot survive.
          selectedCandidateId: candidates[0]?.id ?? null,
        });
        await actions.refreshReview();
      } catch (e) {
        setState("detecting", false);
        fail(e);
      }
    },

    async cluster() {
      setState({ clustering: true, error: null });
      try {
        const clusters = await api.cluster();
        setState({
          clusters,
          clustering: false,
        });
      } catch (e) {
        setState("clustering", false);
        fail(e);
      }
    },

    async refreshReview() {
      try {
        setState("review", await api.review());
      } catch (e) {
        fail(e);
      }
    },

    async verdict(id: string, verdict: "confirm" | "reject" | "flag", notes: string) {
      try {
        await api[verdict](id, notes);
        const candidates = await api.candidates();
        setState({ candidates });
        await actions.refreshReview();
      } catch (e) {
        fail(e);
      }
    },

    /** Ingests a GeoTIFF from disk and re-runs the current query over the new index. */
    async loadGeoTiff(path: string) {
      setState({ searching: true, error: null });
      try {
        await api.loadGeoTiff(path);
        setState("session", await api.status());
        await actions.search(state.query);
      } catch (e) {
        setState("searching", false);
        fail(e);
      }
    },

    /**
     * Synthesizes an archive around a location that fell outside loaded coverage, then
     * refreshes the session-derived state. The caller (Stage 2's area search) is responsible
     * for re-running its own search afterward - that search's parameters are local to the
     * stage, not something the store tracks.
     */
    async generateArchiveAt(latitude: number, longitude: number) {
      setState({ searching: true, error: null });
      try {
        const session = await api.generateArchiveAt(latitude, longitude);
        const candidates = await api.candidates();
        setState({ session, candidates, searching: false });
        await actions.refreshReview();
      } catch (e) {
        setState("searching", false);
        fail(e);
      }
    },

    /** Rocchio relevance feedback: re-ranks using the analyst's confirmed/rejected verdicts. */
    async rerank() {
      const verified = state.review.filter((i) => i.status === "Confirmed");
      const rejected = state.review.filter((i) => i.status === "Rejected");
      if (verified.length === 0 && rejected.length === 0) return;

      setState({ searching: true, error: null });
      try {
        // Feedback is keyed on patches, and a verdict is recorded against a Candidate, so
        // the tile the Candidate sits on is what carries the signal back into the query.
        const patchIdsFor = (ids: string[]) =>
          state.results.filter((r) => ids.includes(r.patch.parentTileId)).map((r) => r.patch.patchId);

        setState({
          results: await api.searchFeedback(
            state.query,
            patchIdsFor(verified.map((i) => i.record.tileId)),
            patchIdsFor(rejected.map((i) => i.record.tileId)),
          ),
          searching: false,
        });
      } catch (e) {
        setState("searching", false);
        fail(e);
      }
    },

    /**
     * Hands a location off to Stage 2 from a Stage 1 Result ("Target here"). Opens the gate
     * on Stage 2 the same way a completed Stage 1 search would.
     */
    targetLocation: (coord: Coord) => {
      setState({ pendingTarget: coord });
      actions.setStage(2);
    },
    clearPendingTarget: () => setState("pendingTarget", null),

    setStage: (stage: StageId) => setState("stage", stage),
    setSensorFilter: (sensorFilter: AppState["sensorFilter"]) => setState("sensorFilter", sensorFilter),
    setSortOrder: (sortOrder: AppState["sortOrder"]) => setState("sortOrder", sortOrder),
    select: (id: string | null) => setState("selectedCandidateId", id),
    setRenderMode: (renderMode: VisualRenderMode) => setState("renderMode", renderMode),
    setQuery: (query: string) => setState("query", query),
    clearError: () => setState("error", null),

    selectedCandidate: () =>
      state.candidates.find((c) => c.id === state.selectedCandidateId) ?? null,

    /** Stage N is reachable once stage N-1 produced a result. Stage 1 is always open. */
    /**
     * What each stage has produced. One derivation, used for both the rail's count badge
     * and the gate, so the two can never disagree.
     */
    stageCount: (stage: StageId): number =>
      ({
        1: state.results.filter((r) => r.similarityScore > 0).length,
        2: state.candidates.length,
        3: state.candidates.length,
        4: state.clusters.length,
        5: state.review.filter((i) => i.status !== "Pending").length,
      })[stage],

    /**
     * A stage opens once the previous one has actually produced something.
     *
     * This used to be a Record of boolean flags written from five different places. Nothing
     * ever set the stage-2 flag, so stage 3 was permanently locked, while the boot-time
     * clustering pass set the stage-4 flag and unlocked stage 5 - a rail showing stages 4
     * and 5 complete but not 3. Deriving it from the data removes the class of bug rather
     * than fixing one instance of it.
     */
    canEnter: (stage: StageId) => stage === 1 || actions.stageCount((stage - 1) as StageId) > 0,

    /** True when this stage has output of its own, i.e. the rail should tick it. */
    isDone: (stage: StageId) => actions.stageCount(stage) > 0,
  };

  return [state, actions] as const;
}

type AppStore = ReturnType<typeof createAppStore>;

const AppContext = createContext<AppStore>();

export function AppProvider(props: ParentProps) {
  return <AppContext.Provider value={createAppStore()}>{props.children}</AppContext.Provider>;
}

export function useApp(): AppStore {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be called inside an AppProvider.");
  return ctx;
}
