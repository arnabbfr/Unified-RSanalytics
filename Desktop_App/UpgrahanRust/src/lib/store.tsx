import { createContext, useContext, type ParentProps } from "solid-js";
import { createStore } from "solid-js/store";
import {
  api,
  type ChangeRecord,
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
  searching: boolean;

  // Stage 2 / 3
  candidates: ChangeRecord[];
  selectedCandidateId: string | null;
  detecting: boolean;

  // Stage 4
  clusters: Cluster[];
  clustering: boolean;

  // Stage 5
  review: ReviewItem[];

  // Gating: a stage unlocks only once the previous one produced something.
  completed: Record<StageId, boolean>;
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
  searching: false,
  candidates: [],
  selectedCandidateId: null,
  detecting: false,
  clusters: [],
  clustering: false,
  review: [],
  completed: { 1: false, 2: false, 3: false, 4: false, 5: false },
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
        // runs a default query as part of initialising the archive.
        await actions.search(state.query);
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
          completed: { ...state.completed, 1: results.length > 0 },
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

    async detect(options: Record<string, unknown>) {
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
          completed: { ...state.completed, 3: candidates.length > 0 },
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
          completed: { ...state.completed, 4: clusters.length > 0 },
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
        setState({ candidates, completed: { ...state.completed, 5: true } });
        await actions.refreshReview();
      } catch (e) {
        fail(e);
      }
    },

    setStage: (stage: StageId) => setState("stage", stage),
    select: (id: string | null) => setState("selectedCandidateId", id),
    setRenderMode: (renderMode: VisualRenderMode) => setState("renderMode", renderMode),
    setQuery: (query: string) => setState("query", query),
    markComplete: (stage: StageId) =>
      setState("completed", { ...state.completed, [stage]: true }),
    clearError: () => setState("error", null),

    selectedCandidate: () =>
      state.candidates.find((c) => c.id === state.selectedCandidateId) ?? null,

    /** Stage N is reachable once stage N-1 produced a result. Stage 1 is always open. */
    canEnter: (stage: StageId) => stage === 1 || state.completed[(stage - 1) as StageId],
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
