import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { For, Match, Show, Switch, createSignal, onCleanup, onMount, type JSX } from "solid-js";
import toast, { Toaster } from "solid-toast";
import IconX from "~icons/lucide/x";
import IconMinus from "~icons/lucide/minus";
import IconSquare from "~icons/lucide/square";
import IconRepeat from "~icons/lucide/repeat-2";
import IconSatellite from "~icons/lucide/satellite";
import IconCheck from "~icons/lucide/check";
import IconImport from "~icons/lucide/import";
import IconCircleHelp from "~icons/lucide/circle-help";
import IconArrowRight from "~icons/lucide/arrow-right";
import IconGauge from "~icons/lucide/gauge";
import IconLoaderCircle from "~icons/lucide/loader-circle";

import { Button } from "~/components/Button";
import { ErrorNote } from "~/components/ui";
import { api } from "~/lib/daemon";
import { cn } from "~/lib/cn";
import { AppProvider, STAGES, useApp, type StageId } from "~/lib/store";
import { StageFindImages } from "~/routes/StageFindImages";
import { StagePickLocation } from "~/routes/StagePickLocation";
import { StageCheckChanges } from "~/routes/StageCheckChanges";
import { StageGroupPlaces } from "~/routes/StageGroupPlaces";
import { StageReviewExport } from "~/routes/StageReviewExport";

/**
 * Window shell, following Cap's (window-chrome).tsx: a borderless Tauri window with its own
 * drawn titlebar (`data-tauri-drag-region`), hairline-divided from the body.
 */
function Titlebar() {
  const [state, actions] = useApp();
  const [canSwitch, setCanSwitch] = createSignal(false);

  const [benchmarking, setBenchmarking] = createSignal(false);

  /** Evaluation suite. Blocks the daemon for a while, so it reports progress and result. */
  const runBenchmark = async () => {
    const outputDirectory = await save({
      title: "Where should the evaluation report go?",
      defaultPath: "benchmark_results",
    });
    if (typeof outputDirectory !== "string") return;

    setBenchmarking(true);
    const pending = toast.loading("Running the evaluation suite…");
    try {
      const { outputDirectory: written } = await api.benchmark(outputDirectory);
      toast.success(`Evaluation report written to ${written}`, { id: pending });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e), { id: pending });
    } finally {
      setBenchmarking(false);
    }
  };

  const pickGeoTiff = () =>
    pickGeoTiffWith((path) => actions.loadGeoTiff(path)).catch((e) =>
      toast.error(e instanceof Error ? e.message : String(e)),
    );

  onMount(async () => {
    try {
      setCanSwitch(await invoke<boolean>("avalonia_available"));
    } catch {
      setCanSwitch(false);
    }
  });

  const appWindow = getCurrentWindow();

  return (
    <header
      data-tauri-drag-region
      class="flex h-11 shrink-0 select-none items-center justify-between gap-3 border-b border-ed-line bg-ed-card px-3"
    >
      <div data-tauri-drag-region class="flex items-center gap-2">
        <IconSatellite class="size-4 text-ed-accent" />
        <span class="text-[13px] font-medium text-ed-text-1">UpaGraha</span>
        <span class="rounded-md bg-ed-ctl px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-ed-text-3">
          Rust
        </span>
      </div>

      <div data-tauri-drag-region class="flex flex-1 items-center justify-center gap-4">
        <Show when={state.session}>
          {(session) => (
            <div class="flex items-center gap-4 text-[11px] text-ed-text-3">
              <span>{session().indexedPatches} indexed</span>
              <span class="text-ed-line-strong">·</span>
              <span>
                {session().candidates} candidate{session().candidates === 1 ? "" : "s"}
                <Show when={session().highConfidenceCandidates > 0}>
                  {" "}({session().highConfidenceCandidates} strong)
                </Show>
              </span>
            </div>
          )}
        </Show>
      </div>

      <div class="flex items-center gap-1">
        <Button variant="ghost" size="sm" title="Load a GeoTIFF into the archive (Ctrl+O)" onClick={pickGeoTiff}>
          <IconImport class="size-3.5" />
          Load GeoTIFF
        </Button>

        <Button
          variant="ghost"
          size="sm"
          disabled={benchmarking()}
          title="Run the precision/recall evaluation suite and write a report"
          onClick={() => void runBenchmark()}
        >
          <Show when={benchmarking()} fallback={<IconGauge class="size-3.5" />}>
            <IconLoaderCircle class="size-3.5 animate-spin" />
          </Show>
          Benchmark
        </Button>

        <Button
          variant="ghost"
          size="icon"
          title="Keyboard shortcuts (F1)"
          onClick={() => showShortcuts()}
        >
          <IconCircleHelp class="size-3.5" />
        </Button>

        {/* Only offered in the combined bundle where both builds sit side by side. */}
        <Show when={canSwitch()}>
          <Button
            variant="ghost"
            size="sm"
            title="Relaunch in the Avalonia build"
            onClick={() => invoke("switch_to_avalonia").catch(() => undefined)}
          >
            <IconRepeat class="size-3.5" />
            Avalonia UI
          </Button>
        </Show>

        <div class="ml-1 flex items-center">
          <WindowButton label="Minimise" onClick={() => appWindow.minimize()}>
            <IconMinus class="size-3.5" />
          </WindowButton>
          <WindowButton label="Maximise" onClick={() => appWindow.toggleMaximize()}>
            <IconSquare class="size-3" />
          </WindowButton>
          <WindowButton label="Close" danger onClick={() => appWindow.close()}>
            <IconX class="size-3.5" />
          </WindowButton>
        </div>
      </div>
    </header>
  );
}

/** Opens the file picker and ingests the chosen scene. Shared by the button and Ctrl+O. */
async function pickGeoTiffWith(load: (path: string) => Promise<void>) {
  const chosen = await open({
    multiple: false,
    filters: [{ name: "GeoTIFF", extensions: ["tif", "tiff"] }],
  });
  if (typeof chosen === "string") await load(chosen);
}

const SHORTCUTS = [
  ["Ctrl+1 … Ctrl+5", "Jump to a stage"],
  ["Ctrl+O", "Load a GeoTIFF"],
  ["Ctrl+F", "Go to search and focus it"],
  ["Ctrl+W", "Close the window"],
  ["F1", "This list"],
  ["F5", "Re-run the current search"],
  ["C / R", "Confirm or reject the candidate (stage 3)"],
  ["P / N", "Previous or next candidate (stage 3)"],
  ["Esc", "Dismiss"],
] as const;

/** The help toast, shared by the titlebar button and the F1 binding. */
function showShortcuts() {
  toast(
    () => (
      <div class="flex flex-col gap-1">
        <p class="text-[12px] font-medium text-ed-text-1">Keyboard shortcuts</p>
        <For each={SHORTCUTS}>
          {([key, what]) => (
            <p class="text-[11px] text-ed-text-2">
              <span class="font-mono text-ed-text-1">{key}</span> — {what}
            </p>
          )}
        </For>
      </div>
    ),
    { duration: 8000 },
  );
}

function WindowButton(props: {
  label: string;
  danger?: boolean;
  onClick: () => void;
  children: JSX.Element;
}) {
  return (
    <button
      type="button"
      aria-label={props.label}
      title={props.label}
      onClick={props.onClick}
      class={cn(
        "flex size-7 items-center justify-center rounded-md text-ed-text-2 transition-colors duration-100",
        props.danger ? "hover:bg-red-9 hover:text-white" : "hover:bg-ed-ctl-hover hover:text-ed-text-1",
      )}
    >
      {props.children}
    </button>
  );
}

/**
 * Stage rail. Gating mirrors the Avalonia app's CanProceedToStep: a stage opens only once
 * the previous one produced a result, so the workflow cannot be entered out of order.
 */
function StageRail() {
  const [state, actions] = useApp();

  /**
   * Live count per stage, so the rail shows how much each step has produced rather than
   * only whether it has been visited. Zero renders as nothing, not as "0" - an empty
   * badge reads as "nothing here yet" without adding noise to every untouched stage.
   */
  const countFor = (stage: StageId): number =>
    ({
      1: state.results.filter((r) => r.similarityScore > 0).length,
      2: state.candidates.length,
      3: state.candidates.length,
      4: state.clusters.length,
      5: state.review.filter((i) => i.status !== "Pending").length,
    })[stage];

  const canAdvance = () => state.stage < 5 && actions.canEnter((state.stage + 1) as StageId);

  /** Why Next is unavailable, phrased as the thing to do rather than the thing missing. */
  const blockedReason = () =>
    ({
      1: "run a search",
      2: "search an area for candidates",
      3: "run change detection",
      4: "group the places",
      5: "",
    })[state.stage];

  return (
    <nav class="flex shrink-0 items-center gap-1 border-b border-ed-line bg-ed-card px-3 py-2">
      <For each={STAGES}>
        {(stage) => {
          const done = () => state.completed[stage.id];
          const current = () => state.stage === stage.id;
          const enabled = () => actions.canEnter(stage.id);

          return (
            <button
              type="button"
              disabled={!enabled()}
              onClick={() => actions.setStage(stage.id)}
              title={enabled() ? stage.caption : `Finish "${STAGES[stage.id - 2]?.name}" first`}
              class={cn(
                "group flex items-center gap-2 rounded-full px-3 py-1.5 text-[12px] transition-[background-color,color,opacity] duration-200",
                current()
                  ? "bg-ed-accent text-white"
                  : enabled()
                    ? "text-ed-text-2 hover:bg-ed-ctl-hover hover:text-ed-text-1"
                    : "cursor-not-allowed text-ed-text-3 opacity-50",
              )}
            >
              <span
                class={cn(
                  "flex size-4 items-center justify-center rounded-full text-[10px] font-medium",
                  current()
                    ? "bg-white/25 text-white"
                    : done()
                      ? "bg-verdict-verified/20 text-verdict-verified"
                      : "bg-ed-ctl-active text-ed-text-3",
                )}
              >
                <Show when={done() && !current()} fallback={stage.id}>
                  <IconCheck class="size-2.5" />
                </Show>
              </span>
              <span class="hidden sm:inline">{stage.name}</span>

              <Show when={countFor(stage.id) > 0}>
                <span
                  class={cn(
                    "hidden rounded-full px-1.5 text-[10px] font-medium tabular-nums md:inline",
                    current() ? "bg-white/20 text-white" : "bg-ed-ctl-active text-ed-text-2",
                  )}
                >
                  {countFor(stage.id)}
                </span>
              </Show>
            </button>
          );
        }}
      </For>

      <div class="flex-1" />

      <Show when={STAGES.find((s) => s.id === state.stage)}>
        {(stage) => (
          <p class="hidden pr-1 text-[11px] text-ed-text-3 lg:block">{stage().caption}</p>
        )}
      </Show>

      {/*
        Avalonia's BtnNextStage. The rail alone leaves the next step implicit; this is the
        one obvious forward action, and it states WHY it is blocked rather than just
        greying out - a disabled control with no reason is the thing users get stuck on.
      */}
      <Show when={state.stage < 5}>
        <Button
          variant="blue"
          size="sm"
          disabled={!canAdvance()}
          title={
            canAdvance()
              ? `Continue to ${STAGES.find((x) => x.id === state.stage + 1)?.name ?? "the next step"}`
              : `Finish this step first — ${blockedReason()}`
          }
          onClick={() => actions.setStage((state.stage + 1) as StageId)}
        >
          Next
          <IconArrowRight class="size-3.5" />
        </Button>
      </Show>
    </nav>
  );
}

function Shell() {
  const [state, actions] = useApp();

  onMount(() => {
    void actions.boot();

    // Keyboard parity with the Avalonia app. Stage-3 verdict keys (C/R/N) are owned by that
    // screen, which knows the selected candidate, so they are deliberately not handled here.
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat) return;

      const target = e.target as HTMLElement | null;
      const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA";

      if (e.ctrlKey) {
        const n = Number(e.key);
        if (n >= 1 && n <= 5 && actions.canEnter(n as StageId)) {
          e.preventDefault();
          actions.setStage(n as StageId);
          return;
        }
        if (e.key === "o" || e.key === "O") {
          e.preventDefault();
          void pickGeoTiffWith((path) => actions.loadGeoTiff(path)).catch((err) =>
            toast.error(err instanceof Error ? err.message : String(err)),
          );
          return;
        }
        if (e.key === "f" || e.key === "F") {
          e.preventDefault();
          // Jump to stage 1 first: focusing a search box that is not mounted is a silent
          // no-op, which is what this did before.
          actions.setStage(1);
          queueMicrotask(() =>
            document.querySelector<HTMLInputElement>("input[data-search-input]")?.focus(),
          );
          return;
        }

        if (e.key === "w" || e.key === "W") {
          e.preventDefault();
          void getCurrentWindow().close();
          return;
        }
      }

      // F1 was advertised in the shortcut list and the help tooltip but never bound.
      if (e.key === "F1") {
        e.preventDefault();
        showShortcuts();
        return;
      }

      if (e.key === "F5" && !typing) {
        e.preventDefault();
        void actions.search(state.query);
        return;
      }

      if (e.key === "Escape") {
        (document.activeElement as HTMLElement | null)?.blur();
        actions.clearError();
      }
    };

    // Tauri delivers OS file drops as a window event, not an HTML drag event.
    const unlistenDrop = getCurrentWindow().onDragDropEvent((event) => {
      if (event.payload.type !== "drop") return;

      const scenes = event.payload.paths.filter((p) => /\.tiff?$/i.test(p));
      if (scenes.length === 0) {
        toast.error("Only .tif and .tiff scenes can be loaded.");
        return;
      }

      // Ingest every dropped scene, sequentially so the index stays consistent. Taking
      // only the first silently discarded the rest of a multi-file drop.
      void (async () => {
        for (const scene of scenes) await actions.loadGeoTiff(scene);
        if (scenes.length > 1) toast.success(`Loaded ${scenes.length} scenes.`);
      })();
    });

    window.addEventListener("keydown", onKey);
    onCleanup(() => {
      window.removeEventListener("keydown", onKey);
      void unlistenDrop.then((off) => off());
    });
  });

  return (
    <div class="flex h-screen w-screen flex-col overflow-hidden bg-ed-window text-ed-text-1">
      <Titlebar />
      <StageRail />

      <Show when={state.error}>
        {(message) => (
          <div class="px-4 pt-3">
            <ErrorNote message={message()} onRetry={() => actions.boot()} />
          </div>
        )}
      </Show>

      <main class="min-h-0 flex-1 overflow-hidden">
        <Switch>
          <Match when={state.connecting}>
            <div class="flex h-full items-center justify-center">
              <div class="flex flex-col items-center gap-2">
                <div class="size-5 animate-spin rounded-full border-2 border-ed-ctl-active border-t-ed-accent" />
                <p class="text-[12px] text-ed-text-3">Starting the analysis engine…</p>
              </div>
            </div>
          </Match>
          <Match when={state.stage === 1}>
            <StageFindImages />
          </Match>
          <Match when={state.stage === 2}>
            <StagePickLocation />
          </Match>
          <Match when={state.stage === 3}>
            <StageCheckChanges />
          </Match>
          <Match when={state.stage === 4}>
            <StageGroupPlaces />
          </Match>
          <Match when={state.stage === 5}>
            <StageReviewExport />
          </Match>
        </Switch>
      </main>

      <StatusBar />
    </div>
  );
}

/** Bottom strip, mirroring the Avalonia app's telemetry row. */
function StatusBar() {
  const [state] = useApp();

  const stage = () => STAGES.find((s) => s.id === state.stage);
  const sensors = () =>
    Array.from(new Set((state.session?.tiles ?? []).map((t) => t.platform))).join(", ") || "none";

  /** Ground sampling distance of the loaded scenes, when they agree on one. */
  const groundSampling = () => {
    const values = new Set((state.session?.tiles ?? []).map((t) => t.groundSamplingDistanceMeters));
    return values.size === 1 ? `${[...values][0]}m GSD` : null;
  };

  // navigator.onLine is the platform's own answer, so there is no polling or probe traffic
  // to maintain. It reports link state, not reachability - enough to tell an analyst
  // whether basemap tiles can be fetched at all, which is what the indicator claims.
  const [online, setOnline] = createSignal(navigator.onLine);
  onMount(() => {
    const up = () => setOnline(true);
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    onCleanup(() => {
      window.removeEventListener("online", up);
      window.removeEventListener("offline", down);
    });
  });

  return (
    <footer class="flex h-7 shrink-0 items-center gap-4 border-t border-ed-line bg-ed-card px-3 text-[11px] text-ed-text-3">
      <span>{state.session?.indexedPatches ?? 0} indexed</span>
      <span class="text-ed-line-strong">·</span>
      <span>
        {state.candidates.length === 0
          ? "no candidates"
          : `${state.candidates.length} candidates (${state.candidates.filter((c) => c.confidence >= 0.85).length} strong)`}
      </span>
      <span class="text-ed-line-strong">·</span>
      <span>{state.review.filter((i) => i.status !== "Pending").length} reviewed</span>

      <div class="flex-1" />

      <span class="truncate">{sensors()}</span>
      <Show when={groundSampling()}>
        {(gsd) => (
          <>
            <span class="text-ed-line-strong">·</span>
            <span>{gsd()}</span>
          </>
        )}
      </Show>
      <span class="text-ed-line-strong">·</span>
      <span
        class="flex items-center gap-1.5"
        title={
          online()
            ? "Network is reachable, so basemap tiles can be fetched."
            : "No network. Analysis is unaffected - it runs locally - but basemaps render from the tile cache only."
        }
      >
        <span
          class={cn(
            "size-1.5 rounded-full",
            online() ? "bg-verdict-verified" : "bg-ed-text-3",
          )}
        />
        {online() ? "Online" : "Air-gapped"}
      </span>
      <span class="text-ed-line-strong">·</span>
      <span class="text-ed-text-2">
        Stage {state.stage} — {stage()?.name}
      </span>
      <Show when={state.searching || state.detecting || state.clustering}>
        <span class="size-2 animate-pulse rounded-full bg-ed-accent" title="Working" />
      </Show>
    </footer>
  );
}

export function App() {
  return (
    <AppProvider>
      <Shell />
      <Toaster
        position="bottom-right"
        toastOptions={{
          duration: 3500,
          style: {
            padding: "8px 14px",
            "border-radius": "12px",
            border: "1px solid var(--ed-line)",
            "font-size": "13px",
            "background-color": "var(--ed-card)",
            color: "var(--ed-text-1)",
            "box-shadow": "var(--ed-pop-shadow)",
          },
        }}
      />
    </AppProvider>
  );
}
