import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import { For, Match, Show, Switch, createSignal, onMount, type JSX } from "solid-js";
import { Toaster } from "solid-toast";
import IconX from "~icons/lucide/x";
import IconMinus from "~icons/lucide/minus";
import IconSquare from "~icons/lucide/square";
import IconRepeat from "~icons/lucide/repeat-2";
import IconSatellite from "~icons/lucide/satellite";
import IconCheck from "~icons/lucide/check";

import { Button } from "~/components/Button";
import { ErrorNote } from "~/components/ui";
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
  const [state] = useApp();
  const [canSwitch, setCanSwitch] = createSignal(false);

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
    </nav>
  );
}

function Shell() {
  const [state, actions] = useApp();

  onMount(() => {
    void actions.boot();

    window.addEventListener("keydown", (e) => {
      if (!e.ctrlKey || e.repeat) return;
      const n = Number(e.key);
      if (n >= 1 && n <= 5 && actions.canEnter(n as StageId)) {
        e.preventDefault();
        actions.setStage(n as StageId);
      }
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
    </div>
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
