import { invoke } from "@tauri-apps/api/core";
import { createResource, createSignal, For, onCleanup, Show } from "solid-js";
import IconRepeat from "~icons/lucide/repeat-2";
import { Button } from "~/components/Button";

/**
 * Handing the session to the Avalonia build.
 *
 * The interaction is Cap's tauri <-> GPUI switch, ported: the toggle itself is the
 * confirmation, and committing to it starts a countdown rather than acting immediately.
 * The numeral is on screen from the first frame and the explanation fades through
 * underneath it, so the user reads why this is happening while they still have time to
 * stop it. Cancel is present for the whole sequence.
 *
 * It counts 5 down to 1 and fires rather than ever showing zero - a visible zero reads as
 * "already happened" while the window is still sitting there.
 */

const SWITCH_SENTENCES = [
  "Switching to the Avalonia build.",
  "Same analysis engine, a different interface over it.",
  "Your loaded archive and verdicts stay where they are.",
  "You can switch back from its View menu.",
];

const SENTENCE_MS = 1300;
const COUNTDOWN_FROM = 5;

interface HandoffTarget {
  preferred: string | null;
  available: boolean;
  path: string | null;
  alreadyRunning: boolean;
}

interface Takeover {
  sentence: number;
  /** Counts COUNTDOWN_FROM down to 1. */
  remaining: number;
  /** Set when the switch was refused; the overlay stays up carrying the reason. */
  error: string | null;
}

export function FrontendSwitch() {
  // Re-probed rather than cached: the sibling can appear or disappear between launches,
  // and "already running" is only true for as long as it is.
  const [target] = createResource(() =>
    invoke<HandoffTarget>("avalonia_target").catch(
      () => ({ preferred: null, available: false, path: null, alreadyRunning: false }),
    ),
  );

  const [takeover, setTakeover] = createSignal<Takeover | null>(null);
  const timers = new Set<ReturnType<typeof setTimeout>>();

  const clearTimers = () => {
    for (const timer of timers) clearTimeout(timer);
    timers.clear();
  };
  onCleanup(clearTimers);

  const cancel = () => {
    clearTimers();
    setTakeover(null);
  };

  const perform = async () => {
    try {
      await invoke("switch_to_avalonia");
      // On success this window is closing, so there is nothing to put back.
    } catch (error) {
      // The handoff rolls its own preference back, so the only thing left to undo is the
      // overlay - which stays up carrying the reason rather than vanishing silently.
      setTakeover((state) =>
        state
          ? {
              ...state,
              error: typeof error === "string" ? error : "Could not open the Avalonia build.",
            }
          : state,
      );
    }
  };


  const start = () => {
    clearTimers();
    setTakeover({ sentence: 0, remaining: COUNTDOWN_FROM, error: null });

    for (let index = 1; index < SWITCH_SENTENCES.length; index += 1) {
      timers.add(
        setTimeout(() => {
          setTakeover((state) => (state ? { ...state, sentence: index } : state));
        }, index * SENTENCE_MS),
      );
    }

    for (let tick = 1; tick < COUNTDOWN_FROM; tick += 1) {
      timers.add(
        setTimeout(() => {
          setTakeover((state) => (state ? { ...state, remaining: COUNTDOWN_FROM - tick } : state));
        }, tick * 1000),
      );
    }

    timers.add(setTimeout(() => void perform(), COUNTDOWN_FROM * 1000));
  };

  return (
    <>
      {/* Offered only in the combined bundle, where the other build is actually present. */}
      <Show when={target()?.available}>
        <Button
          variant="ghost"
          size="sm"
          disabled={takeover() !== null}
          title={
            target()?.alreadyRunning
              ? "The Avalonia build is already running - switch to that window instead."
              : "Close this and reopen in the Avalonia build"
          }
          onClick={start}
        >
          <IconRepeat class="size-3.5" />
          Avalonia UI
        </Button>
      </Show>

      <Show when={takeover()}>
        {(state) => (
          <div class="fixed inset-0 z-50 flex flex-col items-center justify-center gap-6 bg-black/90 px-10">
            <Show when={!state().error}>
              <p class="text-8xl font-semibold leading-none tabular-nums text-white">
                {state().remaining}
              </p>
            </Show>

            <div class="relative h-12 w-full max-w-md">
              <For each={SWITCH_SENTENCES}>
                {(sentence, index) => (
                  <p
                    class="absolute inset-0 flex items-center justify-center text-center text-[15px] text-white transition-opacity duration-500"
                    classList={{
                      "opacity-100": !state().error && index() === state().sentence,
                      "opacity-0": !!state().error || index() !== state().sentence,
                    }}
                  >
                    {sentence}
                  </p>
                )}
              </For>
            </div>

            <Show when={state().error}>
              {(message) => (
                <div role="alert" class="max-w-md text-center text-[13px] text-verdict-rejected">
                  {message()}
                </div>
              )}
            </Show>

            <button
              type="button"
              class="rounded-lg border border-white/25 px-4 py-2 text-[13px] text-white transition-colors hover:bg-white/10"
              onClick={cancel}
            >
              {state().error ? "Close" : "Cancel"}
            </button>
          </div>
        )}
      </Show>
    </>
  );
}
