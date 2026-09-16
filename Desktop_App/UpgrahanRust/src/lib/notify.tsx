import { createSignal, For, Show, onCleanup, type JSX } from "solid-js";
import IconCheck from "~icons/lucide/check";
import IconTriangleAlert from "~icons/lucide/triangle-alert";
import IconLoaderCircle from "~icons/lucide/loader-circle";
import IconX from "~icons/lucide/x";
import { cn } from "~/lib/cn";

/**
 * Notifications, with a Sonner-style collapsed stack.
 *
 * This replaces solid-toast rather than wrapping it. Three of its limits bit in a row:
 * it exports no ToastBar and no children render prop, so a plain-string notice had no
 * dismiss affordance at all; and routing around that through `toast.custom` opts out of
 * the Toaster's own styling, which is what left these rendering with no background.
 * Owning ~150 lines is cheaper than fighting that, and the stacking behaviour below is
 * not something it offers in any form.
 *
 * The interaction is Sonner's (emilkowalski/sonner), which is the one most component
 * libraries have since copied:
 *   - newest sits at the front, nearest the corner;
 *   - older ones peek out behind it, nudged up and scaled down so the stack reads as
 *     depth rather than as a list;
 *   - hovering the stack expands it into a real list, so a burst of notices can be read
 *     without any of them being dismissed first;
 *   - dismiss timers pause while hovered, because a notice you are reading should not
 *     disappear out from under you;
 *   - beyond MAX_VISIBLE, the rest are collapsed into a count rather than tiling.
 *
 * Reusing an id replaces that notice in place instead of stacking a duplicate. The help
 * text uses this: it is the same static content every time, so a second press should
 * refresh it, not pile up another copy.
 */

type Tone = "success" | "error" | "loading" | "plain";

interface Notice {
  id: string;
  tone: Tone;
  content: JSX.Element;
  /** Infinity for notices that stay until replaced or dismissed. */
  duration: number;
  /** Set when the exit transition is running, so it animates out rather than vanishing. */
  leaving: boolean;
}

const MAX_VISIBLE = 3;
/** How far each toast behind the front one peeks out, collapsed. */
const PEEK_PX = 12;
/** Gap between toasts once the stack is expanded. */
const GAP_PX = 10;
const EXIT_MS = 180;

const [notices, setNotices] = createSignal<Notice[]>([]);

/** Live dismiss timers, so hovering can pause and resume them. */
const timers = new Map<string, { remaining: number; startedAt: number; handle?: number }>();

function clearTimer(id: string) {
  const timer = timers.get(id);
  if (timer?.handle !== undefined) clearTimeout(timer.handle);
  timers.delete(id);
}

function armTimer(id: string, remaining: number) {
  if (!Number.isFinite(remaining)) return;

  const handle = window.setTimeout(() => dismiss(id), remaining);
  timers.set(id, { remaining, startedAt: Date.now(), handle });
}

function pauseTimers() {
  for (const [id, timer] of timers) {
    if (timer.handle === undefined) continue;
    clearTimeout(timer.handle);
    timers.set(id, {
      remaining: Math.max(0, timer.remaining - (Date.now() - timer.startedAt)),
      startedAt: Date.now(),
      handle: undefined,
    });
  }
}

function resumeTimers() {
  for (const [id, timer] of [...timers]) {
    if (timer.handle !== undefined) continue;
    armTimer(id, timer.remaining);
  }
}

export function dismiss(id?: string) {
  if (id === undefined) {
    for (const notice of notices()) dismiss(notice.id);
    return;
  }

  clearTimer(id);
  setNotices((list) => list.map((n) => (n.id === id ? { ...n, leaving: true } : n)));
  window.setTimeout(() => setNotices((list) => list.filter((n) => n.id !== id)), EXIT_MS);
}

interface Options {
  /** Reuse an id to replace a notice in place rather than stacking a duplicate. */
  id?: string;
  duration?: number;
}

function push(tone: Tone, content: JSX.Element, options: Options = {}): string {
  const id = options.id ?? `n${Date.now()}${Math.random().toString(36).slice(2, 7)}`;
  const duration = options.duration ?? (tone === "loading" ? Infinity : 4000);

  clearTimer(id);
  setNotices((list) => {
    const replaced: Notice = { id, tone, content, duration, leaving: false };
    const existing = list.findIndex((n) => n.id === id);

    // Replacing keeps its position in the stack; a new notice goes to the front.
    if (existing >= 0) return list.map((n, i) => (i === existing ? replaced : n));
    return [replaced, ...list];
  });

  armTimer(id, duration);
  return id;
}

export const notify = {
  success: (content: JSX.Element, options?: Options) => push("success", content, options),
  error: (content: JSX.Element, options?: Options) => push("error", content, options),
  loading: (content: JSX.Element, options?: Options) => push("loading", content, options),
  plain: (content: JSX.Element, options?: Options) => push("plain", content, options),
  dismiss,
};

/** Normalises the `unknown` from a catch into something worth showing. */
export const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));

const TONE_ICON: Record<Tone, (() => JSX.Element) | null> = {
  success: () => <IconCheck class="size-3.5 shrink-0 text-verdict-verified" />,
  error: () => <IconTriangleAlert class="size-3.5 shrink-0 text-verdict-rejected" />,
  loading: () => <IconLoaderCircle class="size-3.5 shrink-0 animate-spin text-ed-accent" />,
  plain: null,
};

/** Mount once, near the root. Renders the stack. */
export function Notifications() {
  const [expanded, setExpanded] = createSignal(false);
  /** Measured heights, so the expanded layout can be done entirely in transforms. */
  const [heights, setHeights] = createSignal<Record<string, number>>({});

  const visible = () => notices().slice(0, MAX_VISIBLE);
  const hiddenCount = () => Math.max(0, notices().length - MAX_VISIBLE);

  const collapse = () => {
    setExpanded(false);
    resumeTimers();
  };

  onCleanup(() => {
    for (const id of [...timers.keys()]) clearTimer(id);
  });

  /**
   * How far up from the corner this notice sits.
   * Collapsed it is a fixed peek per layer; expanded it clears everything in front of it.
   */
  const offsetFor = (index: number) => {
    if (!expanded()) return index * PEEK_PX;

    const map = heights();
    return visible()
      .slice(0, index)
      .reduce((total, n) => total + (map[n.id] ?? 44) + GAP_PX, 0);
  };

  return (
    <div
      class="pointer-events-none fixed bottom-4 right-4 z-[60] w-[min(26rem,calc(100vw-2rem))]"
      onMouseEnter={() => {
        setExpanded(true);
        pauseTimers();
      }}
      onMouseLeave={collapse}
    >
      {/* Anchored at the bottom; each notice is positioned by transform from there. */}
      <div class="relative">
        <For each={visible()}>
          {(notice, index) => (
            <div
              ref={(el) => {
                // Measure after paint so the expanded offsets are real heights, not guesses.
                requestAnimationFrame(() =>
                  setHeights((h) => ({ ...h, [notice.id]: el.offsetHeight })),
                );
              }}
              class={cn(
                "pointer-events-auto absolute bottom-0 right-0 w-full origin-bottom",
                "rounded-xl border border-ed-line bg-ed-card p-3 shadow-ed-pop",
                "transition-all duration-200 ease-out",
                notice.leaving && "pointer-events-none opacity-0",
              )}
              style={{
                transform: `translateY(-${offsetFor(index())}px) scale(${
                  expanded() ? 1 : 1 - index() * 0.04
                })`,
                // Behind-the-front notices dim only while collapsed; expanded they are all
                // meant to be read.
                opacity: notice.leaving ? 0 : expanded() || index() === 0 ? 1 : 0.85,
                "z-index": MAX_VISIBLE - index(),
              }}
            >
              <div class="flex items-start gap-2">
                {TONE_ICON[notice.tone]?.()}
                <div class="min-w-0 flex-1 text-[13px] leading-snug text-ed-text-1">
                  {notice.content}
                </div>
                <button
                  type="button"
                  aria-label="Dismiss"
                  class="-mr-0.5 -mt-0.5 shrink-0 rounded-md p-0.5 text-ed-text-3 transition-colors hover:bg-ed-ctl-hover hover:text-ed-text-1"
                  onClick={() => dismiss(notice.id)}
                >
                  <IconX class="size-3" />
                </button>
              </div>
            </div>
          )}
        </For>

        <Show when={hiddenCount() > 0 && !expanded()}>
          <div
            class="pointer-events-none absolute bottom-0 right-0 w-full text-right text-[10px] text-ed-text-3"
            style={{ transform: `translateY(-${MAX_VISIBLE * PEEK_PX + 22}px)` }}
          >
            +{hiddenCount()} more
          </div>
        </Show>
      </div>
    </div>
  );
}
