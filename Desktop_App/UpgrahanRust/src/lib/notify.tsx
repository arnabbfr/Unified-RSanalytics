import toast from "solid-toast";
import type { JSX } from "solid-js";
import IconCheck from "~icons/lucide/check";
import IconTriangleAlert from "~icons/lucide/triangle-alert";
import IconLoaderCircle from "~icons/lucide/loader-circle";
import IconX from "~icons/lucide/x";

/**
 * Toasts that can be dismissed.
 *
 * solid-toast 0.5 exports only `toast` and `Toaster` - there is no ToastBar and no children
 * render prop on the Toaster - so a notification rendered from a plain string has no
 * affordance at all and can only be waited out. That is fine for a 3.5s success and wrong
 * for an error or a long-running progress toast.
 *
 * The render-function form of the API does give us the toast's own id, so every notice here
 * is wrapped with a close control and click-to-dismiss.
 */

type Tone = "success" | "error" | "loading" | "plain";

const TONE_ICON: Record<Tone, (() => JSX.Element) | null> = {
  success: () => <IconCheck class="size-3.5 shrink-0 text-verdict-verified" />,
  error: () => <IconTriangleAlert class="size-3.5 shrink-0 text-verdict-rejected" />,
  loading: () => <IconLoaderCircle class="size-3.5 shrink-0 animate-spin text-ed-accent" />,
  plain: null,
};

function body(tone: Tone, id: string, content: JSX.Element) {
  const icon = TONE_ICON[tone];

  return (
    <div
      class="flex cursor-pointer items-start gap-2"
      title="Click to dismiss"
      onClick={() => toast.dismiss(id)}
    >
      {icon?.()}
      <div class="flex-1 text-[13px] leading-snug text-ed-text-1">{content}</div>
      <button
        type="button"
        aria-label="Dismiss"
        class="-mr-0.5 -mt-0.5 shrink-0 rounded-md p-0.5 text-ed-text-3 transition-colors hover:bg-ed-ctl-hover hover:text-ed-text-1"
        onClick={(e) => {
          e.stopPropagation();
          toast.dismiss(id);
        }}
      >
        <IconX class="size-3" />
      </button>
    </div>
  );
}

interface Options {
  /** Reuse an id to replace an existing toast rather than stacking a second copy. */
  id?: string;
  /** Loading notices stay until dismissed or replaced. */
  duration?: number;
}

const make =
  (tone: Tone) =>
  (content: JSX.Element, options: Options = {}): string =>
    toast.custom((t) => body(tone, t.id, content), {
      duration: options.duration ?? (tone === "loading" ? Infinity : 4000),
      ...(options.id ? { id: options.id } : {}),
    });

export const notify = {
  success: make("success"),
  error: make("error"),
  loading: make("loading"),
  plain: make("plain"),
  dismiss: (id?: string) => toast.dismiss(id),
};

/** Normalises the usual `unknown` from a catch into something worth showing. */
export const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));
