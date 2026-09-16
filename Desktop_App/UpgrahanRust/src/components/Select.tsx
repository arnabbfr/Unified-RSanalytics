import { Select as KSelect } from "@kobalte/core/select";
import { Show } from "solid-js";
import IconChevronDown from "~icons/lucide/chevron-down";
import IconCheck from "~icons/lucide/check";
import { cn } from "~/lib/cn";

/**
 * Dropdown built on Kobalte, the way Cap builds its own.
 *
 * A native <select> was here first, and its popup is drawn by the OS rather than the page:
 * no CSS of ours could reach inside it, so the options rendered light-on-light and read as
 * disabled. color-scheme fixes the colour but still leaves an OS-styled list that shares
 * nothing with the rest of the app. This renders the list in the DOM instead, so it takes
 * the same surface, hairline and radius tokens as every other panel.
 *
 * Native date and number inputs elsewhere are still OS-drawn; those are handled by the
 * color-scheme declaration in main.css, which remains the right tool for them.
 */

export interface SelectOption<T extends string> {
  value: T;
  label: string;
  /** Shown under the label in the list, for coverage notes and the like. */
  hint?: string;
  disabled?: boolean;
}

export function Select<T extends string>(props: {
  value: T;
  options: readonly SelectOption<T>[];
  onChange: (value: T) => void;
  title?: string;
  placeholder?: string;
  class?: string;
}) {
  const selected = () => props.options.find((o) => o.value === props.value) ?? null;

  return (
    <KSelect<SelectOption<T>>
      options={[...props.options]}
      optionValue="value"
      optionTextValue="label"
      optionDisabled="disabled"
      value={selected()}
      // Kobalte emits null when a selection is cleared; this control is never clearable,
      // so ignore that rather than pushing an empty value into the store.
      onChange={(option) => option && props.onChange(option.value)}
      placeholder={props.placeholder ?? "Select…"}
      itemComponent={(itemProps) => (
        <KSelect.Item
          item={itemProps.item}
          class={cn(
            "flex cursor-default select-none items-center justify-between gap-3 rounded-md px-2 py-1.5 text-[12px] outline-none",
            "text-ed-text-1 transition-colors duration-100",
            "data-[highlighted]:bg-ed-ctl-hover data-[disabled]:opacity-40",
          )}
        >
          <div class="flex flex-col">
            <KSelect.ItemLabel>{itemProps.item.rawValue.label}</KSelect.ItemLabel>
            <Show when={itemProps.item.rawValue.hint}>
              <span class="text-[10px] text-ed-text-3">{itemProps.item.rawValue.hint}</span>
            </Show>
          </div>
          <KSelect.ItemIndicator>
            <IconCheck class="size-3 text-ed-accent" />
          </KSelect.ItemIndicator>
        </KSelect.Item>
      )}
    >
      <KSelect.Trigger
        title={props.title}
        class={cn(
          "flex h-8 items-center justify-between gap-2 rounded-lg border border-ed-line bg-ed-ctl px-2.5",
          "text-[13px] text-ed-text-1 outline-none transition-colors duration-200",
          "hover:bg-ed-ctl-hover focus-visible:border-ed-accent focus-visible:ring-2 focus-visible:ring-ed-accent/30",
          props.class,
        )}
      >
        <KSelect.Value<SelectOption<T>> class="truncate">
          {(valueState) => valueState.selectedOption()?.label}
        </KSelect.Value>
        <KSelect.Icon>
          <IconChevronDown class="size-3.5 shrink-0 text-ed-text-3" />
        </KSelect.Icon>
      </KSelect.Trigger>

      <KSelect.Portal>
        <KSelect.Content
          class={cn(
            "z-50 min-w-(--kb-popper-anchor-width) overflow-hidden rounded-lg border border-ed-line",
            "bg-ed-card p-1 shadow-ed-pop",
            "animate-in fade-in slide-in-from-top-1 duration-100",
          )}
        >
          <KSelect.Listbox class="max-h-72 overflow-y-auto outline-none" />
        </KSelect.Content>
      </KSelect.Portal>
    </KSelect>
  );
}
