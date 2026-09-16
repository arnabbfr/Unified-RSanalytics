import { cva, type VariantProps } from "cva";
import { splitProps, type ComponentProps } from "solid-js";

/**
 * Ported from Cap's packages/ui-solid/src/Button.tsx.
 *
 * The Cap signature is all here: pill base, the 1.5px inset top-edge bevel that makes solid
 * fills read as very slightly raised, a scale press rather than a colour flash, and one
 * fixed list of transitioned properties at 200ms.
 *
 * `data-variant`/`data-size` are exposed on the DOM node because Cap uses them as an
 * override hook for higher-specificity theme layers.
 */
const styles = cva(
  "outline-offset-2 flex justify-center items-center gap-1.5 focus-visible:outline-solid rounded-full transition-[background-color,border-color,color,box-shadow,filter,opacity,transform] active:scale-[0.98] will-change-transform duration-200 disabled:cursor-not-allowed disabled:active:scale-100",
  {
    defaultVariants: { variant: "gray", size: "md" },
    variants: {
      variant: {
        primary:
          "bg-gray-12 dark-button-shadow text-gray-1 hover:bg-gray-11 disabled:bg-gray-6 disabled:text-gray-9",
        blue:
          "bg-ed-accent text-white border border-ed-accent shadow-[0_1.50px_0_0_rgba(255,255,255,0.20)_inset] hover:bg-ed-accent-2 disabled:bg-gray-6 disabled:text-gray-9",
        destructive:
          "bg-red-9 text-white hover:bg-red-10 disabled:bg-gray-6 disabled:text-gray-9",
        outline:
          "border border-ed-line-strong text-ed-text-1 hover:bg-ed-ctl-hover disabled:text-gray-9",
        gray:
          "bg-ed-ctl hover:bg-ed-ctl-hover data-[selected=true]:bg-ed-ctl-active border border-ed-line gray-button-shadow text-ed-text-1 disabled:text-ed-text-3 disabled:bg-ed-ctl",
        ghost:
          "text-ed-text-2 hover:bg-ed-ctl-hover hover:text-ed-text-1 disabled:text-ed-text-3",
        /* Reserved for the single primary action on a screen, as Cap reserves radialblue. */
        verify:
          "text-white border-0 [background:radial-gradient(90%_100%_at_15%_12%,var(--ed-verify-from)_0%,var(--ed-verify-to)_100%)] shadow-[0_0_0_1px] shadow-emerald-9 hover:opacity-90 disabled:opacity-50",
      },
      size: {
        xs: "text-[11px] px-2 h-5",
        sm: "text-[11px] px-2.5 h-7",
        md: "text-[13px] px-3 h-8",
        lg: "text-[13px] px-4 h-9",
        icon: "size-8 px-0",
      },
    },
  },
);

export type ButtonProps = ComponentProps<"button"> & VariantProps<typeof styles>;

export function Button(props: ButtonProps) {
  const [local, rest] = splitProps(props, ["variant", "size", "class"]);

  return (
    <button
      type="button"
      data-variant={local.variant ?? "gray"}
      data-size={local.size ?? "md"}
      class={styles({ variant: local.variant, size: local.size, class: local.class })}
      {...rest}
    />
  );
}
