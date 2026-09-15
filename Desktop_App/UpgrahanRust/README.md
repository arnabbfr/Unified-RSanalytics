# UpgrahanRust — experimental Rust/Tauri frontend

A second desktop frontend for UpaGraha / GeoSemanticSat, running **alongside** the Avalonia
app rather than replacing it. Built to test whether a web component ecosystem gives better
UI/UX headroom than Avalonia for this workload.

Design language is a deliberate clone of [cap.so](https://cap.so)'s desktop app.

## How it reaches the analysis code

It does **not** reimplement any analysis, and it does **not** call the Python FastAPI backend.

```
Avalonia UI  ──in-process──>  GeoSemanticSat.Core
Tauri UI     ──local HTTP──>  GeoSemanticSat.Daemon  ──>  GeoSemanticSat.Core
```

Both frontends execute the same compiled `Core`, so their numbers match by construction and
an A/B between them compares the *interface*, not the algorithms. `DaemonContractTests`
enforces that the daemon forwards results unchanged.

The daemon binds `127.0.0.1` on an ephemeral port behind a per-process bearer token, and
prints `{"ready":true,"port":N,"token":"..."}` as one JSON line on stdout. `src-tauri/src/daemon.rs`
spawns it, reads that line, and kills the child when the window closes.

## Running it

```bash
python dev.py frontend --ui=tauri     # from the repo root; builds the daemon first
```

Or directly:

```bash
dotnet build ../Upgrahan2/src/GeoSemanticSat.Daemon/GeoSemanticSat.Daemon.csproj -c Release
npm install
npm run desktop
```

## Stack

| Piece | Choice | Why |
| :--- | :--- | :--- |
| Shell | Tauri 2 | Native window, WebView2/WKWebView/WebKitGTK, small bundle |
| UI | SolidJS | What Cap uses, so its components port directly |
| Styling | Tailwind v4 (CSS-first `@theme`) | Cap's approach; tokens are CSS custom properties |
| Primitives | Kobalte | Solid's headless equivalent of Radix |
| Variants | `cva` | Same as Cap's `Button` |
| Colours | Radix Colors v3 + Cap's `--ed-*` set | See `src/styles/theme.css` |
| Font | Geist Sans 400/500/700, body 500 | Cap's exact configuration |
| Map | MapLibre GL JS | Replaces ~1,400 LOC of hand-rolled tile/Mercator code |

## The map and the air gap

MapLibre has no offline mode, and this project targets air-gapped deployment. `src-tauri/src/tiles.rs`
reads and writes `%LOCALAPPDATA%\GeoSemanticSat\MapTileCache\{provider}\{z}\{x}\{y}.png` — the
**same** directory `HybridTileService` uses, so tiles precached by either app serve both. The
frontend registers a `gsscache://` MapLibre protocol that reads from it; with "Offline only"
on, no request ever leaves the machine.

## Vocabulary

`CONTEXT.md` in the repo root is binding here. In short:

- A search returns **Results** (similarity −1..1, never a confidence, never shown at or below zero).
- Change detection produces **Candidates** (evidence score 0..1, explicitly *not* a probability —
  never render it as "N% likely").
- Only a human decision makes something **Verified** or **Rejected**.

`ChangeRecord.id` is a fresh GUID on every detection run, so never persist a selection by id
across a re-detect.

## Layout

```
src/
  app.tsx                  window shell, custom titlebar, stage rail
  lib/daemon.ts            typed client, mirrors Contracts.cs
  lib/store.tsx            thin frontend state (selection, view mode, stage gating)
  styles/theme.css         token values (Radix primitives + Cap's --ed-* set)
  styles/main.css          Tailwind v4 @theme mapping
  components/              Button (ported from Cap), shared surfaces, MapCanvas
  routes/Stage*.tsx        the five workflow stages
src-tauri/
  src/daemon.rs            spawns and supervises gss-daemon
  src/tiles.rs             shared offline tile cache
  src/main.rs              commands, window lifecycle, relaunch-into-Avalonia
```
