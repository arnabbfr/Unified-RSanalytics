# AGENTS.md — Unified-RSanalytics (UpaGraha / GeoSemanticSat)

Instructions for AI coding agents working in this repository.

## Read this first

**If `Jyotirmoy_SETUP.md` exists in the repo root, read it before running anything.**
It is the machine-specific setup guide for this developer (gitignored, so it will
not exist in a fresh clone) and it holds the exact paths, versions, and daily
commands that are verified to work here. Prefer its commands over anything in
`README.md` or `run.md` when they disagree — those two are written for a different
machine and contain stale absolute paths.

## What this repo is

An air-gapped satellite / Earth-observation intelligence platform, split in two:

| Half | Path | Stack |
| :--- | :--- | :--- |
| Analytics backend | `app/` | Python 3.11, FastAPI, rasterio, torch, faiss, SQLAlchemy |
| Desktop studio | `Desktop_App/Upgrahan2/` | .NET 10, C#, Avalonia 11.3.4, ONNX Runtime |
| Desktop studio (experimental) | `Desktop_App/UpgrahanRust/` | Rust, Tauri 2, SolidJS, Tailwind v4, MapLibre |

`Desktop_App/Upgrahan2/src/` contains `GeoSemanticSat.Core` (change detection,
CUSUM onset, DBSCAN clustering, vector index), `.Engine` (embeddings, retrieval),
`.UI` (Avalonia), `.Cli`, and `.Tests` (xunit).

## Architecture fact that changes how you work

**Neither desktop UI calls the FastAPI backend.** There is no `:8000` reference
anywhere in the C# source. Do not "wire a UI to the API" or assume a running backend
unless explicitly asked — changing that is an architecture decision, not a fix.

The two frontends reach `GeoSemanticSat.Core` differently, and this distinction matters:

| Frontend | How it reaches Core |
| :--- | :--- |
| Avalonia (`GeoSemanticSat.UI`) | In-process, direct calls |
| Tauri (`Desktop_App/UpgrahanRust/`) | Over `GeoSemanticSat.Daemon`, a local HTTP sidecar |

`GeoSemanticSat.Daemon` is **not** the Python API. It is a .NET host that references the
same `Core`/`Engine` projects, so both frontends execute identical analysis code and produce
identical numbers. That is deliberate: it makes an A/B between the two UIs a comparison of
the interface rather than of the algorithms. `DaemonContractTests` enforces it.

The daemon binds `127.0.0.1` on an **ephemeral** port behind a per-process bearer token and
prints `{"ready":true,"port":N,"token":"..."}` as one JSON line on stdout. Do not give it a
fixed port and do not bind anything but loopback.

The current developer's assignment is the **frontend (Avalonia UI)**.

## How to run things

Backend runs in Docker; the desktop UI runs natively on Windows (a GUI cannot
usefully run in a Linux container).

`dev.py` in the repo root is the one-command launcher. It starts the Docker engine if
it is down, waits for the compose healthcheck, applies the schema, seeds the sample
rasters on first run, and only then launches the UI. Standard library only, and it
runs the same on Windows, macOS and Linux from any shell.

```bash
python dev.py                        # backend (healthy) then the desktop UI
python dev.py backend                # backend only
python dev.py frontend               # desktop UI only (Avalonia, the default)
python dev.py frontend --ui=tauri    # desktop UI only (experimental Rust/Tauri)
python dev.py sidecar                # analysis daemon alone, for poking the API with curl
python dev.py stop                   # docker compose down
```

The equivalent manual commands still work unchanged:

```powershell
# Backend — from repo root. Docker Desktop must be running first.
docker compose up -d --wait
docker compose exec api python scripts/init_db.py          # first run only
docker compose exec api python scripts/create_sample_data.py  # first run only
# Swagger: http://127.0.0.1:8000/docs   Health: http://127.0.0.1:8000/health

# Desktop UI
dotnet run --project "Desktop_App\Upgrahan2\src\GeoSemanticSat.UI\GeoSemanticSat.UI.csproj"
```

`app/`, `scripts/` and `tests/` are bind-mounted into the container, so Python edits
hot-reload. Only a `requirements.txt` change requires `docker compose up -d --build`.

Docker Desktop on this machine is at
`C:\Users\jyoti\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe`,
**not** the default `C:\Program Files\Docker\`.

## Tests

```powershell
docker compose exec api pytest -v                                              # backend
dotnet test "Desktop_App\Upgrahan2\src\GeoSemanticSat.Tests\GeoSemanticSat.Tests.csproj"  # desktop
```

## Rules

- **Never commit build output.** `bin/`, `obj/`, `.dotnet/` and Rust `target/` are
  covered by `.gitignore` (untracked in commit `c2cbd20`). Still check `git status`
  before committing and stage only real source changes.
- **Do not add `torch` usage to the backend without flagging it.** It is pinned in
  `requirements.txt` but imported nowhere. The Dockerfile deliberately installs the
  **CPU-only** build from `https://download.pytorch.org/whl/cpu`; the PyPI Linux wheel
  pulls ~6 GB of CUDA that this air-gapped CPU backend never uses.
- **Do not move the SQLite database into the `./data` bind mount**, and do not remove
  `/app/dbdata` from the Dockerfile's `mkdir`/`chown` — the named volume inherits
  `appuser` ownership from that directory. Without it the container fails with
  `unable to open database file`.
- **Do not write the api service's `DATABASE_URL` as `${DATABASE_URL:-...}` in
  `docker-compose.yml`.** Compose interpolates `${DATABASE_URL}` from the project `.env`,
  which holds the host-relative `sqlite:///./data/satintel.db` used when running the API
  natively — so the `:-` default never applied and the database silently landed on the
  bind mount. The container path is a literal, overridable via `API_DATABASE_URL`.
- **Do not remove the `libexpat1` apt layer** from the Dockerfile. rasterio's bundled
  GDAL links against it and `python:3.11-slim` does not ship it.
- **If a .NET restore fails with `NU1100: Unable to resolve ...`**, the machine has no
  NuGet package source configured — it is not a bad version pin. Check
  `dotnet nuget list source` and add nuget.org instead of editing any `.csproj`:
  `dotnet nuget add source https://api.nuget.org/v3/index.json -n nuget.org`
- **`SemanticEmbeddingLayout` is the single source of truth for the 128-dim vector.** Never
  hardcode an embedding index. The vector has an appearance block (image-only, unbounded
  reflectance) and shared semantic axes (bounded, written by both encoders). Text-to-image
  similarity compares the semantic axes only; image-to-image uses the full vector. Writing
  text weights into appearance dimensions is what produced negative match scores.
- **Changing the embedding layout invalidates every persisted `.bin` index.** Rebuild with
  `GeoSemanticSat.Cli index <dir> <out>` after any change to either encoder.
- **Use `QualityMaskEngine.IsUsable(flags)`**, never `flags != QualityMaskFlags.Valid` -
  `QualityMaskFlags` is a `[Flags]` enum and Water/HighHaze are informational, not disqualifying.
- **Use `BoundingBox.AreaSquareMetres()`** for area. Do not reintroduce a GSD-times-cos(lat)
  estimate; ground sampling distance is already in ground metres.
- **Water detection must stay MNDWI-gated.** NDWI alone cannot separate open water from new
  concrete, because both collapse NIR. Removing the MNDWI endpoint check reclassifies every
  construction site as WaterExtentVariation.
- **Never commit `Jyotirmoy_SETUP.md` or `.env`.** Both are gitignored.
- The database is SQLite in the `api_db` named volume, **not** in `./data`.
  `./data` is for GeoTIFF rasters. Do not move the DB into the bind mount —
  SQLite WAL locking is unreliable over Docker Desktop's Windows bind mounts.
- PostGIS is optional and off by default (`--profile postgres`).
  `app/models/entities.py` uses plain SQLAlchemy columns and needs no geometry types.
- Don't add dependencies to `requirements.txt` or a `.csproj` without asking —
  this is an air-gapped-target project and every package has to be vendorable offline.
- Match the existing style: `.axaml` + code-behind in the Avalonia UI (no MVVM framework is
  in use), and plain xunit `[Fact]` tests.
- **The Rust frontend clones cap.so's design language deliberately.** Colours come only from
  the semantic tokens in `src/styles/theme.css` (`--ed-*` surfaces, Radix Colors primitives).
  Never write a raw hex or a Tailwind default palette colour in a component; the one
  exception is reading CSS custom properties for MapLibre paint properties, which cannot
  take classes.
- **`CONTEXT.md` vocabulary is binding in both UIs.** Result vs Candidate vs Verified mean
  three different things; similarity is -1..1 and is never shown at or below zero; the
  evidence score ranks candidates and must never be rendered as "N% likely".
- **`ChangeRecord.Id` is a fresh GUID on every detection run.** Do not persist a selection
  by candidate id across a re-detect — the same physical site gets a new id each pass.
