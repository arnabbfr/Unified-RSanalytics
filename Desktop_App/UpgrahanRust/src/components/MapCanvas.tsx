import { invoke } from "@tauri-apps/api/core";
import * as maplibregl from "maplibre-gl";
import type {
  ExpressionSpecification,
  GeoJSONSource,
  Map as MapLibreMap,
  StyleSpecification,
} from "maplibre-gl";
import { createEffect, createSignal, For, onCleanup, onMount, Show } from "solid-js";
import type { Feature, FeatureCollection, Point, Polygon } from "geojson";
import IconPlus from "~icons/lucide/plus";
import IconMinus from "~icons/lucide/minus";
import IconMaximize2 from "~icons/lucide/maximize-2";
import { Button } from "~/components/Button";
import { Select } from "~/components/Select";
import { cn } from "~/lib/cn";
import type { Bbox, ChangeType, Coord } from "~/lib/daemon";

/**
 * Basemap tiles, air-gap aware.
 *
 * Online: the raster source points straight at the provider's https template.
 * "Offline only": the raster source points at a `gsscache://` URL instead, and this custom
 * protocol answers every tile request from the shared on-disk cache via the Rust
 * `read_cached_tile` command - never the network. A cache miss resolves to a transparent 1x1
 * PNG so the map shows an empty tile instead of a broken-image glyph.
 */
const PROTOCOL = "gsscache";

// 1x1 transparent PNG, used to answer offline cache misses without a broken-tile flash.
const TRANSPARENT_PNG_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

function transparentPngBytes(): Uint8Array {
  const binary = atob(TRANSPARENT_PNG_B64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

interface TileProvider {
  id: string;
  label: string;
  urlTemplate: string;
  attribution: string;
  regionLimited: boolean;
}

const DEFAULT_PROVIDER_ID = "EsriSatellite";

function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/** Reads the five ChangeType hues (plus a neutral fallback) from the theme, for marker fill. */
function typeColorExpression(): ExpressionSpecification {
  return [
    "match",
    ["get", "type"],
    "Construction",
    cssVar("--type-construction", "#c2410c"),
    "Clearance",
    cssVar("--type-clearance", "#a16207"),
    "WaterExtentVariation",
    cssVar("--type-water", "#0369a1"),
    "RoadDevelopment",
    cssVar("--type-road", "#6d28d9"),
    "ActivityConcentration",
    cssVar("--type-activity", "#a16207"),
    cssVar("--type-nochange", "#8a8a92"),
  ] as unknown as ExpressionSpecification;
}

function emptyFeatureCollection<G extends Point | Polygon>(): FeatureCollection<G> {
  return { type: "FeatureCollection", features: [] };
}

/**
 * Approximates a circle of `radiusKm` around `centre` as a GeoJSON polygon. Longitude degrees
 * shrink toward the poles, so the east-west radius is corrected by cos(latitude).
 */
function circleFeature(centre: Coord, radiusKm: number, steps = 64): Feature<Polygon> {
  const kmPerDegLat = 110.574;
  const kmPerDegLon = 111.32 * Math.cos((centre.latitude * Math.PI) / 180);
  const degLat = radiusKm / kmPerDegLat;
  const degLon = radiusKm / Math.max(kmPerDegLon, 0.0001);

  const ring: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * 2 * Math.PI;
    ring.push([centre.longitude + degLon * Math.sin(angle), centre.latitude + degLat * Math.cos(angle)]);
  }

  return {
    type: "Feature",
    properties: {},
    geometry: { type: "Polygon", coordinates: [ring] },
  };
}

export interface MapCanvasMarker {
  id: string;
  lat: number;
  lon: number;
  type?: ChangeType;
  selected?: boolean;
}

export interface MapCanvasProps {
  markers?: MapCanvasMarker[];
  bounds?: Bbox;
  onMarkerClick?: (id: string) => void;
  class?: string;
  radiusKm?: number;
  centre?: Coord;
}

const BASEMAP_LAYER = "basemap";
const BASEMAP_SOURCE = "basemap";
const RADIUS_SOURCE = "search-radius";
const RADIUS_FILL_LAYER = "search-radius-fill";
const RADIUS_LINE_LAYER = "search-radius-line";
const MARKERS_SOURCE = "markers";
const MARKERS_LAYER = "markers-circle";

export function MapCanvas(props: MapCanvasProps) {
  let containerRef: HTMLDivElement | undefined;
  let map: MapLibreMap | undefined;

  const [loaded, setLoaded] = createSignal(false);
  const [providers, setProviders] = createSignal<TileProvider[]>([]);
  const [providerId, setProviderId] = createSignal(DEFAULT_PROVIDER_ID);
  const [offlineOnly, setOfflineOnly] = createSignal(false);

  const activeProvider = () =>
    providers().find((p) => p.id === providerId()) ?? providers()[0];

  onMount(() => {
    maplibregl.addProtocol(PROTOCOL, async (params) => {
      // URL shape: gsscache://{provider}/{z}/{x}/{y}
      const rest = params.url.slice(`${PROTOCOL}://`.length);
      const [provider, z, x, y] = rest.split("/");
      if (provider && z && x && y) {
        try {
          const bytes = await invoke<number[] | null>("read_cached_tile", {
            provider,
            z: Number(z),
            x: Number(x),
            y: Number(y),
          });
          if (bytes) return { data: new Uint8Array(bytes).buffer };
        } catch {
          // Falls through to the transparent placeholder below.
        }
      }
      return { data: transparentPngBytes().buffer };
    });

    if (!containerRef) return;

    const emptyStyle: StyleSpecification = { version: 8, sources: {}, layers: [] };
    const instance = new maplibregl.Map({
      container: containerRef,
      style: emptyStyle,
      center: [77.2, 28.61],
      zoom: 4,
      attributionControl: false,
    });
    map = instance;

    instance.on("load", () => {
      instance.addSource(RADIUS_SOURCE, { type: "geojson", data: emptyFeatureCollection<Polygon>() });
      instance.addLayer({
        id: RADIUS_FILL_LAYER,
        type: "fill",
        source: RADIUS_SOURCE,
        paint: { "fill-color": cssVar("--ed-accent", "#0a84ff"), "fill-opacity": 0.12 },
      });
      instance.addLayer({
        id: RADIUS_LINE_LAYER,
        type: "line",
        source: RADIUS_SOURCE,
        paint: {
          "line-color": cssVar("--ed-accent", "#0a84ff"),
          "line-width": 1.5,
          "line-dasharray": [2, 2],
        },
      });

      instance.addSource(MARKERS_SOURCE, { type: "geojson", data: emptyFeatureCollection<Point>() });
      instance.addLayer({
        id: MARKERS_LAYER,
        type: "circle",
        source: MARKERS_SOURCE,
        paint: {
          "circle-color": typeColorExpression(),
          "circle-radius": ["case", ["==", ["get", "selected"], true], 9, 5] as unknown as ExpressionSpecification,
          "circle-stroke-width": ["case", ["==", ["get", "selected"], true], 2, 1] as unknown as ExpressionSpecification,
          "circle-stroke-color": [
            "case",
            ["==", ["get", "selected"], true],
            "#ffffff",
            "rgba(0,0,0,0.35)",
          ] as unknown as ExpressionSpecification,
        },
      });

      instance.on("click", MARKERS_LAYER, (e) => {
        const id = e.features?.[0]?.properties?.id;
        if (typeof id === "string") props.onMarkerClick?.(id);
      });
      instance.on("mouseenter", MARKERS_LAYER, () => {
        instance.getCanvas().style.cursor = "pointer";
      });
      instance.on("mouseleave", MARKERS_LAYER, () => {
        instance.getCanvas().style.cursor = "";
      });

      setLoaded(true);
    });

    invoke<TileProvider[]>("tile_providers")
      .then((list) => {
        setProviders(list);
        if (!list.some((p) => p.id === providerId())) {
          setProviderId(list[0]?.id ?? DEFAULT_PROVIDER_ID);
        }
      })
      .catch(() => setProviders([]));
  });

  onCleanup(() => {
    map?.remove();
    maplibregl.removeProtocol(PROTOCOL);
  });

  // Basemap source/layer: rebuilt (not the whole map) whenever the provider or the
  // offline-only toggle changes, kept beneath the overlay layers added at load.
  createEffect(() => {
    if (!map || !loaded()) return;
    const provider = activeProvider();
    if (!provider) return;
    const offline = offlineOnly();

    const tileUrl = offline ? `${PROTOCOL}://${provider.id}/{z}/{x}/{y}` : provider.urlTemplate;

    if (map.getLayer(BASEMAP_LAYER)) map.removeLayer(BASEMAP_LAYER);
    if (map.getSource(BASEMAP_SOURCE)) map.removeSource(BASEMAP_SOURCE);

    map.addSource(BASEMAP_SOURCE, {
      type: "raster",
      tiles: [tileUrl],
      tileSize: 256,
      attribution: provider.attribution,
    });
    map.addLayer({ id: BASEMAP_LAYER, type: "raster", source: BASEMAP_SOURCE }, RADIUS_FILL_LAYER);
  });

  // Markers: a GeoJSON circle layer, not DOM markers, since there can be many.
  createEffect(() => {
    if (!map || !loaded()) return;
    const source = map.getSource(MARKERS_SOURCE) as GeoJSONSource | undefined;
    if (!source) return;

    const fc: FeatureCollection<Point> = {
      type: "FeatureCollection",
      features: (props.markers ?? []).map((m) => ({
        type: "Feature",
        properties: { id: m.id, type: m.type ?? "NoChange", selected: m.selected === true },
        geometry: { type: "Point", coordinates: [m.lon, m.lat] },
      })),
    };
    source.setData(fc);
  });

  // Search-radius overlay: a translucent circle polygon around centre, sized by radiusKm.
  createEffect(() => {
    if (!map || !loaded()) return;
    const source = map.getSource(RADIUS_SOURCE) as GeoJSONSource | undefined;
    if (!source) return;

    const centre = props.centre;
    const radiusKm = props.radiusKm;
    if (!centre || !radiusKm || radiusKm <= 0) {
      source.setData(emptyFeatureCollection<Polygon>());
      return;
    }
    source.setData({ type: "FeatureCollection", features: [circleFeature(centre, radiusKm)] });
  });

  // Bounds: fit the view, never re-create the map.
  createEffect(() => {
    if (!map || !loaded()) return;
    const b = props.bounds;
    if (!b) return;
    map.fitBounds(
      [
        [b.minLon, b.minLat],
        [b.maxLon, b.maxLat],
      ],
      { padding: 56, maxZoom: 16, duration: 500 },
    );
  });

  function fitToData() {
    if (!map) return;
    const markers = props.markers ?? [];
    if (markers.length === 0) {
      const b = props.bounds;
      if (b) {
        map.fitBounds(
          [
            [b.minLon, b.minLat],
            [b.maxLon, b.maxLat],
          ],
          { padding: 56, duration: 400 },
        );
      }
      return;
    }

    let minLon = Infinity;
    let minLat = Infinity;
    let maxLon = -Infinity;
    let maxLat = -Infinity;
    for (const m of markers) {
      minLon = Math.min(minLon, m.lon);
      maxLon = Math.max(maxLon, m.lon);
      minLat = Math.min(minLat, m.lat);
      maxLat = Math.max(maxLat, m.lat);
    }
    map.fitBounds(
      [
        [minLon, minLat],
        [maxLon, maxLat],
      ],
      { padding: 64, maxZoom: 15, duration: 400 },
    );
  }

  return (
    <div class={cn("relative h-full w-full overflow-hidden", props.class)}>
      <div ref={containerRef} class="h-full w-full" />

      {/* Basemap + offline controls. */}
      <div class="absolute left-3 top-3 flex flex-col gap-1.5 rounded-xl border border-ed-line bg-ed-card/95 p-2 shadow-ed-pop backdrop-blur">
        <Select
          class="w-44"
          value={providerId()}
          title="Basemap provider"
          options={providers().map((p) => ({
            value: p.id,
            label: p.label,
            hint: p.regionLimited ? "Limited coverage" : undefined,
          }))}
          onChange={setProviderId}
        />
        <Button
          variant={offlineOnly() ? "blue" : "gray"}
          size="sm"
          onClick={() => setOfflineOnly((v) => !v)}
          title="Serve tiles only from the local cache - no network requests."
        >
          Offline only
        </Button>
      </div>

      {/* Zoom / fit controls. */}
      <div class="absolute right-3 top-3 flex flex-col gap-1.5">
        <Button variant="gray" size="icon" title="Zoom in" onClick={() => map?.zoomIn()}>
          <IconPlus class="size-4" />
        </Button>
        <Button variant="gray" size="icon" title="Zoom out" onClick={() => map?.zoomOut()}>
          <IconMinus class="size-4" />
        </Button>
        <Button variant="gray" size="icon" title="Fit to data" onClick={fitToData}>
          <IconMaximize2 class="size-4" />
        </Button>
      </div>

      <Show when={activeProvider()}>
        {(provider) => (
          <p class="absolute bottom-1.5 left-2 text-[10px] text-ed-text-3">
            {provider().attribution}
            {offlineOnly() ? " · offline" : ""}
          </p>
        )}
      </Show>
    </div>
  );
}
