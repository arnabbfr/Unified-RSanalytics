//! Basemap tile access for MapLibre, backed by the Avalonia app's existing disk cache.
//!
//! MapLibre GL JS replaces ~1,400 lines of hand-written tile fetching, LRU caching, Mercator
//! maths and pin hit-testing. The one thing it has no equivalent for is this project's
//! air-gap requirement: OfflineStrict must render without ever touching the network.
//!
//! So this module keeps only the part MapLibre cannot do - reading and writing
//! `%LOCALAPPDATA%\GeoSemanticSat\MapTileCache\{provider}\{z}\{x}\{y}.png`, the SAME
//! directory HybridTileService uses. Tiles precached by either app are visible to the other.

use std::fs;
use std::path::PathBuf;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TileProvider {
    pub id: &'static str,
    pub label: &'static str,
    pub url_template: &'static str,
    pub attribution: &'static str,
    /// Cache file extension. HybridTileService keys this off the provider, so it has to
    /// agree per provider or the two apps write past each other.
    pub extension: &'static str,
    /// True when the provider only covers part of the globe, so the UI can say so
    /// instead of silently showing blank tiles outside coverage.
    pub region_limited: bool,
}

/// Mirrors the providers registered in HybridTileService so both frontends offer the same
/// basemaps and share cache directories. Default is ESRI satellite imagery.
pub const PROVIDERS: &[TileProvider] = &[
    TileProvider {
        id: "esri-satellite",
        label: "ESRI World Imagery (Satellite)",
        url_template: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution: "Esri, Maxar, Earthstar Geographics",
        extension: "png",
        region_limited: false,
    },
    TileProvider {
        id: "carto-dark",
        label: "CartoDB Dark Matter (Tactical)",
        url_template: "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attribution: "(C) OpenStreetMap contributors, (C) CARTO",
        extension: "png",
        region_limited: false,
    },
    TileProvider {
        id: "sentinel2-cloudless",
        label: "Sentinel-2 Cloudless 2024 (EOX 10m)",
        url_template: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg",
        attribution: "EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2024). CC BY-NC-SA 4.0, non-commercial use only",
        // EOX serves JPEG, and HybridTileService caches it as .jpg. Writing .png here would
        // put our copy in a file the Avalonia app never looks for.
        extension: "jpg",
        region_limited: false,
    },
    TileProvider {
        id: "osm-standard",
        label: "OpenStreetMap Standard",
        url_template: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution: "(C) OpenStreetMap contributors",
        extension: "png",
        region_limited: false,
    },
    TileProvider {
        id: "usgs-imagery",
        label: "USGS Imagery (United States only)",
        url_template: "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
        attribution: "USGS The National Map / US Department of the Interior",
        extension: "png",
        region_limited: true,
    },
];

fn cache_root() -> Option<PathBuf> {
    // Same location HybridTileService uses (Environment.SpecialFolder.LocalApplicationData).
    dirs_local_data().map(|d| d.join("GeoSemanticSat").join("MapTileCache"))
}

#[cfg(windows)]
fn dirs_local_data() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
}

#[cfg(target_os = "macos")]
fn dirs_local_data() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn dirs_local_data() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
}

/// The single gate on an untrusted provider id.
///
/// An allow-list rather than character filtering: `provider` arrives from the frontend, and
/// every caller here joins it onto a path. Matching the registry means a traversal attempt
/// simply is not a provider, and it hands back the extension at the same time - the two
/// things that have to agree with HybridTileService.
fn provider_dir(provider: &str) -> Option<(PathBuf, &'static str)> {
    let known = PROVIDERS.iter().find(|p| p.id == provider)?;
    Some((cache_root()?.join(known.id), known.extension))
}

fn tile_path(provider: &str, z: u32, x: u32, y: u32) -> Option<PathBuf> {
    let (dir, extension) = provider_dir(provider)?;
    Some(
        dir.join(z.to_string())
            .join(x.to_string())
            .join(format!("{y}.{extension}")),
    )
}

/// Reads a tile from the shared disk cache. Returns None on a miss.
pub fn read_cached(provider: &str, z: u32, x: u32, y: u32) -> Option<Vec<u8>> {
    let path = tile_path(provider, z, x, y)?;
    fs::read(path).ok()
}

/// Writes a fetched tile into the shared cache so the Avalonia app benefits too.
pub fn write_cached(provider: &str, z: u32, x: u32, y: u32, bytes: &[u8]) -> Result<(), String> {
    let path = tile_path(provider, z, x, y).ok_or("Invalid tile coordinates or provider.")?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Could not create the tile cache directory: {e}"))?;
    }

    fs::write(&path, bytes).map_err(|e| format!("Could not write the cached tile: {e}"))
}

/// Counts cached tiles for a provider, so the UI can show what is available offline.
pub fn cached_tile_count(provider: &str) -> usize {
    // cached_tile_count used to join `provider` straight onto the cache root with no
    // validation, so this exposed command could be walked out of the cache directory.
    let Some((root, extension)) = provider_dir(provider) else {
        return 0;
    };

    fn walk(dir: &PathBuf, extension: &str) -> usize {
        let Ok(entries) = fs::read_dir(dir) else {
            return 0;
        };
        entries
            .filter_map(Result::ok)
            .map(|e| {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, extension)
                } else if path.extension().is_some_and(|x| x == extension) {
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    walk(&root, extension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_path_traversal_in_provider() {
        assert!(tile_path("../../etc", 1, 2, 3).is_none());
        assert!(tile_path("a/b", 1, 2, 3).is_none());
        assert!(tile_path("a\\b", 1, 2, 3).is_none());
        assert!(tile_path("", 1, 2, 3).is_none());
    }

    #[test]
    fn accepts_known_provider_ids() {
        for provider in PROVIDERS {
            assert!(
                tile_path(provider.id, 11, 1, 2).is_some(),
                "provider {} should produce a path",
                provider.id
            );
        }
    }

    fn tail_of(provider: &str, z: u32, x: u32, y: u32) -> Vec<String> {
        tile_path(provider, z, x, y)
            .unwrap()
            .components()
            .rev()
            .take(4)
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect()
    }

    /// The ids and extensions here are a contract with HybridTileService.GetDiskCachePath,
    /// not a local naming choice: if they drift, the two apps stop sharing precached tiles
    /// and the offline guarantee quietly only covers whichever app fetched them.
    #[test]
    fn builds_the_same_layout_hybridtileservice_uses() {
        assert_eq!(
            tail_of("esri-satellite", 11, 1502, 852),
            vec!["852.png", "1502", "11", "esri-satellite"]
        );
        // Sentinel-2 is the one provider Avalonia caches as .jpg.
        assert_eq!(
            tail_of("sentinel2-cloudless", 11, 1502, 852),
            vec!["852.jpg", "1502", "11", "sentinel2-cloudless"]
        );
    }

    /// Guards against the registry drifting back to Rust-side names.
    #[test]
    fn provider_ids_match_the_avalonia_registry() {
        let ids: Vec<_> = PROVIDERS.iter().map(|p| p.id).collect();
        assert_eq!(
            ids,
            vec![
                "esri-satellite",
                "carto-dark",
                "sentinel2-cloudless",
                "osm-standard",
                "usgs-imagery"
            ]
        );
    }
}
