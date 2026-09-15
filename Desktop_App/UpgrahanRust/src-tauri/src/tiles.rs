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
    /// True when the provider only covers part of the globe, so the UI can say so
    /// instead of silently showing blank tiles outside coverage.
    pub region_limited: bool,
}

/// Mirrors the providers registered in HybridTileService so both frontends offer the same
/// basemaps and share cache directories. Default is ESRI satellite imagery.
pub const PROVIDERS: &[TileProvider] = &[
    TileProvider {
        id: "EsriSatellite",
        label: "ESRI Satellite",
        url_template: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution: "Esri, Maxar, Earthstar Geographics",
        region_limited: false,
    },
    TileProvider {
        id: "CartoDark",
        label: "CartoDB Dark Matter",
        url_template: "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attribution: "CARTO, OpenStreetMap contributors",
        region_limited: false,
    },
    TileProvider {
        id: "Sentinel2Cloudless",
        label: "Sentinel-2 Cloudless 2024",
        url_template: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg",
        attribution: "Sentinel-2 cloudless by EOX IT Services",
        region_limited: false,
    },
    TileProvider {
        id: "OpenStreetMap",
        label: "OpenStreetMap",
        url_template: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution: "OpenStreetMap contributors",
        region_limited: false,
    },
    TileProvider {
        id: "UsgsTopo",
        label: "USGS Topo (United States only)",
        url_template: "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        attribution: "USGS The National Map",
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

fn tile_path(provider: &str, z: u32, x: u32, y: u32) -> Option<PathBuf> {
    // Reject anything that could escape the cache directory. provider comes from the
    // frontend, so it is untrusted input even though the UI only ever sends known ids.
    if provider.is_empty()
        || provider.contains("..")
        || provider.contains('/')
        || provider.contains('\\')
        || !provider.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return None;
    }

    cache_root().map(|root| {
        root.join(provider)
            .join(z.to_string())
            .join(x.to_string())
            .join(format!("{y}.png"))
    })
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
    let Some(root) = cache_root().map(|r| r.join(provider)) else {
        return 0;
    };

    fn walk(dir: &PathBuf) -> usize {
        let Ok(entries) = fs::read_dir(dir) else {
            return 0;
        };
        entries
            .filter_map(Result::ok)
            .map(|e| {
                let path = e.path();
                if path.is_dir() {
                    walk(&path)
                } else if path.extension().is_some_and(|x| x == "png") {
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    walk(&root)
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

    #[test]
    fn builds_the_same_layout_hybridtileservice_uses() {
        let path = tile_path("EsriSatellite", 11, 1502, 852).unwrap();
        let tail: Vec<_> = path
            .components()
            .rev()
            .take(4)
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();
        assert_eq!(tail, vec!["852.png", "1502", "11", "EsriSatellite"]);
    }
}
