// Experimental Rust/Tauri frontend for UpaGraha / GeoSemanticSat.
//
// Runs alongside the Avalonia app rather than replacing it. Both call the same compiled
// GeoSemanticSat.Core - this one over local HTTP via the daemon - so a comparison between
// them measures the interface, not the analysis.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod daemon;
mod tiles;

use daemon::{DaemonEndpoint, DaemonState};
use tauri::{Manager, State};

/// Starts the analysis daemon and returns where to reach it.
/// The frontend calls this once on mount, before anything else.
#[tauri::command]
fn start_daemon(state: State<'_, DaemonState>) -> Result<DaemonEndpoint, String> {
    daemon::start(&state)
}

#[tauri::command]
fn tile_providers() -> Vec<tiles::TileProvider> {
    tiles::PROVIDERS.to_vec()
}

/// Reads a basemap tile from the shared on-disk cache.
/// Returns None on a miss so the frontend can decide whether to hit the network,
/// which is what keeps OfflineStrict honest.
#[tauri::command]
fn read_cached_tile(provider: String, z: u32, x: u32, y: u32) -> Option<Vec<u8>> {
    tiles::read_cached(&provider, z, x, y)
}

#[tauri::command]
fn write_cached_tile(provider: String, z: u32, x: u32, y: u32, bytes: Vec<u8>) -> Result<(), String> {
    tiles::write_cached(&provider, z, x, y, &bytes)
}

#[tauri::command]
fn cached_tile_count(provider: String) -> usize {
    tiles::cached_tile_count(&provider)
}

/// Relaunches into the Avalonia build and exits this one.
///
/// Only offered when the sibling executable is actually present next to us (the combined
/// demo bundle). Separate downloads simply will not find it, and the UI hides the control.
#[tauri::command]
fn switch_to_avalonia(app: tauri::AppHandle) -> Result<(), String> {
    let exe_name = if cfg!(windows) {
        "GeoSemanticSat.UI.exe"
    } else {
        "GeoSemanticSat.UI"
    };

    let sibling = std::env::current_exe()
        .map_err(|e| format!("Could not resolve the current executable: {e}"))?
        .parent()
        .ok_or("Could not resolve the install directory.")?
        .join(exe_name);

    if !sibling.is_file() {
        return Err(format!(
            "The Avalonia build is not installed alongside this one ({}).",
            sibling.display()
        ));
    }

    std::process::Command::new(&sibling)
        .spawn()
        .map_err(|e| format!("Could not launch the Avalonia build: {e}"))?;

    app.exit(0);
    Ok(())
}

/// True when the sibling Avalonia binary exists, so the UI can hide the switch control.
#[tauri::command]
fn avalonia_available() -> bool {
    let exe_name = if cfg!(windows) {
        "GeoSemanticSat.UI.exe"
    } else {
        "GeoSemanticSat.UI"
    };

    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(exe_name)))
        .is_some_and(|p| p.is_file())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_process::init())
        .manage(DaemonState::new())
        .invoke_handler(tauri::generate_handler![
            start_daemon,
            tile_providers,
            read_cached_tile,
            write_cached_tile,
            cached_tile_count,
            switch_to_avalonia,
            avalonia_available,
        ])
        .on_window_event(|window, event| {
            // Kill the daemon with the window, so closing the UI never strands a child
            // process holding a loopback port.
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.app_handle().try_state::<DaemonState>() {
                    state.shutdown();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running the UpaGraha Tauri application");
}
