// Experimental Rust/Tauri frontend for UpaGraha / GeoSemanticSat.
//
// Runs alongside the Avalonia app rather than replacing it. Both call the same compiled
// GeoSemanticSat.Core - this one over local HTTP via the daemon - so a comparison between
// them measures the interface, not the analysis.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod daemon;
mod handoff;
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

/// Whether the Avalonia build can be switched to, and whether it is already up.
///
/// The UI only offers the control when this reports available, so the switch is never
/// presented in a build where it cannot work.
#[tauri::command]
fn avalonia_target() -> handoff::HandoffTarget {
    handoff::avalonia_target()
}

/// Hands the session to the Avalonia build and closes this one.
///
/// The preference is written before the launch so the app that comes up already owns the
/// session, and rolled back inside the handoff if the launch is refused - so a failure
/// here means nothing changed and the caller only has to revert its own toggle.
#[tauri::command]
fn switch_to_avalonia(app: tauri::AppHandle) -> Result<(), String> {
    handoff::switch_to_avalonia()?;

    // Stop the analysis daemon explicitly rather than relying on the window-destroyed
    // handler: app.exit does not reliably deliver that event, and the Avalonia build runs
    // its analysis in-process, so a surviving daemon is pure waste holding a loopback port.
    // Left alone it accumulates one orphan per switch.
    if let Some(state) = app.try_state::<DaemonState>() {
        state.shutdown();
    }

    // Give the child a moment to get its window up before this one disappears, so the
    // desktop is never briefly empty.
    std::thread::sleep(std::time::Duration::from_millis(600));
    app.exit(0);
    Ok(())
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
            avalonia_target,
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
