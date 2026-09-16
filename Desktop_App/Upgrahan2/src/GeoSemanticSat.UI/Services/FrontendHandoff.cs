using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace GeoSemanticSat.UI.Services;

/// <summary>
/// Handing the session over to the Tauri build.
///
/// The mirror of UpgrahanRust/src-tauri/src/handoff.rs, and the two must agree on the
/// preference file's location and contents - they share nothing else, one being .NET and
/// the other Rust.
///
/// The order matters and is the same on both sides: probe that the sibling exists before
/// offering the switch, write the preference BEFORE launching so the app that comes up
/// already owns the session, and roll the preference back if the launch is refused so a
/// failure leaves nothing behind.
/// </summary>
public static class FrontendHandoff
{
    private const string PreferenceFileName = "preferred-frontend";
    private const string Avalonia = "avalonia";
    private const string Tauri = "tauri";

    private static string TauriExeName =>
        OperatingSystem.IsWindows() ? "upgrahan-rust.exe" : "upgrahan-rust";

    private static string? InstallDirectory => Path.GetDirectoryName(Environment.ProcessPath);

    private static string? PreferencePath =>
        InstallDirectory is { } dir ? Path.Combine(dir, PreferenceFileName) : null;

    /// <summary>Path to the Tauri build when it sits beside this one, else null.</summary>
    public static string? TauriPath
    {
        get
        {
            if (InstallDirectory is not { } dir) return null;
            string candidate = Path.Combine(dir, TauriExeName);
            return File.Exists(candidate) ? candidate : null;
        }
    }

    /// <summary>
    /// True only in the combined bundle. Separate downloads will not find the sibling,
    /// which is the intended answer - switching is a property of the bundle.
    /// </summary>
    public static bool IsAvailable => TauriPath is not null;

    /// <summary>True when the Tauri build is already running, so this would be a no-op.</summary>
    public static bool IsTauriRunning
    {
        get
        {
            try
            {
                string name = Path.GetFileNameWithoutExtension(TauriExeName);
                return Process.GetProcessesByName(name).Any();
            }
            catch
            {
                // Process enumeration can be denied; treat unknown as "not running" so the
                // switch is still offered rather than blocked by a diagnostic failure.
                return false;
            }
        }
    }

    private static void WritePreference(string frontend)
    {
        if (PreferencePath is not { } path) return;
        try
        {
            File.WriteAllText(path, frontend);
        }
        catch
        {
            // A read-only install directory means the preference cannot be recorded. The
            // handoff itself still works; only the remembered choice is lost.
        }
    }

    /// <summary>
    /// Launches the Tauri build. Returns null on success, or a message to show the user.
    /// The caller closes this window only when this returns null.
    /// </summary>
    public static string? SwitchToTauri()
    {
        if (TauriPath is not { } path)
        {
            return "The Tauri build is not installed alongside this one. " +
                   "Download the combined bundle to switch between them.";
        }

        if (IsTauriRunning)
        {
            return "The Tauri build is already running. Switch to that window instead.";
        }

        WritePreference(Tauri);

        try
        {
            // UseShellExecute so the child is not tied to this process's console or
            // standard handles and survives this window closing.
            var startInfo = new ProcessStartInfo
            {
                FileName = path,
                WorkingDirectory = Path.GetDirectoryName(path) ?? string.Empty,
                UseShellExecute = true,
            };

            Process.Start(startInfo);
            return null;
        }
        catch (Exception ex)
        {
            // Nothing was handed over, so the recorded preference is now wrong. Undo it.
            WritePreference(Avalonia);
            return $"Could not launch the Tauri build: {ex.Message}";
        }
    }
}
