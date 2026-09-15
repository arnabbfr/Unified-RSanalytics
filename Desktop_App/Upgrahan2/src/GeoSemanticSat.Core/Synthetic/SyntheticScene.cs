using System;
using System.Collections.Generic;
using GeoSemanticSat.Core.Model;

namespace GeoSemanticSat.Core.Synthetic;

/// <summary>
/// Synthetic multi-temporal scene generation for the demo archive and for tests.
///
/// This lived as a private copy inside GeoSemanticSat.UI.MainWindow (CreateTile plus six
/// Inject* helpers). The daemon needs the same scenes, and a third copy would have meant
/// three places to keep numerically in step, so it moved here.
///
/// GeoSemanticSat.Engine.BenchmarkRunner still holds its own near-identical private copy
/// (CreateSyntheticTile, InjectConstructionChange, ...). That one is deliberately left
/// alone: it backs the published precision/recall/F1 figures and there is no benefit to
/// this change in touching it. Collapsing it onto this class is a separate, verifiable
/// piece of work.
/// </summary>
public static class SyntheticScene
{
    /// <summary>
    /// Forest background with a river running vertically at x in [15, 35].
    /// Reflectance values are the physical premise every Inject* helper below perturbs.
    /// </summary>
    public static SatelliteTile CreateTile(
        string id,
        SensorPlatform platform,
        DateTime timestamp,
        int w,
        int h,
        AffineGeoTransform transform)
    {
        var tile = new SatelliteTile
        {
            TileId = id,
            Platform = platform,
            AcquisitionTimestamp = timestamp,
            Transform = transform,
            Width = w,
            Height = h,
            GroundSamplingDistanceMeters = 10.0,
            SunAzimuthDegrees = 135.0,
            SunElevationDegrees = 45.0,
            Bounds = new BoundingBox(
                transform.A,
                transform.D + h * transform.F,
                transform.A + w * transform.B,
                transform.D)
        };

        var red = new float[h, w];
        var green = new float[h, w];
        var blue = new float[h, w];
        var nir = new float[h, w];
        var swir = new float[h, w];

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (x >= 15 && x <= 35)
                {
                    // River: water absorbs NIR almost completely.
                    blue[y, x] = 0.35f; green[y, x] = 0.30f; red[y, x] = 0.12f;
                    nir[y, x] = 0.04f; swir[y, x] = 0.02f;
                }
                else
                {
                    // Forest canopy: high NIR reflectance.
                    blue[y, x] = 0.08f; green[y, x] = 0.18f; red[y, x] = 0.10f;
                    nir[y, x] = 0.65f; swir[y, x] = 0.15f;
                }
            }
        }

        tile.Bands[SpectralBand.Blue] = blue;
        tile.Bands[SpectralBand.Green] = green;
        tile.Bands[SpectralBand.Red] = red;
        tile.Bands[SpectralBand.NIR] = nir;
        tile.Bands[SpectralBand.SWIR1] = swir;
        return tile;
    }

    /// <summary>New concrete: NIR collapses while red and SWIR rise.</summary>
    public static void InjectConstruction(SatelliteTile tile, int startX, int startY, int w, int h)
        => Fill(tile, startX, startY, w, h, red: 0.45f, nir: 0.22f, swir: 0.52f);

    /// <summary>Cleared ground: vegetation removed, bare soil exposed.</summary>
    public static void InjectClearance(SatelliteTile tile, int startX, int startY, int w, int h)
        => Fill(tile, startX, startY, w, h, red: 0.32f, nir: 0.18f, swir: 0.40f);

    /// <summary>Open water: green dominant, NIR and SWIR near zero.</summary>
    public static void InjectWaterVariation(SatelliteTile tile, int startX, int startY, int w, int h)
        => Fill(tile, startX, startY, w, h, red: 0.08f, nir: 0.03f, swir: 0.01f, green: 0.28f);

    /// <summary>Paved corridor: linear, high SWIR.</summary>
    public static void InjectRoad(SatelliteTile tile, int startX, int startY, int w, int h)
        => Fill(tile, startX, startY, w, h, red: 0.38f, nir: 0.20f, swir: 0.42f);

    /// <summary>
    /// Bare staging ground plus a grid of high-reflectance vehicle signatures with
    /// metallic specular peaks. Drives the ActivityConcentration change class.
    /// </summary>
    public static void InjectVehicles(SatelliteTile tile, int startX, int startY, int w, int h, int numVehicles = 16)
    {
        var red = tile.Bands[SpectralBand.Red];
        var green = tile.Bands[SpectralBand.Green];
        var blue = tile.Bands[SpectralBand.Blue];
        var nir = tile.Bands[SpectralBand.NIR];
        var swir = tile.Bands[SpectralBand.SWIR1];

        // Bare soil / staging base: high BSI, low NDVI.
        for (int y = startY; y < startY + h && y < tile.Height; y++)
        {
            for (int x = startX; x < startX + w && x < tile.Width; x++)
            {
                red[y, x] = 0.38f;
                green[y, x] = 0.30f;
                blue[y, x] = 0.18f;
                nir[y, x] = 0.18f;
                swir[y, x] = 0.48f;
            }
        }

        const int cols = 4;
        for (int v = 0; v < numVehicles; v++)
        {
            int vx = startX + 4 + (v % cols) * 7;
            int vy = startY + 4 + (v / cols) * 7;

            if (vx < 0 || vx + 3 >= tile.Width || vy < 0 || vy + 3 >= tile.Height) continue;

            for (int dy = 0; dy <= 2; dy++)
            {
                for (int dx = 0; dx <= 2; dx++)
                {
                    red[vy + dy, vx + dx] = 0.88f;
                    green[vy + dy, vx + dx] = 0.85f;
                    blue[vy + dy, vx + dx] = 0.82f;
                    nir[vy + dy, vx + dx] = 0.78f;
                    swir[vy + dy, vx + dx] = 0.92f;
                }
            }
        }
    }

    /// <summary>Dark asphalt runway: uniformly low across all bands.</summary>
    public static void InjectAirfield(SatelliteTile tile, int startX, int startY, int length, int width)
    {
        var red = tile.Bands[SpectralBand.Red];
        var green = tile.Bands[SpectralBand.Green];
        var blue = tile.Bands[SpectralBand.Blue];
        var nir = tile.Bands[SpectralBand.NIR];
        var swir = tile.Bands[SpectralBand.SWIR1];

        for (int y = startY; y < startY + width && y < tile.Height; y++)
        {
            for (int x = startX; x < startX + length && x < tile.Width; x++)
            {
                red[y, x] = 0.16f;
                green[y, x] = 0.17f;
                blue[y, x] = 0.18f;
                nir[y, x] = 0.14f;
                swir[y, x] = 0.20f;
            }
        }
    }

    /// <summary>
    /// The four-scene demo archive: Jan/Feb/Mar/Apr 2024 over New Delhi, with construction,
    /// clearance, water variation, a road and vehicle concentrations appearing by T3.
    /// Returned newest-last. T1 is index 0, T3 (the usual change target) is index 2.
    /// </summary>
    public static List<SatelliteTile> BuildDemoArchive(int sceneW = 256, int sceneH = 256)
    {
        var transform = AffineGeoTransform.NorthUp(77.2000, 28.6100, 0.0001, 0.0001);

        SatelliteTile Scene(string id, int month, int day) => CreateTile(
            id,
            SensorPlatform.Sentinel2_Optical,
            new DateTime(2024, month, day, 10, 30, 0, DateTimeKind.Utc),
            sceneW, sceneH, transform);

        var t1 = Scene("S2_20240110_T1", 1, 10);
        var t2 = Scene("S2_20240215_T2", 2, 15);
        var t3 = Scene("S2_20240320_T3", 3, 20);
        var t4 = Scene("S2_20240425_T4", 4, 25);

        InjectConstruction(t3, 60, 60, 40, 40);
        InjectClearance(t3, 160, 40, 40, 40);
        InjectWaterVariation(t3, 20, 160, 30, 40);
        InjectRoad(t3, 120, 140, 100, 10);
        InjectVehicles(t3, 192, 192, 48, 48, numVehicles: 16);

        // Present in the baseline too, so the detector sees a delta rather than novelty.
        InjectVehicles(t1, 192, 192, 48, 48, numVehicles: 8);
        InjectAirfield(t1, 96, 210, 140, 14);
        InjectAirfield(t3, 96, 210, 140, 14);

        InjectConstruction(t4, 60, 60, 40, 40);
        InjectClearance(t4, 160, 40, 40, 40);
        InjectVehicles(t4, 192, 192, 48, 48, numVehicles: 20);

        return new List<SatelliteTile> { t1, t2, t3, t4 };
    }

    /// <summary>Writes a constant reflectance block, leaving unspecified bands untouched.</summary>
    private static void Fill(
        SatelliteTile tile, int startX, int startY, int w, int h,
        float red, float nir, float swir, float? green = null)
    {
        var redBand = tile.Bands[SpectralBand.Red];
        var nirBand = tile.Bands[SpectralBand.NIR];
        var swirBand = tile.Bands[SpectralBand.SWIR1];
        var greenBand = green.HasValue ? tile.Bands[SpectralBand.Green] : null;

        for (int y = startY; y < startY + h && y < tile.Height; y++)
        {
            for (int x = startX; x < startX + w && x < tile.Width; x++)
            {
                redBand[y, x] = red;
                nirBand[y, x] = nir;
                swirBand[y, x] = swir;
                if (greenBand is not null) greenBand[y, x] = green!.Value;
            }
        }
    }
}
