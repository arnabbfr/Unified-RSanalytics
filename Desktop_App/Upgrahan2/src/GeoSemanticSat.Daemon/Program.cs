using System;
using System.IO;
using System.Net;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using GeoSemanticSat.Core.Raster;
using GeoSemanticSat.Daemon;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

// GeoSemanticSat analysis daemon.
//
// Hosts GeoSemanticSat.Core over local HTTP so a non-.NET frontend (the Tauri app) can run
// exactly the same analysis code as the Avalonia app. Because both frontends end up calling
// this same compiled Core, their results are identical by construction - the A/B comparison
// between the two UIs measures the UI, not the algorithms.
//
// Security posture: bound to 127.0.0.1 only, on an ephemeral port, behind a per-process
// bearer token. The port and token are printed to stdout as a single JSON line for the
// parent process to read. Nothing is discoverable by another machine, and an ephemeral port
// avoids both collisions and a firewall prompt.

int requestedPort = 0;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] is "--port" or "-p" && int.TryParse(args[i + 1], out int parsed)) requestedPort = parsed;
}

string token = Convert.ToHexString(RandomNumberGenerator.GetBytes(24));

var builder = WebApplication.CreateSlimBuilder(args);
builder.Logging.ClearProviders();   // stdout is the handshake channel, keep it clean
builder.WebHost.ConfigureKestrel(k =>
{
    // Listen(IPAddress.Loopback, 0) rather than ListenLocalhost(0): Kestrel rejects dynamic
    // port binding on the localhost helper because it would have to bind two sockets
    // (IPv4 + IPv6) and could not guarantee the same port on both.
    k.Listen(IPAddress.Loopback, requestedPort, o => o.Protocols = HttpProtocols.Http1);
});

builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
    o.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    // Enums as names: "Construction", not 1. The frontend should never depend on ordinals,
    // which would silently shift if a ChangeType were ever inserted mid-enum.
    o.SerializerOptions.Converters.Add(new JsonStringEnumConverter());
});

builder.Services.AddSingleton<AnalysisSession>();

// The Tauri frontend runs on its own origin, so calls to this daemon are cross-origin and
// the webview enforces CORS on them. Only the webview origins are allowed - not "*" - so a
// page in an ordinary browser still cannot read responses even if it guessed the port.
const string WebviewCors = "webview";
builder.Services.AddCors(o => o.AddPolicy(WebviewCors, policy => policy
    .WithOrigins(
        "http://tauri.localhost",    // Tauri v2 on Windows
        "https://tauri.localhost",
        "tauri://localhost",         // Tauri v2 on macOS and Linux
        "http://localhost:5183")     // vite dev server
    .AllowAnyHeader()
    .AllowAnyMethod()));

var app = builder.Build();

app.UseCors(WebviewCors);

// Every route except /health requires the token.
app.Use(async (ctx, next) =>
{
    // A CORS preflight cannot carry the token header - the browser decides what to send,
    // and it never forwards custom headers on OPTIONS. Rejecting it here would block every
    // cross-origin call before the real request was ever made.
    if (HttpMethods.IsOptions(ctx.Request.Method) ||
        ctx.Request.Path.StartsWithSegments("/health"))
    {
        await next();
        return;
    }

    string? presented = ctx.Request.Headers["X-GSS-Token"].FirstOrDefault()
                        ?? ctx.Request.Query["token"].FirstOrDefault();

    if (presented is null || !CryptographicOperations.FixedTimeEquals(
            System.Text.Encoding.UTF8.GetBytes(presented),
            System.Text.Encoding.UTF8.GetBytes(token)))
    {
        ctx.Response.StatusCode = StatusCodes.Status401Unauthorized;
        return;
    }

    await next();
});

// Turn the Core exceptions we expect into useful status codes rather than a 500.
app.Use(async (ctx, next) =>
{
    try { await next(); }
    catch (KeyNotFoundException ex)
    {
        ctx.Response.StatusCode = StatusCodes.Status404NotFound;
        await ctx.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
    catch (FileNotFoundException ex)
    {
        ctx.Response.StatusCode = StatusCodes.Status404NotFound;
        await ctx.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
    catch (DirectoryNotFoundException ex)
    {
        // Exporting to a path whose parent does not exist used to escape as a bare 500 with
        // an empty body. It is a bad request, and the caller can act on it.
        ctx.Response.StatusCode = StatusCodes.Status400BadRequest;
        await ctx.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
    catch (UnauthorizedAccessException ex)
    {
        ctx.Response.StatusCode = StatusCodes.Status403Forbidden;
        await ctx.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
    catch (ArgumentException ex)
    {
        ctx.Response.StatusCode = StatusCodes.Status400BadRequest;
        await ctx.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
    catch (InvalidOperationException ex)
    {
        ctx.Response.StatusCode = StatusCodes.Status409Conflict;
        await ctx.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
});

static IResult Bmp(MemoryStream stream)
{
    stream.Position = 0;
    return Results.File(stream.ToArray(), "image/bmp");
}

// ---- system ----
app.MapGet("/health", () => Results.Ok(new { status = "ok", service = "geosemanticsat-daemon" }));

// ---- session ----
app.MapPost("/session/init", (AnalysisSession s) => Results.Ok(s.Initialize()));
app.MapGet("/session/status", (AnalysisSession s) => Results.Ok(s.Status()));
app.MapPost("/session/load", (AnalysisSession s, Contracts.ExportRequest req) =>
    Results.Ok(s.LoadGeoTiff(req.Path)));
app.MapPost("/session/generate", (AnalysisSession s, Contracts.GenerateArchiveRequest req) =>
{
    req.Validate();
    return Results.Ok(s.GenerateArchiveAt(req.Latitude, req.Longitude));
});
app.MapPost("/benchmark", (AnalysisSession s, Contracts.BenchmarkRequest req) =>
    Results.Ok(new { outputDirectory = s.RunBenchmark(req.OutputDirectory) }));

// ---- search (returns Results - see CONTEXT.md) ----
app.MapPost("/search/text", (AnalysisSession s, Contracts.TextSearchRequest req) =>
{
    req.Validate();
    return Results.Ok(s.SearchByText(req));
});
app.MapPost("/search/similar", (AnalysisSession s, Contracts.SimilarSearchRequest req) =>
{
    req.Validate();
    return Results.Ok(s.SearchSimilar(req));
});
app.MapPost("/search/feedback", (AnalysisSession s, Contracts.FeedbackSearchRequest req) =>
{
    req.Validate();
    return Results.Ok(s.SearchWithFeedback(req));
});
app.MapGet("/search/explain", (string q) =>
    Results.Ok(new { explanation = AnalysisSession.ExplainQuery(q) }));

// ---- change detection (returns Candidates) ----
app.MapPost("/change/detect", (AnalysisSession s, Contracts.DetectRequest req) =>
    Results.Ok(s.DetectChanges(req)));
app.MapGet("/change/candidates", (AnalysisSession s) => Results.Ok(s.Candidates()));
app.MapPost("/change/search", (AnalysisSession s, Contracts.ChangeSearchRequest req) =>
{
    req.Validate();
    return Results.Ok(s.SearchChanges(req));
});

// ---- clustering ----
app.MapPost("/cluster", (AnalysisSession s, Contracts.ClusterRequest req) =>
    Results.Ok(s.Cluster(req)));

// ---- review / triage ----
app.MapGet("/review", (AnalysisSession s) => Results.Ok(s.Review()));
// NoContent, not Ok: these succeed without returning anything, and Results.Ok() emits a
// 200 with an empty body, which a JSON client cannot distinguish from a truncated response.
app.MapPost("/review/{id}/confirm", (AnalysisSession s, string id, Contracts.VerdictRequest req) =>
    s.Confirm(id, req.Notes) ? Results.NoContent() : Results.NotFound());
app.MapPost("/review/{id}/reject", (AnalysisSession s, string id, Contracts.VerdictRequest req) =>
    s.Reject(id, req.Notes) ? Results.NoContent() : Results.NotFound());
app.MapPost("/review/{id}/flag", (AnalysisSession s, string id, Contracts.VerdictRequest req) =>
    s.Flag(id, req.Notes) ? Results.NoContent() : Results.NotFound());

// ---- export ----
app.MapPost("/export/geojson", (AnalysisSession s, Contracts.ExportRequest req) =>
{
    s.ExportGeoJson(req.Path);
    return Results.Ok(new { path = req.Path });
});

// ---- imagery ----
// RasterVisualizer already emits BMP byte streams, so this half of the boundary was
// serialization-shaped before the daemon existed.
app.MapGet("/image/tile", (AnalysisSession s, string handle,
                           int x = 0, int y = 0, int w = -1, int h = -1,
                           string mode = "TrueColorRGB") =>
    Bmp(s.WithTile(handle, t => RasterVisualizer.RenderTileToBmpStream(
        t, x, y, w, h, Enum.Parse<VisualRenderMode>(mode, ignoreCase: true)))));

app.MapGet("/image/heatmap", (AnalysisSession s, string? changeId) =>
    Bmp(s.WithHeatmapContext((t1, t2, changes) =>
        RasterVisualizer.RenderChangeHeatmapBmpStream(
            t1, t2, changes,
            changeId is null ? null : changes.FirstOrDefault(c => c.Id == changeId)))));

app.MapGet("/image/focused/{changeId}", (AnalysisSession s, string changeId,
                                         string mode = "TrueColorRGB", string heatmap = "CVA") =>
    s.WithCandidate(changeId, (t1, t2, rec) =>
    {
        var f = RasterVisualizer.RenderFocusedSite(
            t1, t2, rec, Enum.Parse<VisualRenderMode>(mode, ignoreCase: true), heatmap);
        return Results.Ok(new
        {
            t1 = Convert.ToBase64String(f.T1Stream.ToArray()),
            t2 = Convert.ToBase64String(f.T2Stream.ToArray()),
            overlay = Convert.ToBase64String(f.OverlayStream.ToArray()),
            cropX = f.CropX, cropY = f.CropY, cropWidth = f.CropWidth, cropHeight = f.CropHeight,
            peakMagnitude = f.PeakMagnitude, meanMagnitude = f.MeanMagnitude,
        });
    }));

app.MapGet("/image/indices/{changeId}", (AnalysisSession s, string changeId) =>
    s.WithCandidate(changeId, (t1, t2, rec) =>
    {
        var m = RasterVisualizer.RenderFocusedIndexHeatmaps(t1, t2, rec);
        return Results.Ok(new
        {
            ndvi = Convert.ToBase64String(m.NdviStream.ToArray()),
            ndbi = Convert.ToBase64String(m.NdbiStream.ToArray()),
            ndwi = Convert.ToBase64String(m.NdwiStream.ToArray()),
            bsi = Convert.ToBase64String(m.BsiStream.ToArray()),
            cropX = m.CropX, cropY = m.CropY, cropWidth = m.CropWidth, cropHeight = m.CropHeight,
        });
    }));

app.MapGet("/image/spectral/{changeId}", (AnalysisSession s, string changeId) =>
    Bmp(s.WithCandidate(changeId, (t1, t2, rec) =>
        RasterVisualizer.RenderSpectralProfileChart(t1, t2, rec))));

// Handshake: the parent process reads this one line to learn where to connect.
app.Lifetime.ApplicationStarted.Register(() =>
{
    var addresses = app.Services
        .GetRequiredService<Microsoft.AspNetCore.Hosting.Server.IServer>()
        .Features.Get<Microsoft.AspNetCore.Hosting.Server.Features.IServerAddressesFeature>();

    string address = addresses?.Addresses.FirstOrDefault() ?? "http://127.0.0.1:0";
    int port = new Uri(address).Port;

    Console.WriteLine(JsonSerializer.Serialize(new { ready = true, port, token }));
    Console.Out.Flush();
});

app.Run();
