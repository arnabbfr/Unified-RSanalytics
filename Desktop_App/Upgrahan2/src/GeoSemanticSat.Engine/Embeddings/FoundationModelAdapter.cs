using System;
using System.Collections.Generic;
using GeoSemanticSat.Core.Model;
using GeoSemanticSat.Core.Processing;
using GeoSemanticSat.Core.VectorIndex;
using VectorIndexStore = GeoSemanticSat.Core.VectorIndex.VectorIndex;

namespace GeoSemanticSat.Engine.Embeddings;

/// <summary>
/// Pretrained Geospatial Foundation Model kinds supported by UpaGraha / GeoSemanticSat.
/// </summary>
public enum FoundationModelKind
{
    NativeBaseline,
    TerraMind,          // ibm-esa-geospatial/TerraMind-1.0-base
    SatMaePP,           // BiliSakura/SATMAE-PP-transformers
    GfmComposition,     // GFM_Composition_Pretraining (SAR + Optical)
    PrithviTemporal     // ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL
}

/// <summary>
/// C# Offline Adapter for Pretrained Geospatial Foundation Models.
/// Bridges ONNX runtime outputs with the single source of truth 128-dimensional
/// vector space defined in <see cref="SemanticEmbeddingLayout"/>.
/// </summary>
public class FoundationModelAdapter : IDisposable
{
    private readonly FoundationModelKind _modelKind;
    private readonly OnnxModelRunner? _onnxRunner;

    public FoundationModelKind ModelKind => _modelKind;
    public bool IsNeuralActive => _onnxRunner != null && _onnxRunner.IsModelLoaded;

    public FoundationModelAdapter(FoundationModelKind modelKind = FoundationModelKind.NativeBaseline, string? onnxPath = null)
    {
        _modelKind = modelKind;
        string? resolvedPath = !string.IsNullOrEmpty(onnxPath) && System.IO.File.Exists(onnxPath) 
            ? onnxPath 
            : ResolveModelPath(modelKind);

        if (!string.IsNullOrEmpty(resolvedPath) && System.IO.File.Exists(resolvedPath))
        {
            _onnxRunner = new OnnxModelRunner(resolvedPath);
        }
    }

    /// <summary>
    /// Auto-discover staged ONNX foundation model weights across standard repository directories.
    /// </summary>
    public static string? ResolveModelPath(FoundationModelKind kind)
    {
        string modelName = kind switch
        {
            FoundationModelKind.TerraMind => "terramind",
            FoundationModelKind.SatMaePP => "satmaepp",
            FoundationModelKind.GfmComposition => "gfm",
            FoundationModelKind.PrithviTemporal => "prithvi",
            _ => "baseline"
        };

        if (modelName == "baseline") return null;

        string[] candidateSubpaths = new[]
        {
            Path.Combine("fine_tune", "checkpoints", modelName, "model.onnx"),
            Path.Combine("..", "..", "..", "..", "..", "fine_tune", "checkpoints", modelName, "model.onnx"),
            Path.Combine("models", modelName, "model.onnx"),
            Path.Combine("models", $"{modelName}.onnx"),
            Path.Combine("..", "..", "..", "..", "..", "models", $"{modelName}.onnx"),
        };

        foreach (var sub in candidateSubpaths)
        {
            try
            {
                string fullPath = Path.GetFullPath(sub);
                if (File.Exists(fullPath)) return fullPath;
            }
            catch { }
        }

        return null;
    }

    /// <summary>
    /// Encode satellite imagery patch using the active foundation model architecture.
    /// Falls back to <see cref="MultiSpectralVisionEncoder.EncodePatch"/> if neural weights are not staged.
    /// </summary>
    public float[] EncodePatch(SatelliteTile tile, int startX, int startY, int patchW, int patchH)
    {
        // 1. If ONNX weights are loaded, run neural inference
        if (IsNeuralActive && _onnxRunner != null)
        {
            float[]? neuralOutput = RunNeuralInference(tile, startX, startY, patchW, patchH);
            if (neuralOutput != null && neuralOutput.Length >= SemanticEmbeddingLayout.Dimension)
            {
                VectorIndexStore.NormalizeInPlace(neuralOutput);
                return neuralOutput;
            }
        }

        // 2. High-Fidelity offline native spectral foundation mapping
        float[] embedding = MultiSpectralVisionEncoder.EncodePatch(tile, startX, startY, patchW, patchH);

        // Apply model-specific compositional weighting
        switch (_modelKind)
        {
            case FoundationModelKind.TerraMind:
                // Boost cross-modal semantic axes
                embedding[SemanticEmbeddingLayout.StructureNearWater] *= 1.1f;
                embedding[SemanticEmbeddingLayout.LinearContinuity] *= 1.05f;
                break;

            case FoundationModelKind.SatMaePP:
                // Grouped multi-spectral balance
                embedding[SemanticEmbeddingLayout.Vegetation] *= 1.05f;
                embedding[SemanticEmbeddingLayout.TextureEnergy] *= 1.1f;
                break;

            case FoundationModelKind.GfmComposition:
                // If SAR is present, emphasize high-contrast microwave scattering
                if (tile.HasBand(SpectralBand.SAR_VV))
                {
                    embedding[SemanticEmbeddingLayout.ManMadeContrast] *= 1.2f;
                    embedding[SemanticEmbeddingLayout.ActivityPeaks] *= 1.15f;
                }
                break;

            case FoundationModelKind.PrithviTemporal:
                // Emphasize onset dynamics
                embedding[SemanticEmbeddingLayout.ClearedGround] *= 1.1f;
                embedding[SemanticEmbeddingLayout.BuiltUp] *= 1.05f;
                break;

            default:
                break;
        }

        VectorIndexStore.NormalizeInPlace(embedding);
        return embedding;
    }

    /// <summary>
    /// Run pixel-level flood inundation segmentation across a satellite tile.
    /// Returns a 2D float array of flood probabilities in [0..1].
    /// </summary>
    public float[,] SegmentFloodInundation(SatelliteTile tile, float threshold = 0.50f)
    {
        int h = tile.Height;
        int w = tile.Width;
        float[,] probMap = new float[h, w];

        if (IsNeuralActive && _onnxRunner != null)
        {
            float[]? neuralPred = RunNeuralInference(tile, 0, 0, w, h);
            if (neuralPred != null && neuralPred.Length == h * w)
            {
                int idx = 0;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        probMap[y, x] = neuralPred[idx++];
                    }
                }
                return probMap;
            }
        }

        // Deterministic MNDWI / SAR Water Inundation Mapping
        var green = tile.GetBandOrFallback(SpectralBand.Green, SpectralBand.Red);
        var swir = tile.GetBandOrFallback(SpectralBand.SWIR1, SpectralBand.NIR);
        bool hasSar = tile.HasBand(SpectralBand.SAR_VV);
        var vv = hasSar ? tile.GetBandOrFallback(SpectralBand.SAR_VV, SpectralBand.Red) : null;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float g = green[y, x];
                float s = swir[y, x];
                float mndwi = (g - s) / (g + s + 1e-6f);

                if (hasSar && vv != null)
                {
                    float sarVal = vv[y, x];
                    probMap[y, x] = (mndwi > 0.15f || sarVal < 0.15f) ? 0.95f : 0.05f;
                }
                else
                {
                    probMap[y, x] = mndwi > 0.10f ? 0.90f : 0.10f;
                }
            }
        }

        return probMap;
    }

    private float[]? RunNeuralInference(SatelliteTile tile, int startX, int startY, int patchW, int patchH)
    {
        if (_onnxRunner == null) return null;

        var red = tile.GetBandOrFallback(SpectralBand.Red, SpectralBand.Red);
        var green = tile.GetBandOrFallback(SpectralBand.Green, SpectralBand.Red);
        var blue = tile.GetBandOrFallback(SpectralBand.Blue, SpectralBand.Red);
        var nir = tile.GetBandOrFallback(SpectralBand.NIR, SpectralBand.Red);

        // Prepare flat tensor [1, 4, patchH, patchW]
        float[] tensorData = new float[4 * patchH * patchW];
        int idx = 0;
        float[][,] bands = new float[][,] { red, green, blue, nir };

        for (int b = 0; b < 4; b++)
        {
            for (int y = startY; y < startY + patchH; y++)
            {
                for (int x = startX; x < startX + patchW; x++)
                {
                    int clampedY = Math.Clamp(y, 0, tile.Height - 1);
                    int clampedX = Math.Clamp(x, 0, tile.Width - 1);
                    tensorData[idx++] = bands[b][clampedY, clampedX];
                }
            }
        }

        return _onnxRunner.RunInference(tensorData, new int[] { 1, 4, patchH, patchW });
    }

    public void Dispose()
    {
        _onnxRunner?.Dispose();
    }
}
