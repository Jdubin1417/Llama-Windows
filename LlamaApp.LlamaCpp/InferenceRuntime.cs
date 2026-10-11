namespace LlamaApp.Llama;

/// <summary>Backend chosen for the app-managed llama.cpp binary.</summary>
public static class InferenceRuntime
{
    public const string Automatic = "auto";
    public const string Cuda = "cuda";
    public const string Rocm = "rocm";
    public const string Vulkan = "vulkan";
    public const string Cpu = "cpu";

    public static string Normalize(string? value) => value?.ToLowerInvariant() switch
    {
        Cuda => Cuda,
        Rocm => Rocm,
        Vulkan => Vulkan,
        Cpu => Cpu,
        _ => Automatic,
    };

    /// <summary>
    /// The official Windows installer tries CUDA, ROCm, Vulkan, then CPU.
    /// SKIP_* variables narrow that choice while retaining its CPU fallback.
    /// </summary>
    internal static IReadOnlyDictionary<string, string> InstallerEnvironment(string? backend)
    {
        var skips = new Dictionary<string, string>();
        switch (Normalize(backend))
        {
            case Cuda:
                skips["SKIP_ROCM"] = "1";
                skips["SKIP_VULKAN"] = "1";
                break;
            case Rocm:
                skips["SKIP_CUDA"] = "1";
                skips["SKIP_VULKAN"] = "1";
                break;
            case Vulkan:
                skips["SKIP_CUDA"] = "1";
                skips["SKIP_ROCM"] = "1";
                break;
            case Cpu:
                skips["SKIP_CUDA"] = "1";
                skips["SKIP_ROCM"] = "1";
                skips["SKIP_VULKAN"] = "1";
                break;
        }
        return skips;
    }

    internal static bool RequiresManagedInstall(string? requested, string? installed)
        => Normalize(requested) != Normalize(installed);

    /// <summary>Resolve a saved GPU choice to the current llama.cpp device id.</summary>
    public static DeviceSelection SelectDevice(string? backend, string? gpuName, DeviceProbe probe)
    {
        var requested = Normalize(backend);
        if (requested == Cpu)
            return new DeviceSelection("none", null);
        if (!probe.Succeeded)
            return new DeviceSelection("none", "GPU detection failed; using the CPU.");
        if (requested == Automatic && string.IsNullOrWhiteSpace(gpuName) &&
            !probe.Devices.Any(d => d.Kind == DeviceKind.Rocm))
            return new DeviceSelection(null, null); // Keep llama.cpp's defaults.

        var candidates = probe.Devices.Where(device => requested switch
        {
            Cuda => device.Kind == DeviceKind.Cuda,
            Rocm => device.Kind == DeviceKind.Rocm,
            Vulkan => device.Kind == DeviceKind.Vulkan,
            _ => device.Kind != DeviceKind.Cpu && device.Kind != DeviceKind.Unknown,
        }).ToArray();

        if (!string.IsNullOrWhiteSpace(gpuName))
        {
            var match = GpuDeviceChoice.Resolve(gpuName, candidates);
            return match is null
                ? new DeviceSelection("none", $"{gpuName} is unavailable; using the CPU.")
                : match.CompatibilityError is { } error
                    ? new DeviceSelection("none", null) { Error = error }
                    : new DeviceSelection(match.Id, null);
        }

        var compatible = candidates.Where(d => d.CompatibilityError is null).ToArray();
        if (candidates.Length > 0 && compatible.Length == 0)
            return new DeviceSelection("none", null) { Error = candidates[0].CompatibilityError };
        return candidates.Length == 0
            ? new DeviceSelection("none", $"No {requested.ToUpperInvariant()} GPU is available; using the CPU.")
            : new DeviceSelection(string.Join(',', compatible.Select(device => device.Id)), null);
    }

    /// <summary>Validated --n-gpu-layers value; null means llama.cpp's automatic default.</summary>
    internal static string? GpuLayersArgument(string? value)
    {
        if (string.Equals(value, "all", StringComparison.OrdinalIgnoreCase)) return "all";
        return int.TryParse(value, out var count) && count is >= 0 and <= 1000
            ? count.ToString(System.Globalization.CultureInfo.InvariantCulture)
            : null;
    }
}

/// <summary>Resolved CLI device and any reason for a CPU fallback.</summary>
public sealed record DeviceSelection(string? DeviceArgument, string? FallbackReason)
{
    public string? Error { get; init; }
}
