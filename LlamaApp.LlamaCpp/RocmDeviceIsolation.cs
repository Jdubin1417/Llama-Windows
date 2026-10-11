using System.Diagnostics;

namespace LlamaApp.Llama;

internal sealed record RocmDeviceIsolation(string VisibleDevices, Dictionary<string, string> DeviceIds)
{
    internal static RocmDeviceIsolation? Create(DeviceProbe probe, string? inheritedVisibility = null)
    {
        var rocm = probe.Devices.Where(d => d.Kind == DeviceKind.Rocm).ToArray();
        if (rocm.Length == 0) return null;
        var inherited = string.IsNullOrWhiteSpace(inheritedVisibility) ? null :
            inheritedVisibility.Split(',', StringSplitOptions.TrimEntries);
        var compatible = rocm.Where(d => d.CompatibilityError is null)
            .OrderBy(d => RocmCompatibility.DeviceIndex(d.Id)).ToArray();
        var physical = new List<string>();
        var ids = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < compatible.Length; i++)
        {
            var index = RocmCompatibility.DeviceIndex(compatible[i].Id)
                ?? throw new InvalidOperationException("The ROCm device index could not be resolved.");
            if (inherited is not null && (index >= inherited.Length ||
                string.IsNullOrWhiteSpace(inherited[index])))
                throw new InvalidOperationException("The inherited HIP_VISIBLE_DEVICES mapping could not be resolved.");
            physical.Add(inherited is null ? index.ToString(System.Globalization.CultureInfo.InvariantCulture) : inherited[index]);
            ids.Add(compatible[i].Id, $"ROCm{i}");
        }
        return new RocmDeviceIsolation(physical.Count == 0 ? "-1" : string.Join(',', physical), ids);
    }

    internal static string? InheritedVisibility =>
        Environment.GetEnvironmentVariable("HIP_VISIBLE_DEVICES") ??
        Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");

    internal void Apply(ProcessStartInfo startInfo) =>
        startInfo.Environment["HIP_VISIBLE_DEVICES"] = VisibleDevices;

    internal string? MapArgument(string? argument)
    {
        if (argument is null) return null;
        var ids = argument.Split(',').Select(id =>
            DeviceIds.TryGetValue(id, out var mapped) ? mapped :
                RocmCompatibility.DeviceIndex(id) is not null ? "none" : id).ToArray();
        return ids.Contains("none", StringComparer.Ordinal) ? "none" : string.Join(',', ids);
    }

    internal bool Matches(RocmDeviceIsolation other) =>
        VisibleDevices == other.VisibleDevices && DeviceIds.Count == other.DeviceIds.Count &&
        DeviceIds.All(pair => other.DeviceIds.TryGetValue(pair.Key, out var value) && value == pair.Value);

    internal static bool SameDevices(DeviceProbe original, DeviceProbe current) =>
        original.Succeeded && current.Succeeded && original.Devices.Count == current.Devices.Count &&
        original.Devices.All(d => current.Devices.Any(other => d.Id == other.Id && d.Name == other.Name &&
            d.Architecture == other.Architecture && d.CompatibilityError == other.CompatibilityError));
}

internal sealed record RocmServerState(int Pid, long StartedUtcTicks, string BinaryPath,
    long BinaryModifiedUtcTicks, RocmDeviceIsolation Isolation, string? ActiveDeviceId,
    Dictionary<string, string> DeviceNames, string RuntimeBackend, string? GpuDeviceName)
{
    internal bool Matches(int pid, long startedUtcTicks, string? binaryPath, long binaryModifiedUtcTicks,
        RocmDeviceIsolation desired, DeviceProbe probe, string backend, string? gpuName) =>
        Pid == pid && StartedUtcTicks == startedUtcTicks &&
        string.Equals(BinaryPath, binaryPath, StringComparison.OrdinalIgnoreCase) &&
        BinaryModifiedUtcTicks == binaryModifiedUtcTicks &&
        Isolation is not null && Isolation.DeviceIds is not null && Isolation.Matches(desired) &&
        DeviceNames is not null && DeviceNames.Count == probe.Devices.Count &&
        probe.Devices.All(d => DeviceNames.TryGetValue(d.Id, out var name) && name == d.Name) &&
        RuntimeBackend == backend && GpuDeviceName == gpuName;
}
