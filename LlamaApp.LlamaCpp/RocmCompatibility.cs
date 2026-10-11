using System.Buffers.Binary;
using System.Reflection.PortableExecutable;
using System.Text;
using System.Text.RegularExpressions;
using LlamaApp.Common;

namespace LlamaApp.Llama;

internal static partial class RocmCompatibility
{
    private static ReadOnlySpan<byte> BundleMagic => "__CLANG_OFFLOAD_BUNDLE__"u8;

    internal static async Task<IReadOnlyList<LlamaDevice>> InspectAsync(
        string binary, IReadOnlyList<LlamaDevice> devices, CancellationToken cancel)
    {
        if (!devices.Any(d => d.Kind == DeviceKind.Rocm)) return devices;

        IReadOnlyList<string> targets = [];
        IReadOnlyDictionary<int, (string Name, string Architecture)> hardware =
            new Dictionary<int, (string, string)>();
        string? inspectionError = null;
        try
        {
            targets = await Task.Run(() => ReadTargets(binary), cancel);
            var bin = HipSdkEnvironment.FindBin();
            var hipInfo = bin is null ? null : Path.Combine(bin, "hipInfo.exe");
            if (hipInfo is not null && File.Exists(hipInfo))
            {
                var output = await DeviceQuery.RunAsync(hipInfo, [], cancel);
                if (output is not null) hardware = ParseHipInfo(output);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or BadImageFormatException or
                                   System.ComponentModel.Win32Exception)
        {
            Log.Warn(ex, "ROCm compatibility inspection failed");
            inspectionError = "The installed ROCm kernel targets could not be read.";
        }

        return devices.Select(device =>
        {
            if (device.Kind != DeviceKind.Rocm) return device;
            var index = DeviceIndex(device.Id);
            var architecture = index is not null && hardware.TryGetValue(index.Value, out var info) &&
                string.Equals(info.Name, device.Name, StringComparison.Ordinal)
                    ? info.Architecture : null;
            var error = inspectionError ?? CompatibilityError(device.Name, architecture, targets);
            if (error is not null) Log.Warn(error);
            return device with { Architecture = architecture, CompatibilityError = error };
        }).ToArray();
    }

    internal static string? CompatibilityError(
        string name, string? architecture, IReadOnlyList<string> targets)
    {
        if (targets.Count == 0)
            return $"Cannot verify the installed ROCm kernels for {name}. Use Vulkan/CPU or install a ROCm build with readable HIP kernel targets.";
        if (architecture is null)
            return $"Cannot verify the ROCm architecture of {name}. Install the HIP SDK (including hipInfo.exe), or use Vulkan/CPU.";
        if (targets.Any(target => Supports(target, architecture))) return null;
        return $"{name} ({architecture}) is incompatible with the installed ROCm build ({string.Join(", ", targets)}). " +
            "Install a matching ROCm build, or switch to Vulkan/CPU.";
    }

    internal static bool Supports(string target, string architecture)
    {
        var required = target.Split(':');
        var actual = architecture.Split(':');
        return required[0] == actual[0] && required.Skip(1).All(actual.Contains);
    }

    internal static int? DeviceIndex(string id) =>
        id.StartsWith("ROCm", StringComparison.OrdinalIgnoreCase) &&
        int.TryParse(id.AsSpan(4), out var index) && index >= 0 ? index : null;

    internal static IReadOnlyDictionary<int, (string Name, string Architecture)> ParseHipInfo(string output)
    {
        var result = new Dictionary<int, (string, string)>();
        int? index = null;
        string? name = null;
        foreach (var line in output.Split('\n'))
        {
            var match = HipInfoDeviceRegex().Match(line.Trim());
            if (match.Success)
            {
                index = int.TryParse(match.Groups[1].Value, out var ordinal) ? ordinal : null;
                name = null;
            }
            else if (line.TrimStart().StartsWith("Name:", StringComparison.Ordinal))
                name = line[(line.IndexOf(':') + 1)..].Trim();
            else if (line.TrimStart().StartsWith("gcnArchName:", StringComparison.Ordinal) &&
                     index is not null && name is not null)
            {
                var arch = line[(line.IndexOf(':') + 1)..].Trim();
                if (ArchitectureRegex().IsMatch(arch)) result[index.Value] = (name, arch);
            }
        }
        return result;
    }

    // Intersect targets across kernel bundles: a union could approve a GPU
    // for which only some of the runtime's kernels were compiled.
    internal static IReadOnlyList<string> ReadTargets(string binary)
    {
        HashSet<string>? targets = null;
        var directory = Path.GetDirectoryName(binary)!;
        var files = new[] { binary }.Concat(Directory.EnumerateFiles(directory, "ggml-*.dll")
            .Where(path => Path.GetFileName(path).StartsWith("ggml-cuda", StringComparison.OrdinalIgnoreCase) ||
                           Path.GetFileName(path).StartsWith("ggml-hip", StringComparison.OrdinalIgnoreCase)));
        foreach (var file in files)
        {
            using var stream = File.OpenRead(file);
            using var pe = new PEReader(stream);
            foreach (var section in pe.PEHeaders.SectionHeaders.Where(s =>
                         s.Name.StartsWith(".hip", StringComparison.Ordinal)))
            {
                if (section.SizeOfRawData > 256 * 1024 * 1024)
                    throw new BadImageFormatException("ROCm kernel section is too large to inspect.");
                var data = pe.GetSectionData(section.VirtualAddress)
                    .GetContent(0, Math.Min(section.VirtualSize, section.SizeOfRawData)).AsSpan();
                IntersectBundles(data, ref targets);
            }
        }
        return targets?.Order(StringComparer.Ordinal).ToArray() ?? [];
    }

    internal static IReadOnlyList<string> ReadBundleTargets(ReadOnlySpan<byte> data)
    {
        HashSet<string>? targets = null;
        IntersectBundles(data, ref targets);
        return targets?.Order(StringComparer.Ordinal).ToArray() ?? [];
    }

    private static void IntersectBundles(ReadOnlySpan<byte> data, ref HashSet<string>? targets)
    {
        var position = 0;
        while (position < data.Length)
        {
            var found = data[position..].IndexOf(BundleMagic);
            if (found < 0) break;
            position += found;
            var bundle = data[position..];
            var cursor = BundleMagic.Length;
            var count = ReadUInt64(bundle, ref cursor);
            if (count is 0 or > 4096) throw new BadImageFormatException("Invalid HIP bundle count.");
            var bundleTargets = new HashSet<string>(StringComparer.Ordinal);
            ulong end = (ulong)cursor;
            for (ulong i = 0; i < count; i++)
            {
                var offset = ReadUInt64(bundle, ref cursor);
                var size = ReadUInt64(bundle, ref cursor);
                var length = ReadUInt64(bundle, ref cursor);
                if (length > 4096 || length > (ulong)(bundle.Length - cursor) ||
                    offset > (ulong)bundle.Length || size > (ulong)bundle.Length - offset)
                    throw new BadImageFormatException("Invalid HIP bundle bounds.");
                var triple = Encoding.ASCII.GetString(bundle.Slice(cursor, (int)length));
                cursor += (int)length;
                end = Math.Max(end, offset + size);
                var match = HipTargetRegex().Match(triple);
                if (match.Success) bundleTargets.Add(match.Groups[1].Value);
                else if (triple.StartsWith("hip", StringComparison.Ordinal))
                    throw new BadImageFormatException("Unrecognized HIP kernel target.");
            }
            if (end < (ulong)cursor)
                throw new BadImageFormatException("HIP bundle payload overlaps its header.");
            if (bundleTargets.Count > 0)
            {
                if (targets is null) targets = bundleTargets;
                else targets.IntersectWith(bundleTargets);
            }
            position += (int)end;
        }
    }

    private static ulong ReadUInt64(ReadOnlySpan<byte> data, ref int cursor)
    {
        if (cursor > data.Length - sizeof(ulong))
            throw new BadImageFormatException("Truncated HIP bundle.");
        var value = BinaryPrimitives.ReadUInt64LittleEndian(data[cursor..]);
        cursor += sizeof(ulong);
        return value;
    }

    [GeneratedRegex(@"^device#\s+(\d+)$", RegexOptions.CultureInvariant)]
    private static partial Regex HipInfoDeviceRegex();

    [GeneratedRegex(@"^gfx[0-9a-f]+(?::(?:xnack|sramecc)[+-])*$", RegexOptions.CultureInvariant)]
    private static partial Regex ArchitectureRegex();

    [GeneratedRegex(@"^hip(?:v4)?-amdgcn-amd-amdhsa--(gfx[0-9a-f]+(?::(?:xnack|sramecc)[+-])*)$", RegexOptions.CultureInvariant)]
    private static partial Regex HipTargetRegex();
}
