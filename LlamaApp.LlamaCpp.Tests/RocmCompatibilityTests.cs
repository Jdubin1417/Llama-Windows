using System.Diagnostics;
using System.Buffers.Binary;
using System.Reflection.PortableExecutable;
using System.Text;
using System.Text.Json;
using LlamaApp.Llama;
using Xunit;

namespace LlamaApp.LlamaCpp.Tests;

public sealed class RocmCompatibilityTests
{
    private static byte[] Bundle(params string[] architectures)
    {
        var triples = architectures.Select(arch => Encoding.ASCII.GetBytes("hip-amdgcn-amd-amdhsa--" + arch)).ToArray();
        var headerSize = 24 + 8 + triples.Sum(triple => 24 + triple.Length);
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write(Encoding.ASCII.GetBytes("__CLANG_OFFLOAD_BUNDLE__"));
        writer.Write((ulong)triples.Length);
        foreach (var triple in triples)
        {
            writer.Write((ulong)headerSize);
            writer.Write(0UL);
            writer.Write((ulong)triple.Length);
            writer.Write(triple);
        }
        return stream.ToArray();
    }

    [Fact]
    public void ReadsSingleArchitectureAndMultiArchitectureBundles()
    {
        Assert.Equal(["gfx1151"], RocmCompatibility.ReadBundleTargets(Bundle("gfx1151")));
        Assert.Equal(["gfx1151", "gfx1200"], RocmCompatibility.ReadBundleTargets(Bundle("gfx1200", "gfx1151")));
    }

    [Fact]
    public void AllKernelsMustSupportTheArchitecture()
    {
        var data = Bundle("gfx1151", "gfx1200").Concat(new byte[8]).Concat(Bundle("gfx1151")).ToArray();
        Assert.Equal(["gfx1151"], RocmCompatibility.ReadBundleTargets(data));
        Assert.Empty(RocmCompatibility.ReadBundleTargets(Bundle("gfx1151").Concat(Bundle("gfx1200")).ToArray()));
    }

    [Fact]
    public void IgnoresOrdinaryArchitectureStringsOutsideBundles()
        => Assert.Empty(RocmCompatibility.ReadBundleTargets(Encoding.ASCII.GetBytes("some text gfx1200 gfx1151")));

    [Fact]
    public void TruncatedOrOutOfBoundsBundlesAreRejected()
    {
        Assert.Throws<BadImageFormatException>(() => RocmCompatibility.ReadBundleTargets(Bundle("gfx1151")[..30]));
        var bytes = Bundle("gfx1151");
        Array.Fill(bytes, (byte)255, 32, 8);
        Assert.Throws<BadImageFormatException>(() => RocmCompatibility.ReadBundleTargets(bytes));
    }

    [Fact]
    public void ReadsKernelTargetsFromPeSectionsAndBackendDlls()
    {
        var directory = Path.Combine(Path.GetTempPath(), "llama-rocm-pe-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            static void WriteFixture(string path, byte[] data)
            {
                var image = File.ReadAllBytes(typeof(RocmCompatibilityTests).Assembly.Location);
                using var pe = new PEReader(new MemoryStream(image));
                var sectionHeader = pe.PEHeaders.CoffHeaderStartOffset + 20 + pe.PEHeaders.CoffHeader.SizeOfOptionalHeader;
                var directoryOffset = pe.PEHeaders.PEHeader!.Magic == PEMagic.PE32Plus ? 112 : 96;
                image.AsSpan(pe.PEHeaders.CoffHeaderStartOffset + 20 + directoryOffset + 14 * 8, 8).Clear();
                var name = image.AsSpan(sectionHeader, 8);
                name.Clear();
                ".hip_fat"u8.CopyTo(name);
                BinaryPrimitives.WriteInt32LittleEndian(image.AsSpan(sectionHeader + 8), data.Length);
                BinaryPrimitives.WriteInt32LittleEndian(image.AsSpan(sectionHeader + 16), data.Length);
                BinaryPrimitives.WriteInt32LittleEndian(image.AsSpan(sectionHeader + 20), image.Length);
                File.WriteAllBytes(path, image.Concat(data).ToArray());
            }

            var binary = Path.Combine(directory, "llama.exe");
            WriteFixture(binary, Bundle("gfx1151", "gfx1200"));
            Assert.Equal(["gfx1151", "gfx1200"], RocmCompatibility.ReadTargets(binary));
            WriteFixture(Path.Combine(directory, "ggml-hip.dll"), Bundle("gfx1151"));
            Assert.Equal(["gfx1151"], RocmCompatibility.ReadTargets(binary));
            WriteFixture(Path.Combine(directory, "ggml-hip.dll"), Bundle("gfx1200"));
            Assert.Equal(["gfx1200"], RocmCompatibility.ReadTargets(binary));
        }
        finally
        {
            File.Delete(Path.Combine(directory, "llama.exe"));
            File.Delete(Path.Combine(directory, "ggml-hip.dll"));
            Directory.Delete(directory);
        }
    }

    [Theory]
    [InlineData("gfx1151", "gfx1151", true)]
    [InlineData("gfx1151", "gfx1200", false)]
    [InlineData("gfx90a", "gfx90a:xnack+", true)]
    [InlineData("gfx90a:xnack+", "gfx90a:xnack-", false)]
    [InlineData("gfx90a:xnack+", "gfx90a", false)]
    [InlineData("gfx90a:xnack+:sramecc-", "gfx90a:sramecc-:xnack+", true)]
    public void TargetFeaturesMustAlsoBeCompatible(string target, string architecture, bool expected)
        => Assert.Equal(expected, RocmCompatibility.Supports(target, architecture));

    [Fact]
    public void ParsesHipInfoByOrdinalNotByGpuModelName()
    {
        var hardware = RocmCompatibility.ParseHipInfo("""
            device#                           0
            Name:                             Radeon
            gcnArchName:                      gfx1200
            peers:                            device#1
            device#                           1
            Name:                             Radeon
            gcnArchName:                      gfx1151
            """);
        Assert.Equal(("Radeon", "gfx1200"), hardware[0]);
        Assert.Equal(("Radeon", "gfx1151"), hardware[1]);
        Assert.Empty(RocmCompatibility.ParseHipInfo("device# 0\nName: Radeon\ngcnArchName: not-an-architecture"));
    }

    [Fact]
    public void UnknownOrIncompatibleArchitecturesHaveActionableErrors()
    {
        Assert.Null(RocmCompatibility.CompatibilityError("Radeon", "gfx1151", ["gfx1151"]));
        Assert.Contains("gfx1200", RocmCompatibility.CompatibilityError("Radeon", "gfx1200", ["gfx1151"]));
        Assert.Contains("Vulkan/CPU", RocmCompatibility.CompatibilityError("Radeon", "gfx1151", []));
        Assert.Contains("hipInfo.exe", RocmCompatibility.CompatibilityError("Radeon", null, ["gfx1151"]));
    }

    private static DeviceProbe MixedCards => new(true,
    [
        new LlamaDevice
        {
            Id = "ROCm0", Name = "Discrete Radeon", Kind = DeviceKind.Rocm, Architecture = "gfx1200",
            CompatibilityError = "Discrete Radeon (gfx1200) is incompatible with the gfx1151 build.",
        },
        new LlamaDevice { Id = "ROCm1", Name = "Integrated Radeon", Kind = DeviceKind.Rocm, Architecture = "gfx1151" },
    ]);

    [Theory]
    [InlineData("auto")]
    [InlineData("rocm")]
    public void AutomaticSelectionExcludesIncompatibleDetectedGpus(string backend)
    {
        var selection = InferenceRuntime.SelectDevice(backend, "", MixedCards);
        Assert.Equal("ROCm1", selection.DeviceArgument);
        Assert.Null(selection.Error);
        Assert.NotNull(InferenceRuntime.SelectDevice(backend, "Discrete Radeon", MixedCards).Error);
    }

    [Fact]
    public void NoCompatibleGpuIsAnErrorNotAnUnexplainedCpuFallback()
    {
        var selection = InferenceRuntime.SelectDevice("rocm", "", new DeviceProbe(true, [MixedCards.Devices[0]]));
        Assert.NotNull(selection.Error);
        Assert.Null(selection.FallbackReason);
        Assert.Null(InferenceRuntime.SelectDevice("cpu", "", MixedCards).Error);
    }

    [Fact]
    public void ProcessMaskAndGlobalAndModelIdsUseTheSameRemapping()
    {
        var probe = MixedCards;
        var isolation = RocmDeviceIsolation.Create(probe)!;
        Assert.Equal("1", isolation.VisibleDevices);
        Assert.Equal("ROCm0", isolation.MapArgument("ROCm1"));
        Assert.Equal("none", isolation.MapArgument("ROCm0"));
        Assert.Equal("none", isolation.MapArgument("ROCm1,ROCm0"));
        Assert.Equal("none", isolation.MapArgument("none"));
        Assert.Null(isolation.MapArgument(null));
        var ids = probe.Devices.ToDictionary(d => GpuDeviceChoice.Key(d, probe.Devices), d => isolation.MapArgument(d.Id)!);
        var preset = ModelPresets.Render(null, new Dictionary<string, ModelPromptProcessingProfile>
        {
            ["supported"] = new() { GpuDeviceName = "Integrated Radeon", GpuLayers = "all" },
            ["unsupported"] = new() { GpuDeviceName = "Discrete Radeon", GpuLayers = "all" },
            ["cpu"] = new() { GpuDeviceName = "none" },
        }, new(), ids, isolation.MapArgument("ROCm1"))!;
        Assert.Contains("[*]\ndevice = ROCm0", preset);
        Assert.Contains("[supported]\ndevice = ROCm0\nn-gpu-layers = all", preset);
        Assert.Contains("[unsupported]\ndevice = none\nn-gpu-layers = 0", preset);
        Assert.Contains("[cpu]\ndevice = none\nn-gpu-layers = 0", preset);
    }

    [Fact]
    public void PreservesInheritedVisibilityAndDuplicateGpuChoiceKeys()
    {
        var probe = new DeviceProbe(true,
        [
            MixedCards.Devices[1] with { Id = "ROCm0", Name = "Radeon" },
            MixedCards.Devices[1] with { Id = "ROCm1", Name = "Radeon" },
        ]);
        var isolation = RocmDeviceIsolation.Create(probe, "3, 7")!;
        Assert.Equal("3,7", isolation.VisibleDevices);
        Assert.Equal("ROCm1", isolation.MapArgument(GpuDeviceChoice.Resolve(
            GpuDeviceChoice.Key(probe.Devices[1], probe.Devices), probe.Devices)!.Id));
        Assert.Throws<InvalidOperationException>(() => RocmDeviceIsolation.Create(probe, "3"));
        Assert.Equal("-1", RocmDeviceIsolation.Create(new DeviceProbe(true, [MixedCards.Devices[0]]))!.VisibleDevices);
        Assert.Null(RocmDeviceIsolation.Create(new DeviceProbe(true, [])));
    }

    [Fact]
    public void ApplyingIsolationDoesNotChangeTheAppsEnvironment()
    {
        var before = Environment.GetEnvironmentVariable("HIP_VISIBLE_DEVICES");
        var psi = new ProcessStartInfo();
        RocmDeviceIsolation.Create(MixedCards)!.Apply(psi);
        Assert.Equal("1", psi.Environment["HIP_VISIBLE_DEVICES"]);
        Assert.Equal(before, Environment.GetEnvironmentVariable("HIP_VISIBLE_DEVICES"));
    }

    [Fact]
    public void RunningServerRequiresStableDeviceOrderButNotStableFreeMemory()
    {
        var probe = MixedCards;
        Assert.True(RocmDeviceIsolation.SameDevices(probe, new DeviceProbe(true,
            probe.Devices.Select(d => d with { FreeBytes = 100 }).ToArray())));
        Assert.False(RocmDeviceIsolation.SameDevices(probe, new DeviceProbe(true,
            [probe.Devices[0] with { Id = "ROCm1" }, probe.Devices[1] with { Id = "ROCm0" }])));
        Assert.False(RocmDeviceIsolation.SameDevices(probe, new DeviceProbe(true, [probe.Devices[1]])));
        Assert.False(RocmDeviceIsolation.SameDevices(probe, new DeviceProbe(false, [])));
    }

    [Fact]
    public void AdoptionStateRoundTripsAndRejectsReusedPidsChangedBinariesOrDevices()
    {
        var probe = MixedCards;
        var isolation = RocmDeviceIsolation.Create(probe)!;
        var saved = new RocmServerState(123, 456, "llama.exe", 789, isolation, "ROCm1",
            probe.Devices.ToDictionary(d => d.Id, d => d.Name), "rocm", "Integrated Radeon");
        var restored = JsonSerializer.Deserialize<RocmServerState>(JsonSerializer.Serialize(saved))!;
        Assert.True(restored.Matches(123, 456, "llama.exe", 789, isolation, probe, "rocm", "Integrated Radeon"));
        Assert.False(restored.Matches(124, 456, "llama.exe", 789, isolation, probe, "rocm", "Integrated Radeon"));
        Assert.False(restored.Matches(123, 457, "llama.exe", 789, isolation, probe, "rocm", "Integrated Radeon"));
        Assert.False(restored.Matches(123, 456, "llama.exe", 790, isolation, probe, "rocm", "Integrated Radeon"));
        Assert.False(restored.Matches(123, 456, "other.exe", 789, isolation, probe, "rocm", "Integrated Radeon"));
        Assert.False(restored.Matches(123, 456, "llama.exe", 789, isolation, probe, "rocm", "Discrete Radeon"));
        Assert.False(restored.Matches(123, 456, "llama.exe", 789, isolation,
            new DeviceProbe(true, [probe.Devices[0] with { Name = "Changed" }, probe.Devices[1]]),
            "rocm", "Integrated Radeon"));
    }
}
