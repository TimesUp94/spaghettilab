// Logs both players' per-frame inputs while Strive plays a replay, for building a
// labelled dataset of the moves behind combo openers. Hooks and struct offsets
// come from WistfulHopes/GGSTReplayTakeover (MIT); the offsets were re-found for
// the 2026-09 Strive build (object manager +0xC38, analysers +0x3868 (P1) and +0x39E2 (P2), each the first of three,
// game frame +0x3E7C). Player objects (pointers at +0x928/+0x930): action name
// +0x38A8, facing +0x3AC (0 = right), x/y +0x3B0/+0x3B4, HP +0x1254.
using System.Globalization;
using Stopwatch = System.Diagnostics.Stopwatch;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using UE4SSDotNetFramework.Framework;

namespace GGSTInputLogger.Common;

[StructLayout(LayoutKind.Explicit)]
public struct BattleObjectManager
{
    [FieldOffset(0x3868)] public byte InputAnalyzerP1;
    [FieldOffset(0x39E2)] public byte InputAnalyzerP2;
    [FieldOffset(0x3E7C)] public int GameFrame;
    [FieldOffset(0x928)] public unsafe byte* PlayerP1;
    [FieldOffset(0x930)] public unsafe byte* PlayerP2;
}

[StructLayout(LayoutKind.Explicit)]
public struct BattleState
{
    [FieldOffset(0x118)] public int RoundCount;
}

[StructLayout(LayoutKind.Explicit)]
public struct RedGameStateBattle
{
    [FieldOffset(0xC38)] public unsafe BattleObjectManager* ObjectManager;
    [FieldOffset(0xC50)] public unsafe BattleState* BattleState;
}

public static class Main
{
    private const int ModeReplay = 0x17, ModeTraining = 0x8;
    private static readonly string LogDir = Environment.GetEnvironmentVariable("GGST_INPUT_LOG_DIR")
                                            ?? @"G:\Projects\replanal-data\ggst_logs";

    unsafe delegate void UpdateBattleFunc(RedGameStateBattle* @this, float dt);
    unsafe delegate void UpdateBattleInputFunc(byte* analyzer, ushort recFlag);
    delegate byte GetGameModeFunc(IntPtr @this);

    private static IntPtr _battleHook, _inputHook;
    private static UpdateBattleFunc? _battleOrig, _battleThunk;
    private static UpdateBattleInputFunc? _inputOrig, _inputThunk;
    private static unsafe delegate* unmanaged[Cdecl]<RedGameStateBattle*, float, void> _battlePtr;
    private static unsafe delegate* unmanaged[Cdecl]<byte*, ushort, void> _inputPtr;
    private static GetGameModeFunc? _getGameMode;
    private static ObjectReference? _gameCommon;

    private static unsafe RedGameStateBattle* _state;
    private static StreamWriter? _out;
    private static int _p1 = -1, _p2 = -1, _lastFrame = int.MinValue, _lastRound = -1;
    private static long _rows, _calls;
    private static int _nullLogged;
    private static long _an1, _an2, _inputCalls;
    private static int _r1 = -1, _r2 = -1;
    private static int _diagFrames;
    private static StreamWriter? _diag, _trace;
    private static long _traceRows;
    private static unsafe long st0() => _state != null && _state->ObjectManager != null ? (long)_state->ObjectManager : 0;
    private static IntPtr ReturnAddr() => IntPtr.Zero;

    private static bool InReplay()
    {
        _gameCommon ??= ObjectReference.Find("REDGameCommon");
        if (_gameCommon is null || _gameCommon.Pointer == 0 || _getGameMode is null) return false;
        var mode = _getGameMode(_gameCommon.Pointer);
        return mode == ModeReplay || mode == ModeTraining;
    }

    private static void Open()
    {
        Directory.CreateDirectory(LogDir);
        var path = Path.Combine(LogDir, $"inputs_{DateTime.Now:yyyyMMdd_HHmmss}.csv");
        _out = new StreamWriter(path) { AutoFlush = true };
        _out.WriteLine("qpc,qpc_freq,call,dt,round,game_frame,p1_rec,p2_rec,"
                       + "p1_act,p1_face,p1_x,p1_y,p1_hp,p2_act,p2_face,p2_x,p2_y,p2_hp");
        _trace = new StreamWriter(Path.Combine(LogDir, $"calls_{DateTime.Now:yyyyMMdd_HHmmss}.csv")) { AutoFlush = true };
        _trace.WriteLine("game_frame,call,analyzer_off,rec_flag,ret");
        _traceRows = 0;
        _rows = 0;
        Debug.Log(LogLevel.Default, $"[InputLogger] logging to {path}");
    }

    private static unsafe string Player(byte* p)
    {
        if (p == null || !Readable((IntPtr)p, 0x3900)) return ",,,,";
        var act = Marshal.PtrToStringAnsi((IntPtr)(p + 0x38A8), 32) ?? "";
        var nul = act.IndexOf('\0');
        if (nul >= 0) act = act[..nul];
        return string.Create(CultureInfo.InvariantCulture,
            $"{act},{*(int*)(p + 0x3AC)},{*(int*)(p + 0x3B0)},{*(int*)(p + 0x3B4)},{*(int*)(p + 0x1254)}");
    }

    private static void Close()
    {
        if (_out is null) return;
        _out.Flush();
        _out.Dispose();
        _trace?.Dispose();
        _trace = null;
        Debug.Log(LogLevel.Default, $"[InputLogger] closed after {_rows} rows");
        _out = null;
    }

    [DllImport("user32.dll")]
    private static extern short GetAsyncKeyState(int key);

    private const int VkF6 = 0x75, VkF7 = 0x76;
    private static int _dumpLeft, _dumpSize;
    private static BinaryWriter? _dump;
    private static List<long>? _dumpTargets;
    private static bool _f6Down, _f7Down;

    // Memory snapshots for finding fields: F6 dumps one large frame (object manager
    // plus every readable object it points to), F7 dumps the next 30 frames of the
    // same objects at a smaller size. The label comes from dump_label.txt.
    private static unsafe void Snapshots(RedGameStateBattle* state)
    {
        var om = (long)state->ObjectManager;
        if (om == 0) return;
        bool f6 = (GetAsyncKeyState(VkF6) & 0x8000) != 0, f7 = (GetAsyncKeyState(VkF7) & 0x8000) != 0;
        if (_dumpLeft == 0 && ((f6 && !_f6Down) || (f7 && !_f7Down)))
        {
            var labelFile = Path.Combine(LogDir, "dump_label.txt");
            var label = File.Exists(labelFile) ? File.ReadAllText(labelFile).Trim() : "x";
            Directory.CreateDirectory(LogDir);
            _dump = new BinaryWriter(File.Create(Path.Combine(LogDir, $"dump_{label}_{DateTime.Now:HHmmss}.bin")));
            _dumpLeft = f6 ? 1 : 30;
            _dumpSize = f6 ? 0x40000 : 0x8000;
            _dumpTargets = new List<long> { om };
            for (int off = 0; off < 0x10000 && _dumpTargets.Count < 96; off += 8)
            {
                if (!Readable((IntPtr)(om + off), 8)) break;
                long q = *(long*)(om + off);
                if ((q & 7) == 0 && (q < om || q >= om + 0x40000) && !_dumpTargets.Contains(q) && Readable((IntPtr)q, 0x100))
                    _dumpTargets.Add(q);
            }
            Debug.Log(LogLevel.Default, $"[InputLogger] dump '{label}': {_dumpTargets.Count} objects, {_dumpLeft} frame(s)");
        }
        _f6Down = f6;
        _f7Down = f7;
        if (_dumpLeft == 0 || _dump is null || _dumpTargets is null) return;
        _dump.Write(state->ObjectManager->GameFrame);
        _dump.Write(_dumpTargets.Count);
        foreach (var t in _dumpTargets)
        {
            int n = _dumpSize;
            while (n > 0x100 && !Readable((IntPtr)t, n)) n /= 2;
            _dump.Write(t);
            _dump.Write(n);
            _dump.Write(new ReadOnlySpan<byte>((void*)t, n));
        }
        if (--_dumpLeft == 0)
        {
            _dump.Dispose();
            _dump = null;
        }
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe void BattleHook(RedGameStateBattle* @this, float dt)
    {
        _state = @this;
        _battleOrig!(@this, dt);
        try
        {
            if (@this->ObjectManager != null) Snapshots(@this);
            if (!InReplay())
            {
                Close();
                return;
            }
            if (_out is null) Open();
            var om = @this->ObjectManager;
            var bs = @this->BattleState;
            if (om == null)
            {
                if (_nullLogged++ < 20)
                    Debug.Log(LogLevel.Warning, $"[InputLogger] state={(IntPtr)@this:X} om={(IntPtr)om:X} bs={(IntPtr)bs:X}");
                return;
            }
            // The battle-state offset is unconfirmed for this build, so only read
            // through it when the pointer lands in committed, readable memory.
            int frame = om->GameFrame, round = Readable((IntPtr)bs, 0x120) ? bs->RoundCount : -1;
            // The replay list also counts as replay mode, so a second replay never
            // re-enters the mode: start a new file when the frame counter restarts.
            if (_rows > 0 && frame < _lastFrame - 60)
            {
                Close();
                Open();
            }
            _calls++;
            if (frame == _lastFrame && round == _lastRound) return;
            _lastFrame = frame;
            _lastRound = round;
            _out!.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{Stopwatch.GetTimestamp()},{Stopwatch.Frequency},{_calls},{dt:F5},{round},{frame},{_p1},{_p2},{Player(om->PlayerP1)},{Player(om->PlayerP2)}"));
            _rows++;
        }
        catch (Exception e)
        {
            Debug.Log(LogLevel.Error, $"[InputLogger] {e}");
        }
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe void InputHook(byte* analyzer, ushort recFlag)
    {
        var a = (long)analyzer;
        if (_an1 == 0) _an1 = a;
        else if (a != _an1 && _an2 == 0) _an2 = a;
        if (a == _an1) _r1 = recFlag; else if (a == _an2) _r2 = recFlag;
        _inputCalls++;
        if (_trace is not null && st0() is var om0 && om0 != 0 && _traceRows < 600 * 6)
        {
            _traceRows++;
            _trace.WriteLine($"{_lastFrame},{_inputCalls},{a - om0:X},{recFlag},{(long)ReturnAddr():X}");
        }
        var st = _state;
        if (st != null && st->ObjectManager != null)
        {
            var om = st->ObjectManager;
            if (analyzer == &om->InputAnalyzerP1) _p1 = recFlag;
            else if (analyzer == &om->InputAnalyzerP2) _p2 = recFlag;
        }
        _inputOrig!(analyzer, recFlag);
    }

    // Finds where today's build keeps the object manager: logs every pointer in the
    // battle state that lands up to 64 KB below the input analysers, then dumps the
    // ints around the old frame-counter offset for 240 frames.
    private static unsafe void Diagnose(byte* state)
    {
        if (_an1 == 0 || _an2 == 0 || _diagFrames >= 240) return;
        if (_diag is null)
        {
            _diag = new StreamWriter(Path.Combine(LogDir, $"diag_{DateTime.Now:yyyyMMdd_HHmmss}.txt")) { AutoFlush = true };
            long lo = Math.Min(_an1, _an2);
            _diag.WriteLine($"state={(long)state:X} an1={_an1:X} an2={_an2:X} an_delta={_an2 - _an1:X}");
            for (int off = 0; off < 0x1000; off += 8)
            {
                long q = *(long*)(state + off);
                if (q > lo - 0x10000 && q <= lo) _diag.WriteLine($"cand off={off:X} ptr={q:X} below_an={lo - q:X}");
            }
        }
        long baseP = Math.Min(_an1, _an2) - 0x37A8;
        var sb = new System.Text.StringBuilder();
        sb.Append($"f{_diagFrames} calls={_inputCalls} r1={_r1} r2={_r2} |");
        for (int off = 0x3C00; off < 0x3F00; off += 4) sb.Append($" {*(int*)(baseP + off)}");
        _diag.WriteLine(sb.ToString());
        if (++_diagFrames == 240) { _diag.Dispose(); _diag = null; }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemInfo
    {
        public IntPtr BaseAddress, AllocationBase;
        public uint AllocationProtect;
        public IntPtr RegionSize;
        public uint State, Protect, Type;
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr VirtualQuery(IntPtr address, out MemInfo info, IntPtr length);

    private static bool Readable(IntPtr p, int size)
    {
        if (p == IntPtr.Zero) return false;
        if (VirtualQuery(p, out var mi, (IntPtr)Marshal.SizeOf<MemInfo>()) == IntPtr.Zero) return false;
        const uint commit = 0x1000, readable = 0x02 | 0x04 | 0x20 | 0x40;
        return mi.State == commit && (mi.Protect & readable) != 0
               && (long)p + size <= (long)mi.BaseAddress + (long)mi.RegionSize;
    }

    public static void StartMod()
    {
        var battle = Hooking.SigScan("40 53 57 41 54 41 55 48 81 EC 88 00 00 00");
        var input = Hooking.SigScan("0F B7 41 02 44 0F B7 D2");
        var mode = Hooking.SigScan("0F B6 81 F0 02 00 00 C3");
        Debug.Log(LogLevel.Default, $"[InputLogger] UpdateBattle={battle:X} UpdateBattleInput={input:X} GetGameMode={mode:X}");
        if (battle == 0 || input == 0 || mode == 0)
        {
            Debug.Log(LogLevel.Warning, "[InputLogger] a signature was not found; not hooking");
            return;
        }
        _getGameMode = Marshal.GetDelegateForFunctionPointer<GetGameModeFunc>(mode);
        unsafe
        {
            // The delegates handed to native code must stay referenced for the
            // life of the hook, or the GC frees them and the game crashes.
            IntPtr orig = 0;
            _battlePtr = &BattleHook;
            _battleThunk = (s, f) => _battlePtr(s, f);
            _battleHook = Hooking.Hook(battle, Marshal.GetFunctionPointerForDelegate(_battleThunk), ref orig);
            _battleOrig = Marshal.GetDelegateForFunctionPointer<UpdateBattleFunc>(orig);

            orig = 0;
            _inputPtr = &InputHook;
            _inputThunk = (a, r) => _inputPtr(a, r);
            _inputHook = Hooking.Hook(input, Marshal.GetFunctionPointerForDelegate(_inputThunk), ref orig);
            _inputOrig = Marshal.GetDelegateForFunctionPointer<UpdateBattleInputFunc>(orig);
        }
    }

    public static void StopMod()
    {
        Hooking.Unhook(_battleHook);
        Hooking.Unhook(_inputHook);
        Close();
    }
}
