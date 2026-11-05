// Logs both players' per-frame inputs while Strive plays a replay, for building a
// labelled dataset of the moves behind combo openers. Hooks and struct offsets
// come from WistfulHopes/GGSTReplayTakeover (MIT); the offsets were re-found for
// the 2026-09 Strive build (object manager +0xC38, analysers +0x3868 (P1) and +0x39E2 (P2), each the first of three,
// game frame +0x3E7C). Player objects (pointers at +0x928/+0x930): action name
// +0x38A8, facing +0x3AC (0 = right), x/y +0x3B0/+0x3B4, HP +0x1254, sprite name +0x12F8.
// +0x928 is an array of 102 object pointers: P1, P2, then a pool of 100 objects
// (projectiles, summons, effects; 0x6300 bytes apart, same layout as the players:
// +0x2C0 = the owner, a player or another object, +0x2C8 = the opponent).
// +0xD00 of a player = the object that last hit (or was blocked by) it: the attacking
// player for a direct hit, the pool object for projectiles, summons, spells.
//
// Box logging (off unless GGST_LOG_BOXES=1): how to read and transform the boxes is ported from
// AltimorTASDK/strivehitboxes (MIT, Copyright (c) 2021 Altimor); see BoxRecord below. Its offsets
// were for a 2021 build, so the 5.03 ones were re-checked against the running game.
// Entity (players and pool objects): +0x78 box array (type int, x, y, w, h floats), +0x114 19 box
// counts by type (hurt, hit, ..., [18] = follow-up), +0x1CC frames in the action, +0x288 hitstop,
// +0x320 attached entity, +0x3BC angle, +0x3CC/+0x3D0 scale, +0x398/+0x39C/+0x3A0 state flags,
// +0x7A8 throw range. Object manager: +0x910 active entity count, +0xC80 active entity list.
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
    private static readonly string LogDir = Setting("GGST_INPUT_LOG_DIR") ?? @"G:\Projects\replanal-data\ggst_logs";

    // Environment variables first. A game started through Steam inherits Steam's environment,
    // not the caller's, so %LOCALAPPDATA%\ggst_logger.env (NAME=value lines) can set them too.
    private static Dictionary<string, string>? _envFile;
    private static string? Setting(string name)
    {
        var v = Environment.GetEnvironmentVariable(name);
        if (v is not null) return v;
        if (_envFile is null)
        {
            _envFile = new Dictionary<string, string>();
            try
            {
                var f = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ggst_logger.env");
                if (File.Exists(f))
                    foreach (var line in File.ReadAllLines(f))
                        if (line.IndexOf('=') is > 0 and var eq)
                            _envFile[line[..eq].Trim()] = line[(eq + 1)..].Trim();
            }
            catch (Exception) { }
        }
        return _envFile.TryGetValue(name, out var w) ? w : null;
    }

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
                       + "p1_act,p1_face,p1_x,p1_y,p1_hp,p2_act,p2_face,p2_x,p2_y,p2_hp,p1_sprite,p2_sprite,"
                       + "objs,p1_hitby,p2_hitby" + (LogRefs ? ",p1_refs,p2_refs" : "") + (BoxesOn ? ",p1_boxes,p2_boxes,obj_boxes" : ""));
        _trace = new StreamWriter(Path.Combine(LogDir, $"calls_{DateTime.Now:yyyyMMdd_HHmmss}.csv")) { AutoFlush = true };
        _trace.WriteLine("game_frame,call,analyzer_off,rec_flag,ret");
        _traceRows = 0;
        _rows = 0;
        _lastBoxes.Clear();
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

    // Current sprite, e.g. "sol000_11": the first three letters are the character code.
    private static unsafe string Sprite(byte* p)
    {
        if (p == null || !Readable((IntPtr)(p + 0x12F8), 32)) return "";
        var spr = Marshal.PtrToStringAnsi((IntPtr)(p + 0x12F8), 32) ?? "";
        var nul = spr.IndexOf('\0');
        return nul >= 0 ? spr[..nul] : spr;
    }

    private const int Pool = 100, ObjSize = 0x6300, PlayerScan = 0x20000, HitByOff = 0xD00;
    // GGST_LOG_REFS=1 (set before starting the game) adds the research columns: every
    // field of the players and pool objects that points at one of the 102 objects.
    private static readonly bool LogRefs = Setting("GGST_LOG_REFS") == "1";
    private static readonly Dictionary<long, string> _refName = new();
    private static long _refLo, _refHi;

    // Names the 102 objects of the object array: P1, P2, o0..o99.
    private static unsafe void BuildRefs(BattleObjectManager* om)
    {
        _refName.Clear();
        _refLo = long.MaxValue;
        _refHi = 0;
        var arr = (long*)((byte*)om + 0x928);
        for (int i = 0; i < Pool + 2; i++)
        {
            long q = arr[i];
            if (q == 0) continue;
            _refName[q] = i == 0 ? "P1" : i == 1 ? "P2" : $"o{i - 2}";
            _refLo = Math.Min(_refLo, q);
            _refHi = Math.Max(_refHi, q);
        }
    }

    private static unsafe string HitBy(byte* p)
    {
        if (p == null || !Readable((IntPtr)(p + HitByOff), 8)) return "";
        return _refName.TryGetValue(*(long*)(p + HitByOff), out var n) ? n : "";
    }

    // The player an object belongs to: follows the owner field through parent objects.
    private static unsafe string Root(byte* o)
    {
        for (int k = 0; k < 6 && o != null; k++)
        {
            long q = *(long*)(o + 0x2C0);
            if (!_refName.TryGetValue(q, out var n)) return "";
            if (n[0] == 'P') return n;
            if (q == (long)o || !Readable((IntPtr)q, ObjSize)) return "";
            o = (byte*)q;
        }
        return "";
    }

    // Research column: every field of an object that points at one of the 102 objects,
    // as "offset:name" pairs (the owner / opponent fields at 0x2B8-0x2C8 left out).
    private static unsafe string Refs(byte* p, int size)
    {
        if (p == null || !Readable((IntPtr)p, size)) return "";
        var sb = new System.Text.StringBuilder();
        for (int off = 0; off < size; off += 8)
        {
            if (off >= 0x2B8 && off <= 0x2C8) continue;
            long q = *(long*)(p + off);
            if (q < _refLo || q > _refHi || !_refName.TryGetValue(q, out var n)) continue;
            if (sb.Length > 0) sb.Append(';');
            sb.Append(off.ToString("X")).Append(':').Append(n);
        }
        return sb.ToString();
    }

    // The pool objects with an action: "index/player/action/x/y/parent", joined by '|'
    // (player: the owner at the end of the owner chain; parent: the owning object when
    // the owner is another object). With LogRefs, "/refs" follows.
    private static unsafe string Objects(BattleObjectManager* om)
    {
        BuildRefs(om);
        var arr = (long*)((byte*)om + 0x928);
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < Pool; i++)
        {
            var o = (byte*)arr[i + 2];
            if (o == null || !Readable((IntPtr)o, ObjSize)) continue;
            var act = Marshal.PtrToStringAnsi((IntPtr)(o + 0x38A8), 32) ?? "";
            var nul = act.IndexOf('\0');
            if (nul >= 0) act = act[..nul];
            if (act.Length == 0) continue;
            _refName.TryGetValue(*(long*)(o + 0x2C0), out var parent);
            if (parent is not null && parent[0] == 'P') parent = "";
            if (sb.Length > 0) sb.Append('|');
            sb.Append(string.Create(CultureInfo.InvariantCulture,
                $"{i}/{Root(o)}/{act}/{*(int*)(o + 0x3B0)}/{*(int*)(o + 0x3B4)}/{parent}"));
            if (LogRefs) sb.Append('/').Append(Refs(o, ObjSize));
        }
        return sb.ToString();
    }

    // ---- Boxes (GGST_LOG_BOXES=1) ----
    private static readonly bool LogBoxes = Setting("GGST_LOG_BOXES") == "1";
    private static bool BoxesOn => LogBoxes && _boxFuncsFound;
    private static bool _boxFuncsFound;
    private static unsafe delegate* unmanaged<byte*, byte, int> _isActive;       // hits can connect now
    private static unsafe delegate* unmanaged<byte*, byte> _pushboxActive;
    private static unsafe delegate* unmanaged<byte*, int*, int*, int*, int*, void> _getPushbox;  // world units
    private const int MaxBoxes = 128, BoxBytes = 20, BoxTypes = 19, MaxAttached = 4;
    private static readonly Dictionary<string, string> _lastBoxes = new();

    // The array holds every box of the current sprite, also while the hit is not active:
    // flag 1 says whether it can connect. Local box units x scale = game units; sprites are
    // authored facing left with y down, so forward = -x * scale and up = -y * scale.
    private static unsafe string BoxList(byte* e, double sx, double sy, int angle)
    {
        long arr = *(long*)(e + 0x78);
        int total = 0;
        for (int i = 0; i < BoxTypes; i++)
        {
            int c = *(int*)(e + 0x114 + i * 4);
            if (c < 0 || c > MaxBoxes) return "";
            total += c;
        }
        if (total == 0 || total > MaxBoxes || !Readable((IntPtr)arr, total * BoxBytes)) return "";
        double rad = angle * Math.PI / 180000.0, cos = Math.Cos(rad), sin = Math.Sin(rad);
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < total; i++)
        {
            var b = (byte*)arr + i * BoxBytes;
            int type = *(int*)b;
            double x = *(float*)(b + 4), y = *(float*)(b + 8), w = *(float*)(b + 12), h = *(float*)(b + 16);
            double f0 = double.MaxValue, f1 = double.MinValue, u0 = double.MaxValue, u1 = double.MinValue;
            for (int k = 0; k < 4; k++)
            {
                double px = -(x + (k == 1 || k == 2 ? w : 0)) * sx, py = -(y + (k >= 2 ? h : 0)) * sy;
                double rx = angle == 0 ? px : px * cos - py * sin, ry = angle == 0 ? py : px * sin + py * cos;
                f0 = Math.Min(f0, rx); f1 = Math.Max(f1, rx);
                u0 = Math.Min(u0, ry); u1 = Math.Max(u1, ry);
            }
            if (sb.Length > 0) sb.Append('+');
            sb.Append(CultureInfo.InvariantCulture, $"{type}:{(long)f0}:{(long)u0}:{(long)f1}:{(long)u1}");
        }
        return sb.ToString();
    }

    // One entity: x:y:face:action_frames:hitstop:flags:scale_x:scale_y:angle:throw_range:
    // push_back:push_bottom:push_front:push_top;boxes. Boxes are type:back:bottom:front:top in game
    // units, relative to the entity's origin and facing (front > 0 = ahead of it), joined by '+',
    // or '=' when equal to this key's previous record. Type 0 = hurt, 1 = hit, others as the game
    // has them. Flags (hex): 1 hit active, 2 pushbox active, 4 counter-hit state, 8 strike invulnerable,
    // 10 throw invulnerable, 20 wake-up, 40 cinematic, 80 airborne, 100 landed a hit.
    // withPush = false for attached entities, which the overlay this is ported from never asks for one.
    private static unsafe string? BoxRecord(byte* e, string key, bool withPush, bool keepEmpty, bool? activeOf = null)
    {
        if (e == null || !Readable((IntPtr)e, ObjSize)) return keepEmpty ? "" : null;
        int x = *(int*)(e + 0x3B0), y = *(int*)(e + 0x3B4), face = *(int*)(e + 0x3AC);
        int sx = *(int*)(e + 0x3CC), sy = *(int*)(e + 0x3D0), angle = *(int*)(e + 0x3BC);
        var boxes = BoxList(e, sx, sy, angle);
        _lastBoxes.TryGetValue(key, out var last);
        if (boxes.Length == 0 && !keepEmpty)
        {
            _lastBoxes[key] = "";
            return null;
        }
        _lastBoxes[key] = boxes;
        uint st0 = *(uint*)(e + 0x398), st1 = *(uint*)(e + 0x39C), st2 = *(uint*)(e + 0x3A0), st3 = *(uint*)(e + 0x1B0);
        bool cine = (st3 & 0x4000000) != 0, isPlayer = *(byte*)(e + 0x20) != 0;
        bool active = activeOf ?? (cine || _isActive(e, (byte)(isPlayer ? 0 : 1)) != 0);
        int flags = (active ? 1 : 0) | (withPush && _pushboxActive(e) != 0 ? 2 : 0) | ((st0 & 0x100) != 0 ? 4 : 0)
                    | ((st1 & 0x10) != 0 || (st1 & 0x40) != 0 || *(int*)(e + 0x1220) > 0 ? 8 : 0)
                    | ((st1 & 0x20) != 0 || (st1 & 0x40) != 0 || *(int*)(e + 0x1218) > 0 ? 0x10 : 0)
                    | ((st1 & 0x40) != 0 ? 0x20 : 0) | (cine ? 0x40 : 0) | ((st0 & 1) != 0 ? 0x80 : 0) | ((st2 & 0x20) != 0 ? 0x100 : 0);
        int pb = 0, pt = 0, pf = 0, pu = 0;
        if (withPush)
        {
            int l = 0, t = 0, r = 0, b = 0;
            _getPushbox(e, &l, &t, &r, &b);
            pb = face == 0 ? l - x : x - r;
            pf = face == 0 ? r - x : x - l;
            pt = Math.Min(t, b) - y;
            pu = Math.Max(t, b) - y;
        }
        return string.Create(CultureInfo.InvariantCulture,
            $"{x}:{y}:{face}:{*(int*)(e + 0x1CC)}:{*(int*)(e + 0x288)}:{flags:x}:{sx}:{sy}:{angle}:{*(int*)(e + 0x7A8)}:{pb}:{pt}:{pf}:{pu};{(boxes.Length > 0 && boxes == last ? "=" : boxes)}");
    }

    // A player's or object's entities attached to it are drawn with the parent's hit-active state.
    private static unsafe void Attached(byte* e, string key, System.Text.StringBuilder sb)
    {
        if (!Readable((IntPtr)e, ObjSize)) return;
        bool active = (*(uint*)(e + 0x1B0) & 0x4000000) != 0 || _isActive(e, (byte)(*(byte*)(e + 0x20) != 0 ? 0 : 1)) != 0;
        var a = *(byte**)(e + 0x320);
        for (int k = 0; k < MaxAttached && a != null; k++)
        {
            var name = $"{key}a{k}";
            var rec = BoxRecord(a, name, false, false, active);
            if (rec is not null)
            {
                if (sb.Length > 0) sb.Append('|');
                sb.Append(name).Append('~').Append(rec);
            }
            if (!Readable((IntPtr)a, ObjSize)) break;
            a = *(byte**)(a + 0x320);
        }
    }

    // Active non-player entities as key~record joined by '|', key = pool name (o5), or attachedNN.
    private static unsafe string ObjectBoxes(BattleObjectManager* om)
    {
        var sb = new System.Text.StringBuilder();
        var ents = (long*)((byte*)om + 0xC80);
        int n = *(int*)((byte*)om + 0x910);
        if (n < 0 || n > 107) n = 0;
        for (int i = 0; i < n; i++)
        {
            long q = ents[i];
            if (q == 0 || q == (long)om->PlayerP1 || q == (long)om->PlayerP2) continue;
            if (!_refName.TryGetValue(q, out var key)) key = $"x{i}";
            var rec = BoxRecord((byte*)q, key, true, false);
            if (rec is not null)
            {
                if (sb.Length > 0) sb.Append('|');
                sb.Append(key).Append('~').Append(rec);
            }
            Attached((byte*)q, key, sb);
        }
        if (om->PlayerP1 != null) Attached(om->PlayerP1, "P1", sb);
        if (om->PlayerP2 != null) Attached(om->PlayerP2, "P2", sb);
        return sb.ToString();
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
            var line = string.Create(CultureInfo.InvariantCulture, $"{Stopwatch.GetTimestamp()},{Stopwatch.Frequency},{_calls},{dt:F5},{round},{frame},{_p1},{_p2},{Player(om->PlayerP1)},{Player(om->PlayerP2)},{Sprite(om->PlayerP1)},{Sprite(om->PlayerP2)},{Objects(om)},{HitBy(om->PlayerP1)},{HitBy(om->PlayerP2)}");
            if (LogRefs) line += $",{Refs(om->PlayerP1, PlayerScan)},{Refs(om->PlayerP2, PlayerScan)}";
            if (BoxesOn) line += $",{BoxRecord(om->PlayerP1, "P1", true, true)},{BoxRecord(om->PlayerP2, "P2", true, true)},{ObjectBoxes(om)}";
            _out!.WriteLine(line);
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
            if (LogBoxes)
            {
                // Read-only game functions, called from the battle hook on the game thread.
                var active = Hooking.SigScan("40 53 48 83 EC 20 F6 81 70 07 00 00 01 48 8B D9 74 1D F7 81 B0 01");
                var pushActive = Hooking.SigScan("40 53 48 83 EC 20 48 8B C1 48 8B D9 33 C9 38 48 20 48 0F 44 C1");
                var getPush = Hooking.SigScan("99 48 8B CB 2B C2 D1 F8 44");   // inside the function, as in strivehitboxes
                Debug.Log(LogLevel.Default, $"[InputLogger] boxes: IsActive={active:X} PushboxActive={pushActive:X} GetPushbox={getPush:X}");
                if (active != 0 && pushActive != 0 && getPush != 0)
                {
                    _isActive = (delegate* unmanaged<byte*, byte, int>)active;
                    _pushboxActive = (delegate* unmanaged<byte*, byte>)pushActive;
                    _getPushbox = (delegate* unmanaged<byte*, int*, int*, int*, int*, void>)(getPush - 0x5B);
                    _boxFuncsFound = true;
                }
                else Debug.Log(LogLevel.Warning, "[InputLogger] a box signature was not found; boxes not logged");
            }
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
