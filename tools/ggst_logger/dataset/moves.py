"""GGST character identification and move naming from internal action names (logger *_act).

- `infer_character(acts)`: character from the set of actions a player used in a replay
  (3-letter / lowercase prefixes, else signature actions). A `char` value logged by the
  logger (column `p1_char` / `p2_char`, name or numeric id) overrides it: `resolve_character`.
- `move_info(char, act)`: (notation, name, category, flags) for an action.
- `classify(row)`: clean move / category for one opener row of build.py.

Sources: Dustloop GGST move data (wiki GGST/<Char>/Data, special/super order and inputs),
checked on the frame strips of ggst_dataset/strips where marked `verified`. Entries marked
`inferred` follow the Dustloop order / the move's variants but were not seen on video.

Categories: normal, command normal, special, super, throw, projectile, burst, other.
"""
from __future__ import annotations

import collections
import re

# ---------------------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------------------

# Action-name prefixes (case-sensitive) -> character.
CHAR_CODES = {
    "SOL": "Sol", "KYK": "Ky", "MAY": "May", "AXL": "Axl", "CHP": "Chipp", "POT": "Potemkin",
    "FAU": "Faust", "MLL": "Millia", "ZAT": "Zato-1", "RAM": "Ramlethal", "LEO": "Leo",
    "NAG": "Nagoriyuki", "GIO": "Giovanna", "ANJ": "Anji", "INO": "I-No", "GLD": "Goldlewis",
    "JKO": "Jack-O", "COS": "Happy Chaos", "BKN": "Baiken", "TST": "Testament",
    "BGT": "Bridget", "SIN": "Sin", "BED": "Bedman?", "ASK": "Asuka", "JHN": "Johnny",
    "ELP": "Elphelt", "ABA": "A.B.A", "SLY": "Slayer", "sly": "Slayer", "DZY": "Dizzy",
    "VEN": "Venom", "UNI": "Unika", "LUC": "Lucy",
}

# Game roster order (character select / internal id), for the logger's future char field.
# UNVERIFIED: confirm the numbering against the memory value when the field is added.
CHAR_IDS = ["Sol", "Ky", "May", "Axl", "Chipp", "Potemkin", "Faust", "Millia", "Zato-1",
            "Ramlethal", "Leo", "Nagoriyuki", "Giovanna", "Anji", "I-No", "Goldlewis", "Jack-O",
            "Happy Chaos", "Baiken", "Testament", "Bridget", "Sin", "Bedman?", "Asuka", "Johnny",
            "Elphelt", "A.B.A", "Slayer", "Dizzy", "Venom", "Unika", "Lucy"]

# Characters whose specials carry no prefix: distinctive action names (no shared names such as
# HomingJump / DustFinish / WorldSideBreak_*). Every set was checked against Dustloop and the
# strips (Giovanna = "GIOVANNA" on the HUD, Slayer = "SLAYER").
CHAR_SIGNATURES = {
    "Asuka": {"Special_MP", "Magic", "Magic_air", "PhysicalLaw", "Draw_or_Discard",
              "Draw_or_Discard_air", "Change_deck", "Change_deck_air", "fuwafuwa"},
    "Leo": {"EisenSturmC", "EisenSturmD", "ErstWind", "ZweitWind", "GraviertWurdeC", "GraviertWurdeD",
            "Turbulenz", "GlaenzenDunkel", "StahlWirbel", "LeidenschaftDirigent", "Semuke"},
    "Sol": {"BanditBringer_Land", "BanditBringer_Air", "BanditRevolver_Land", "Fefnir", "GunFlame",
            "GunFlameFeint", "VolcanicViperLandS", "VolcanicViperLandHS", "VolcanicViperAirS",
            "NewSidewinder", "Vortex", "BukkirabouNiNageru", "BukkiraExe"},
    "Ky": {"StunEdge1", "StunEdge2", "AirStunEdge1", "StunDipper", "FadulArc", "Diaekura",
           "VaporThrustC", "VaporThrustD", "AirVaporThrustD", "SacredEdge", "DragonInstall", "Lightning"},
    "Chipp": {"AlphaBlade", "AlphaBlade_yoko", "AlphaBlade_Cancel", "AirAlphaBlade_yoko",
              "AirAlphaBlade_Cancel", "GammaBlade", "BankiMessai", "Shuriken", "Sushi", "Sukiyaki",
              "Banzai", "ItoAtk", "ThreadRun_Jump"},
    "Ramlethal": {"BajonetoC", "BajonetoD", "Dauro", "Erarurumo1", "Erarurumo2", "Erarurumo3",
                  "AgresaOrdono", "Morutobato", "Ondo", "Sabrobato", "ShirudoLand", "Falstelo"},
    "Potemkin": {"HammerFall", "HammerFallBrake", "HeatKnucle", "HeatTackle", "FDB", "PotemkinBuster",
                 "PotemkinBusterExe", "HeavenlyPBuster", "MegaFistFront", "MegaFistBack", "SlideHead"},
    "Nagoriyuki": {"Bloodsucking_Universe", "Bloodsucking_UniverseExe", "Hukyou", "Hukyou_B",
                   "Shizuriyuki", "Wasureyuki", "Zarameyuki", "Kamuriyuki", "berserk", "berserkEnd"},
    "Axl": {"AxlBomber", "Katatumuri", "Katatumuri_Air", "Kyokusageki", "Mogaribue", "Niwatazumi",
            "Rensengeki", "Sensageki", "Rensen_Bomb", "Rensen_Bomb_Just", "WinterMantis",
            "ByakueRenshou", "OneVision"},
    "Baiken": {"Atemi", "Kabari", "HimoKabari", "KabariHasei", "KenjyuLand", "TatamiLand",
               "YouZanSen", "TsuraneSanzuWatashi"},
    "Faust": {"Ai", "Kuebiko", "MettaGari", "NanigaDerukana", "NaNaNaNanigaDerukana", "Nekkakonnrenn",
              "Nekkakonnrenn_Air", "NiceShot_NoHasei", "Tsukimasu", "exciting"},
    "Anji": {"Fujin", "Issokutobi", "KachoFugetsu", "Kou", "Midare", "Nagiha", "Rin", "Sai", "Shin",
             "Shitsu", "Suigetsu"},
    "Millia": {"Artemis", "BadMoon", "IronSavior", "Kapel", "KousokuRakka", "LustShaker", "Mirazh",
               "SeptemVoices", "TandemTopC", "TandemTopD", "Winger"},
    "I-No": {"AirHover_4", "AirHover_6", "Chemical", "Chemical_Air", "Genkai_Land", "KouutsuOnkai",
             "KouutsuOnkai_Air", "Kyouai", "KyougenB", "KyougenC", "KyougenD", "TaibokuC", "TaibokuD"},
    "Giovanna": {"Shervi", "Special1", "Special2_air", "Special2_land", "Special3", "Special4"},
    "Goldlewis": {"Drone", "Minigun", "Swing_pattern1", "Swing_pattern3", "AirSwing_pattern1"},
}
COMMON_PREFIXES = ("CmnAct", "NmlAtk", "WSB_", "WinAction", "ThrowExe", "AirThrowExe",
                   "CounterBlitz", "WorldSideBreak", "HomingJump", "DustFinish")


def infer_character(acts) -> str:
    names = collections.Counter(acts)
    codes = collections.Counter()
    for a, n in names.items():
        m = re.match(r"([A-Za-z]{3})_", a)
        if m and m.group(1) in CHAR_CODES:
            codes[CHAR_CODES[m.group(1)]] += n
    if codes:
        return codes.most_common(1)[0][0]
    own = {a for a in names if not a.startswith(COMMON_PREFIXES)}
    best = max(CHAR_SIGNATURES.items(), key=lambda kv: len(kv[1] & own))
    if best[1] & own:
        return best[0]
    return "unknown:" + "/".join(sorted(own)[:4])


def resolve_character(acts, logged=None) -> tuple[str, str]:
    """(character, source). `logged` = the logger's char column values for this player (name or
    numeric roster id); when present and consistent it wins over the action-name inference."""
    inferred = infer_character(acts)
    if logged is not None and len(logged):
        vals = collections.Counter(str(v).strip() for v in logged if str(v).strip() not in ("", "nan"))
        if vals:
            v = vals.most_common(1)[0][0]
            try:
                i = int(float(v))
                name = CHAR_IDS[i] if 0 <= i < len(CHAR_IDS) else f"id{i}"
            except ValueError:
                name = v
            return name, "logged" if inferred.startswith("unknown") or inferred == name else "logged(conflict:" + inferred + ")"
    return inferred, "inferred"


# ---------------------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------------------

BTN = {"A": "P", "B": "K", "C": "S", "D": "H", "E": "D"}
NORMAL, CMD, SPECIAL, SUPER, THROW, PROJ, BURST, OTHER = (
    "normal", "command normal", "special", "super", "throw", "projectile", "burst", "other")

# flags: proj = the move's hit is a projectile / detached object; grab = command grab;
# nonattack = not an attack (a hit while in it comes from an earlier projectile);
# inferred = mapping from Dustloop order / variants, not seen on video; verified = seen on strips.
# act -> (notation, name, category, flags)
M = {}


def _add(char, table):
    M[char] = {k: (v[0], v[1], v[2], set(v[3].split()) if len(v) > 3 else set()) for k, v in table.items()}


_add("Asuka", {
    # which spell was cast is not in the action name (see SPELL_NOTE)
    "Magic": ("spell", "spell (236X)", SPECIAL, "proj"),
    "Magic_air": ("j.spell", "spell (j.236X)", SPECIAL, "proj"),
    "Draw_or_Discard": ("214X", "Draw/Discard", OTHER, "nonattack"),
    "Draw_or_Discard_air": ("j.214X", "Draw/Discard", OTHER, "nonattack"),
    "Special_MP": ("22S", "Recover Mana", OTHER, "nonattack"),
    "Change_deck": ("22H~X", "Change Test Case", OTHER, "nonattack"),
    "Change_deck_air": ("j.22H~X", "Change Test Case", OTHER, "nonattack"),
    "PhysicalLaw": ("22P/K", "Chant (Synthesize)?", OTHER, "nonattack inferred"),
    "fuwafuwa": ("", "float", OTHER, "nonattack"),
    "ASK_Ultimate1": ("632146P/K", "Submicron Particle Sphere", SUPER),
    "ASK_Ultimate2": ("632146S", "Bookmark (Full Order)", SUPER),
})
_add("Elphelt", {
    "ELP_Special1_S_Air": ("j.236S", "Air Miss Charlotte", SPECIAL, "proj"),
    "ELP_Special1_HS_Air": ("j.236H", "Air Miss Charlotte", SPECIAL, "proj verified"),
    "ELP_Special1_S_Land": ("236S", "Miss Charlotte", SPECIAL, "proj"),
    "ELP_Special1_HS_Land": ("236H", "Miss Charlotte", SPECIAL, "proj verified"),
    "ELP_Special2": ("214K|214H", "Out of Repair / Bomb-Bomb Chocolat (unresolved)", SPECIAL, "inferred"),
    "ELP_Special4": ("214K|214H", "Out of Repair / Bomb-Bomb Chocolat (unresolved)", SPECIAL, "inferred"),
    "ELP_Special3_Start": ("214S", "Here I Go!", SPECIAL, "verified"),
    "ELP_Special3_Chudan": ("214S~P", "Up High!", SPECIAL, "verified"),
    "ELP_Special3_Chudan_2nd": ("214S~X~P", "Up High! (Finisher)", SPECIAL, "verified"),
    "ELP_Special3_Gedan": ("214S~K", "Down Low!", SPECIAL, "verified"),
    "ELP_Special3_Gedan_2nd": ("214S~X~K", "Down Low! (Finisher)", SPECIAL),
    "ELP_Special3_Finish": ("214S~H", "Nailed It!", SPECIAL),
    "ELP_SpecialCancel_FDash": ("", "special cancel dash", OTHER, "nonattack"),
    "ELP_Ultimate1": ("632146H", "Juganto Da Parfeo", SUPER),
    "ELP_Ultimate2": ("236236K", "Bomb-Bombnniere", SUPER, "inferred"),
})
_add("Lucy", {  # numbering checked on strips (SP_01 gun shot full screen, SP_06 has _Cancel)
    "LUC_SP_01": ("236S", "Unity", SPECIAL, "proj verified"),
    "LUC_SP_02": ("214K", "Sprint Attack", SPECIAL, "verified"),
    "LUC_SP_03": ("214K~S", "Sprint Attack - Sweep", SPECIAL, "inferred"),
    "LUC_SP_04": ("214K~H", "Sprint Attack - Cross Up", SPECIAL, "verified"),
    "LUC_SP_05": ("623H", "Monowire - Breakout", SPECIAL, "verified"),
    "LUC_SP_06": ("214H", "Monowire", SPECIAL, "verified"),
    "LUC_SP_06_Cancel": ("214H~P/D", "Monowire Cancel", OTHER, "nonattack"),
    "LUC_ULT_01": ("632146H", "Live Wire", SUPER, "verified"),
    "LUC_ULT_02": ("632146S", "Quickhack", SUPER, "inferred"),
})
_add("Slayer", {  # Dustloop order == internal numbering; SP_08 / SP_02 checked on strips
    "sly_SP_01": ("236P", "Mappa Hunch P", SPECIAL),
    "sly_SP_02": ("236K", "Mappa Hunch K", SPECIAL, "verified"),
    "sly_SP_03": ("214P", "Dandy Step P", SPECIAL),
    "sly_SP_04": ("214K", "Dandy Step K", SPECIAL),
    "sly_SP_05": ("214P/K~P", "Pilebunker", SPECIAL),
    "sly_SP_06": ("214P/K~K", "Bump Ahead", SPECIAL),
    "sly_SP_07": ("214P/K~S", "It's Late", SPECIAL),
    "sly_SP_08": ("214P/K~H", "Master's Hammer", SPECIAL, "verified"),
    "sly_SP_09": ("63214H", "Bloodsucking Universe", SPECIAL, "grab"),
    "sly_SP_09_exe": ("63214H", "Bloodsucking Universe", SPECIAL, "grab"),
    "sly_SP_10": ("44~6S", "Hand of Doom", SPECIAL),
    "sly_ULT_01": ("632146S", "Super Mappa Hunch", SUPER, "inferred"),
    "sly_ULT_02": ("236236H", "Last Horizon", SUPER),
})
_add("A.B.A", {  # SP1 = the only air/exe special; SP2-SP6 = the five with JR ("Moroha") versions
    "ABA_SP1_Land": ("214H", "Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Land_exe": ("214H", "Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Land_Moroha": ("JR 214H", "JR Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Air": ("j.214H", "Air Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Air_exe": ("j.214H", "Air Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP2": ("63214P", "Changing and Swaying", SPECIAL, "inferred grab"),
    "ABA_SP2_Moroha": ("JR 63214P", "JR Changing and Swaying", SPECIAL, "inferred grab"),
    "ABA_SP3": ("214K", "Haul and Heed", SPECIAL, "inferred"),
    "ABA_SP3_Moroha": ("JR 214K", "JR Haul and Heed", SPECIAL, "inferred"),
    "ABA_SP4": ("236K", "Intertwine and Tilt", SPECIAL, "verified"),
    "ABA_SP4_Moroha": ("JR 236K", "JR Intertwine and Tilt", SPECIAL, "verified"),
    "ABA_SP5": ("236S", "Menace and Groan", SPECIAL, "inferred"),
    "ABA_SP5_Moroha": ("JR 236S", "JR Menace and Groan", SPECIAL, "inferred"),
    "ABA_SP6": ("236S~6S", "Restriction and Constraint", SPECIAL, "inferred"),
    "ABA_SP6_Moroha": ("JR 236S~6S", "JR Restriction and Constraint", SPECIAL, "inferred"),
    "ABA_SP7": ("214S", "Frenzy and Astonishment", SPECIAL, "inferred"),
    "ABA_SP8": ("623H", "Judgment and Sentiment", SPECIAL, "inferred"),
    "ABA_Moroha_End": ("", "Jealous Rage Deactivation", OTHER, "nonattack"),
    "ABA_ULT1": ("632146H", "The Law is Key, Key is King.", SUPER),
    "ABA_ULT1_exe": ("632146H", "The Law is Key, Key is King.", SUPER),
    "ABA_ULT2": ("632146K", "Keeper of the Key", SUPER, "inferred"),
    "ABA_ULT2_Moroha": ("JR 632146K", "JR Keeper of the Key", SUPER, "inferred"),
})
_add("Leo", {
    "GraviertWurdeC": ("[4]6S", "S Graviert Wurde", SPECIAL, "proj"),
    "GraviertWurdeD": ("[4]6H", "H Graviert Wurde", SPECIAL, "proj"),
    "EisenSturmC": ("[2]8S", "S Eisensturm", SPECIAL),
    "EisenSturmD": ("[2]8H", "H Eisensturm", SPECIAL),
    "ErstWind": ("236S", "Kaltes Gestober Erst", SPECIAL),
    "ZweitWind": ("236H", "Kaltes Gestober Zweit", SPECIAL),
    "Turbulenz": ("214S", "Turbulenz", SPECIAL),
    "GlaenzenDunkel": ("bt.214K", "Glaenzendes Dunkel", SPECIAL),
    "GlaenzenDunkelExe": ("bt.214K", "Glaenzendes Dunkel", SPECIAL),
    "SchildBrechen": ("bt.214H?", "Blitzschlag?", SPECIAL, "inferred"),
    "GuardAttack": ("bt.P", "Kahn-Schild (guard attack)", SPECIAL, "inferred"),
    "StahlWirbel": ("632146S", "Stahl Wirbel", SUPER),
    "LeidenschaftDirigent": ("632146H", "Leidenschaft des Dirigenten", SUPER),
    "Semuke": ("bt.", "backturn", OTHER, "nonattack"),
    "SemukeFWalk": ("bt.6", "backturn walk", OTHER, "nonattack"),
    "SemukeBWalk": ("bt.4", "backturn walk", OTHER, "nonattack"),
    "SemukeFDashStep": ("bt.66", "backturn dash", OTHER, "nonattack"),
    "SemukeBDashStep": ("bt.44", "backturn dash", OTHER, "nonattack"),
    "SemukeJump": ("bt.8", "backturn jump", OTHER, "nonattack"),
    "SemukeReverse": ("bt.22", "backturn reverse", OTHER, "nonattack"),
    "SemukeJumpAttack": ("bt.j.X?", "backturn jump attack (unresolved)", CMD, "inferred"),
})
_add("Goldlewis", {
    **{f"Swing_pattern{i}": (f"{p}H", f"Behemoth Typhoon ({p})", SPECIAL, "inferred")
       for i, p in enumerate(["41236", "63214", "23698", "21478", "69874", "47896", "87412", "89632"], 1)},
    **{f"AirSwing_pattern{i}": (f"j.{p}H", f"Air Behemoth Typhoon ({p})", SPECIAL, "inferred")
       for i, p in enumerate(["41236", "63214", "23698", "21478", "69874", "47896", "87412", "89632"], 1)},
    "Drone": ("214S", "Thunderbird", SPECIAL, "proj inferred"),
    "Minigun": ("236S", "Skyfish", SPECIAL, "inferred"),
    "GLD_Ultimate1": ("632146P", "Down With The System", SUPER, "grab"),
    "GLD_Ultimate2": ("236236K", "Burn It Down", SUPER, "proj"),
})
_add("Nagoriyuki", {
    "Zarameyuki": ("236S", "Zarameyuki", SPECIAL, "proj"),
    "Hukyou": ("236K", "Forward Fukyo", SPECIAL),
    "Hukyou_B": ("214K", "Backward Fukyo", SPECIAL),
    "Kamuriyuki": ("214H", "Kamuriyuki", SPECIAL),
    "Shizuriyuki": ("623H", "Shizuriyuki", SPECIAL),
    "Bloodsucking_Universe": ("623P", "Bloodsucking Universe", SPECIAL, "grab"),
    "Bloodsucking_UniverseExe": ("623P", "Bloodsucking Universe", SPECIAL, "grab"),
    "NAG_NewSpecial": ("214S", "Yukimoyoi", SPECIAL, "inferred"),
    "Wasureyuki": ("632146S", "Wasureyuki", SUPER),
    "Zansetsu": ("632146H", "Zansetsu", SUPER),
    "berserk": ("", "Blood Rage", OTHER, "nonattack"),
    "berserkEnd": ("", "Blood Rage end", OTHER, "nonattack"),
})
_add("Sol", {
    "GunFlame": ("236P", "Gun Flame", SPECIAL, "proj"),
    "GunFlameFeint": ("214P", "Gun Flame (Feint)", SPECIAL),
    "VolcanicViperLandS": ("623S", "S Volcanic Viper", SPECIAL),
    "VolcanicViperLandHS": ("623H", "H Volcanic Viper", SPECIAL),
    "VolcanicViperAirS": ("j.623S", "Aerial S Volcanic Viper", SPECIAL),
    "VolcanicViperAirHS": ("j.623H", "Aerial H Volcanic Viper", SPECIAL),
    "NewSidewinder": ("623H~6H", "Side Winder Ignited", SPECIAL),
    "BanditRevolver_Land": ("236K", "Bandit Revolver", SPECIAL),
    "BanditRevolver_Air": ("j.236K", "Aerial Bandit Revolver", SPECIAL),
    "BanditBringer_Land": ("214K", "Bandit Bringer", SPECIAL),
    "BanditBringer_Air": ("j.214K", "Aerial Bandit Bringer", SPECIAL),
    "BukkirabouNiNageru": ("623K", "Wild Throw", SPECIAL, "grab"),
    "BukkiraExe": ("623K", "Wild Throw", SPECIAL, "grab"),
    "Vortex": ("214S", "Night Raid Vortex", SPECIAL),
    "Fefnir": ("41236H", "Fafnir", SPECIAL),
    "TyrantRave": ("632146H", "Tyrant Rave", SUPER, "inferred"),
})
_add("Ky", {
    "StunEdge1": ("236S", "Stun Edge", SPECIAL, "proj"),
    "StunEdge2": ("236H", "Charged Stun Edge", SPECIAL, "proj"),
    "AirStunEdge1": ("j.236S", "Aerial S Stun Edge", SPECIAL, "proj"),
    "AirStunEdge2": ("j.236H", "Aerial H Stun Edge", SPECIAL, "proj"),
    "VaporThrustC": ("623S", "S Vapor Thrust", SPECIAL),
    "VaporThrustD": ("623H", "H Vapor Thrust", SPECIAL),
    "AirVaporThrustC": ("j.623S", "Air S Vapor Thrust", SPECIAL),
    "AirVaporThrustD": ("j.623H", "Air H Vapor Thrust", SPECIAL),
    "StunDipper": ("236K", "Stun Dipper", SPECIAL),
    "FadulArc": ("214K", "Foudre Arc", SPECIAL),
    "Diaekura": ("214S", "Dire Eclat", SPECIAL),
    "Lightning": ("214P|632146H", "Lightning Strike / Ride the Lightning (unresolved)", SPECIAL, "inferred"),
    "SacredEdge": ("236236P", "Sacred Edge", SUPER, "proj"),
    "DragonInstall": ("214214H", "Dragon Install", SUPER, "nonattack"),
})
_add("Chipp", {
    "AlphaBlade": ("236K", "Alpha Blade (Diagonal)", SPECIAL, "inferred"),
    "AlphaBlade_yoko": ("236P", "Alpha Blade (Horizontal)", SPECIAL, "inferred"),
    "AlphaBlade_Cancel": ("236P/K", "Alpha Blade", SPECIAL),
    "AirAlphaBlade_yoko": ("j.236P", "Air Alpha Blade (Horizontal)", SPECIAL, "inferred"),
    "AirAlphaBlade_Cancel": ("j.236P/K", "Air Alpha Blade", SPECIAL),
    "GammaBlade": ("236H", "Gamma Blade", SPECIAL, "proj"),
    "Shuriken": ("j.214P", "Shuriken", SPECIAL, "proj"),
    "Sushi": ("236S", "Resshou", SPECIAL, "inferred"),
    "Sukiyaki": ("236S 236S", "Rokusai", SPECIAL, "inferred"),
    "Banzai": ("236S 236K", "Senshuu", SPECIAL, "inferred"),
    "ItoAtk": ("214H", "Tightrope (attack)", SPECIAL, "inferred"),
    "BankiMessai": ("236236P", "Banki Messai", SUPER),
    **{k: ("214H", "Tightrope", OTHER, "nonattack") for k in
       ("ThreadRun_Jump", "ThreadRun_Down", "ThreadRun_K_Down", "ThreadRun_S_Down", "ThreadRun_HS_Down",
        "ThreadRun_6HS_Down", "WallRun_Down")},
})
_add("Ramlethal", {
    "BajonetoC": ("236S", "S Bajoneto", SPECIAL, "proj"),
    "BajonetoD": ("236H", "H Bajoneto", SPECIAL, "proj"),
    "Dauro": ("623P", "Dauro", SPECIAL),
    "Ondo": ("236K", "Ondo", SPECIAL),
    "ShirudoLand": ("214K", "Sildo Detruo", SPECIAL),
    "Sabrobato": ("214H", "Sabrubato", SPECIAL),
    "Erarurumo1": ("214P", "Erarlumo (1)", SPECIAL),
    "Erarurumo2": ("214P 4P", "Erarlumo (2)", SPECIAL),
    "Erarurumo3": ("214P 4P 4P", "Erarlumo (3)", SPECIAL),
    "AgresaOrdono": ("j.214S", "Agressa Ordono", SPECIAL),
    "Falstelo": ("632146H", "Calvados", SUPER, "inferred"),
    "Morutobato": ("236236S", "Mortobato", SUPER),
})
_add("Potemkin", {
    "HammerFall": ("[4]6H", "Hammer Fall", SPECIAL),
    "HammerFallBrake": ("[4]6H~P", "Hammer Fall Break", SPECIAL),
    "PotemkinBuster": ("632146P", "Potemkin Buster", SPECIAL, "grab"),
    "PotemkinBusterExe": ("632146P", "Potemkin Buster", SPECIAL, "grab"),
    "HeatKnucle": ("623H", "Heat Knuckle", SPECIAL, "grab"),
    "HeatTackle": ("623K", "Heat Tackle", SPECIAL),
    "MegaFistFront": ("236P", "Forward Mega Fist", SPECIAL),
    "MegaFistBack": ("214P", "Backward Mega Fist", SPECIAL),
    "SlideHead": ("236S", "Slide Head", SPECIAL),
    "FDB": ("214K", "F.D.B.", SPECIAL),
    "HeavenlyPBuster": ("632146S", "Heavenly Potemkin Buster", SUPER, "grab"),
})
_add("Faust", {
    "Tsukimasu": ("41236K", "Thrust", SPECIAL),
    "NiceShot_NoHasei": ("214H", "Hole In One", SPECIAL, "inferred"),
    "NanigaDerukana": ("236P", "What Could This Be?", SPECIAL, "proj"),
    "NanigaDerukana_Keep": ("236[P]", "What Could This Be? (hold)", SPECIAL, "proj"),
    "NaNaNaNanigaDerukana": ("236236P", "W-W-What Could This Be?", SUPER, "proj"),
    "Nekkakonnrenn": ("236S", "Mix Mix Mix", SPECIAL),
    "Nekkakonnrenn_Air": ("j.236S", "Air Mix Mix Mix", SPECIAL),
    "MettaGari": ("236H", "Snip Snip Snip", SPECIAL, "grab"),
    "MettaGariExe": ("236H", "Snip Snip Snip", SPECIAL, "grab"),
    "Ai": ("j.236P", "Love", SPECIAL, "proj"),
    "Kuebiko": ("214P/K/S", "Scarecrow", SPECIAL, "nonattack"),
    "exciting": ("632146H", "Bone-crushing Excitement", SUPER),
    "excitingexe": ("632146H", "Bone-crushing Excitement", SUPER),
    "CrouchFWalk": ("3", "crawl", OTHER, "nonattack"),
    "CrouchBWalk": ("1", "crawl", OTHER, "nonattack"),
})
_add("I-No", {
    "TaibokuC": ("236S", "S Stroke the Big Tree", SPECIAL),
    "TaibokuD": ("236H", "H Stroke the Big Tree", SPECIAL),
    "KyougenB": ("j.236K", "K Sultry Performance", SPECIAL),
    "KyougenC": ("j.236S", "S Sultry Performance", SPECIAL),
    "KyougenD": ("j.236H", "H Sultry Performance", SPECIAL),
    "KyougenHasei": ("j.236HH", "H Leap", SPECIAL),
    "KouutsuOnkai": ("214P", "Antidepressant Scale", SPECIAL, "proj"),
    "KouutsuOnkai_Air": ("j.214P", "Air Antidepressant Scale", SPECIAL, "proj"),
    "Chemical": ("214K", "Chemical Love", SPECIAL, "proj"),
    "Chemical_Air": ("j.214K", "Air Chemical Love", SPECIAL, "proj"),
    "Kyouai": ("214S", "Mad Love Agitato", SPECIAL, "inferred"),
    "Genkai_Land": ("632146S", "Ultimate Fortissimo", SUPER, "inferred"),
    "AirHover_4": ("j.4 (hover)", "hoverdash", OTHER, "nonattack"),
    "AirHover_6": ("j.6 (hover)", "hoverdash", OTHER, "nonattack"),
})
_add("Millia", {
    "TandemTopC": ("236S", "S Tandem Top", SPECIAL, "proj"),
    "TandemTopD": ("236H", "H Tandem Top", SPECIAL, "proj"),
    "LustShaker": ("214S", "Lust Shaker", SPECIAL),
    "IronSavior": ("214P", "Iron Savior", SPECIAL),
    "BadMoon": ("j.236P", "Bad Moon", SPECIAL),
    "KousokuRakka": ("j.236K", "Turbo Fall", SPECIAL),
    "Mirazh": ("214K", "Mirazh", SPECIAL),
    "Artemis": ("214H", "Artemis", SPECIAL),
    "Kapel": ("j.236H", "Kapel", SPECIAL),
    "SeptemVoices": ("236236S", "Septem Voices", SUPER),
    "Winger": ("632146H", "Winger", SUPER),
})
_add("Baiken", {
    "TatamiLand": ("236K", "Tatami Gaeshi", SPECIAL),
    "TatamiAir": ("j.236K", "Air Tatami Gaeshi", SPECIAL),
    "Kabari": ("41236S", "S Kabari", SPECIAL, "inferred"),
    "HimoKabari": ("41236H", "H Kabari", SPECIAL, "inferred"),
    "KabariHasei": ("41236H~H", "Kabari Followup", SPECIAL),
    "YouZanSen": ("j.236S", "Youzansen", SPECIAL),
    "Atemi": ("236P", "Hiiragi", SPECIAL),
    "TsuraneSanzuWatashi": ("236236S", "Tsurane Sanzu-watashi", SUPER),
    "TsuraneSanzuWatashiExe": ("236236S", "Tsurane Sanzu-watashi", SUPER),
    "KenjyuLand": ("214214P", "Kenjyu", SUPER, "proj"),
})
_add("Axl", {
    "Rensengeki": ("236S", "Sickle Flash", SPECIAL),
    "Sensageki": ("236S 8", "Soaring Chain Strike", SPECIAL),
    "Kyokusageki": ("236S 2", "Spinning Chain Strike", SPECIAL),
    "Rensen_Bomb": ("236S S", "Winter Cherry", SPECIAL, "proj"),
    "Rensen_Bomb_Just": ("236S S", "Winter Cherry (just)", SPECIAL, "proj"),
    "WinterMantis": ("41236H", "Winter Mantis", SPECIAL, "grab"),
    "WinterMantisExe_Far": ("41236H", "Winter Mantis", SPECIAL, "grab"),
    "Niwatazumi": ("214S", "Rainwater", SPECIAL),
    "Katatumuri": ("214H", "Snail", SPECIAL),
    "Katatumuri_Air": ("j.214H", "Air Snail", SPECIAL),
    "AxlBomber": ("j.236H", "Axl Bomber", SPECIAL),
    "Mogaribue": ("214K", "Whistling Wind", SPECIAL),
    "ByakueRenshou": ("236236H", "Sickle Storm", SUPER),
    "OneVision": ("632146P", "One Vision", SUPER),
    "OneVision_Add": ("632146P", "One Vision (attack)", SUPER),
})
_add("Anji", {
    "Shitsu": ("236P", "Shitsu", SPECIAL, "proj"),
    "Suigetsu": ("236K", "Suigetsu No Hakobi", OTHER, "nonattack"),
    "Midare": ("236K~214P", "Midare", SPECIAL),
    "Kou": ("236S", "Kou", SPECIAL),
    "Fujin": ("236H", "Fuujin", SPECIAL),
    "Shin": ("236H P", "Shin: Ichishiki", SPECIAL),
    "Issokutobi": ("236H K", "Issokutobi", SPECIAL),
    "Nagiha": ("236H S", "Nagiha", SPECIAL),
    "Rin": ("236H H", "Rin", SPECIAL),
    "Sai": ("632146H", "Issei Ougi: Sai", SUPER),
    "KachoFugetsu": ("632146S", "Kachoufuugetsu Kai", SUPER),
    "Dodge": ("", "dodge (unresolved)", OTHER, "nonattack"),
})
_add("Giovanna", {  # only Special2 (air+land = Sol Poente) and LandUltimate are anchored
    "Special2_land": ("214S", "Sol Poente", SPECIAL, "inferred"),
    "Special2_air": ("j.214S", "Air Sol Poente", SPECIAL, "inferred"),
    "Special1": ("236K|214K|623S", "Trovao/Sepultura/Sol Nascente (unresolved)", SPECIAL, "inferred"),
    "Special3": ("236K|214K|623S", "Trovao/Sepultura/Sol Nascente (unresolved)", SPECIAL, "inferred"),
    "Special4": ("236K|214K|623S", "Trovao/Sepultura/Sol Nascente (unresolved)", SPECIAL, "inferred"),
    "Shervi": ("214H", "Chave", SPECIAL, "inferred"),
    "LandUltimate": ("632146H", "Ventania", SUPER),
    "AirUltimate": ("j.236236H", "Tempestade", SUPER),
})

SPELL_NOTE = ("Asuka's spell identity is not in p*_act (always Magic / Magic_air). Candidates for "
              "the logger: the spell projectile object's own action name (spawned entities have "
              "their own state names, e.g. the spell's internal name), or Asuka's hand/deck "
              "slots in her player struct (the cast slot index at the Magic frame).")

COMMON = {
    "CmnActBurst": ("Burst", "Psych Burst", BURST, set()),
    "CmnActRomanCancel": ("RC", "Roman Cancel", OTHER, {"nonattack"}),
    "NmlAtkThrow": ("Throw", "Ground Throw", THROW, set()),
    "ThrowExe": ("Throw", "Ground Throw", THROW, set()),
    "NmlAtkAirThrow": ("j.Throw", "Air Throw", THROW, set()),
    "AirThrowExe": ("j.Throw", "Air Throw", THROW, set()),
    "DustFinish": ("j.X (homing jump)", "Finish Blow", OTHER, set()),
    "HomingJump": ("", "homing jump", OTHER, {"nonattack"}),
    "HomingJumpLoop": ("", "homing jump", OTHER, {"nonattack"}),
    "CounterBlitz": ("5D/4D", "Deflect Shield", OTHER, set()),
}

# Normals: NmlAtk[Air]<numpad><A-E>[Near|Far][_suffix]; Leo's backturn normals Semuke5<A-E>.
_NML = re.compile(r"NmlAtk(Air)?(\d)([A-E])(Near|Far)?(_.*)?")
_SEMUKE = re.compile(r"Semuke(\d)([A-E])")


def normal_info(act: str):
    m = _NML.fullmatch(act)
    if m:
        air, d, b, nf, suf = m.groups()
        btn = BTN[b]
        if nf == "Near":
            s = "c.S"
        elif nf == "Far":
            s = "f.S"
        else:
            s = (("j." + ("" if d == "5" else d)) if air else d) + btn
        if suf:
            s += f"({suf[1:]})"       # e.g. A.B.A Jealous Rage normals: 2K(Moroha)
        cat = NORMAL if d in ("5", "2") else CMD   # 6P, 6K, 6H, j.2K, 3K ...
        stance = "air" if air else ("crouch" if d in ("1", "2", "3") else "stand")
        return s, "", cat, {"btn:" + btn, "stance:" + stance}
    m = _SEMUKE.fullmatch(act)
    if m:
        d, b = m.groups()
        return f"bt.{BTN[b]}", "backturn normal", CMD, {"btn:" + BTN[b], "stance:stand"}
    return None


def move_info(char: str, act: str):
    """(notation, name, category, flags) or None when unknown."""
    if act in COMMON:
        return COMMON[act]
    if act.startswith("WorldSideBreak"):
        return ("", "wallbreak", OTHER, set())
    n = normal_info(act)
    if n:
        return n
    ci = M.get(char, {}).get(act)
    if ci:
        return ci
    for t in M.values():          # character unknown / mislabelled: unique name anywhere
        if act in t:
            return t[act]
    if act.startswith(("CmnAct", "WSB_", "WinAction")):
        return ("", act, OTHER, {"nonattack"})
    return None


def notation(act: str, char: str = "") -> str:
    """Back-compat: internal action -> notation (NmlAtk5CFar -> f.S); unknown names pass through."""
    mi = move_info(char, act)
    if mi and mi[0]:
        return mi[0]
    return act


def classify(row: dict) -> dict:
    """Clean move fields for one opener row (build.py extract_openers + attacker_char).

    The hitting move is the attacker's current action unless that is not an attack (moving,
    guarding, Asuka drawing cards, Leo backturn walk ...): then a projectile/object from the
    attacker's last attack hit, and the move is that attack (category 'projectile')."""
    char = row.get("attacker_char", "")
    act = row["move_act"]
    mi = move_info(char, act)
    src = "action"
    if (mi is None or "nonattack" in mi[3]) and act not in ("CmnActBurst",):
        la = row.get("last_real_attack_act") or row.get("last_attack_act") or ""
        lmi = move_info(char, la) if la else None
        if lmi and "nonattack" not in lmi[3]:
            mi, act, src = lmi, la, "last_attack"
        elif mi is None:
            mi = ("", act, OTHER, {"unmapped"})
    nota, name, cat, flags = mi
    if src == "last_attack" and cat not in (SUPER, BURST, THROW):
        cat = PROJ
    elif "proj" in flags and cat == SPECIAL and row.get("projectile_likely") and (row.get("attacker_act_age") or 0) > 45:
        cat = PROJ                       # the special's object hit long after the cast
    btn = next((f[4:] for f in flags if f.startswith("btn:")), None)
    stance = next((f[7:] for f in flags if f.startswith("stance:")), None)
    return dict(
        char=char, move=nota or name or act, move_name=name, category=cat, move_src=src,
        move_src_act=act, move_flags=" ".join(sorted(f for f in flags if ":" not in f)) or None,
        button=btn, stance=stance, mapped="unmapped" not in flags,
    )


def coverage(acts_by_char: dict) -> dict:
    """{char: {'mapped': [...], 'unmapped': [...]}} for the given action names."""
    out = {}
    for ch, acts in acts_by_char.items():
        mp, um = [], []
        for a in sorted(acts):
            (mp if move_info(ch, a) else um).append(a)
        out[ch] = {"mapped": mp, "unmapped": um}
    return out


if __name__ == "__main__":       # write moves.json: the mapping table + coverage of the dataset
    import json
    import sys
    from pathlib import Path
    out = {"characters": {"codes": CHAR_CODES, "signatures": {k: sorted(v) for k, v in CHAR_SIGNATURES.items()},
                          "roster_ids_unverified": CHAR_IDS},
           "common": {k: list(v[:3]) for k, v in COMMON.items()},
           "moves": {ch: {a: {"notation": v[0], "name": v[1], "category": v[2], "flags": sorted(v[3])}
                          for a, v in t.items()} for ch, t in M.items()},
           "asuka_spell_note": SPELL_NOTE}
    pq_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if pq_path:
        import pyarrow.parquet as pq
        rows = pq.read_table(pq_path).to_pylist()
        by = collections.defaultdict(set)
        for r in rows:
            by[r["char"]].add(r["move_act"])
            if r.get("move_src_act"):
                by[r["char"]].add(r["move_src_act"])
        out["opener_coverage"] = coverage(by)
    Path(__file__).with_name("moves.json").write_text(json.dumps(out, indent=1))
    print("wrote", Path(__file__).with_name("moves.json"))
