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
    "VEN": "Venom", "UNI": "Unika", "LUC": "Lucy", "USG": "Jam", "ABS": "Robo-Ky",
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


def resolve_character(acts, logged=None, sprites=None) -> tuple[str, str]:
    """(character, source). `sprites` = the logger's sprite names (sol000_11: the first three
    letters are the character code) win over everything: the action-name inference only knows
    the characters in moves.json and mistakes some (Bedman? for Giovanna). `logged` = the
    logger's char column values (name or numeric roster id), next."""
    if sprites is not None and len(sprites):
        codes = collections.Counter(str(v)[:3].upper() for v in sprites if len(str(v)) > 3)
        if codes:
            code = codes.most_common(1)[0][0]
            if code in CHAR_CODES:
                return CHAR_CODES[code], "sprite"
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
# command = orders a summon that stays out (Servant, Eddie, crow): its later hits are this move's;
# inferred = mapping from Dustloop order / variants, not seen on video; verified = seen on strips;
# inputs = the motion + button checked against the logged inputs (verify_inputs.py);
# dustloop = the notation checked on Dustloop's frame data (rare moves the logged inputs
# are too few or too sloppy to settle).
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
    # the inputs settle the motions; which of the two names goes with which is not checked
    "ELP_Special2": ("214K", "Out of Repair", SPECIAL, "inputs dustloop"),
    "ELP_Special4": ("214H", "Bomb-Bomb Chocolat", SPECIAL, "inputs dustloop"),
    "ELP_Special3_Start": ("214S", "Here I Go!", SPECIAL, "verified"),
    "ELP_Special3_Start_easy_2nd": ("214S~S", "Here I Go! (2nd)", SPECIAL, "inferred"),
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
    "sly_ULT_01": ("236236H", "Last Horizon", SUPER, "inputs"),
    "sly_ULT_02": ("632146S", "Super Mappa Hunch", SUPER, "inputs"),
    "sly_ULT_01_exe": ("236236H", "Last Horizon", SUPER, "inputs"),
})
_add("A.B.A", {  # SP1 = the only air/exe special; SP2-SP6 = the five with JR ("Moroha") versions
    "ABA_SP1_Land": ("214H", "Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Land_exe": ("214H", "Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Land_Moroha": ("JR 214H", "JR Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Air": ("j.214H", "Air Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Air_exe": ("j.214H", "Air Bonding and Dissolving", SPECIAL, "inferred"),
    "ABA_SP1_Air_Moroha": ("JR j.214H", "JR Air Bonding and Dissolving", SPECIAL, "inferred"),
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
    "ABA_ULT1_Moroha": ("JR 632146H", "JR The Law is Key, Key is King.", SUPER, "inferred"),
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
    "StahlWirbel": ("632146S", "Stahl Wirbel", SUPER, "dustloop"),
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
BEHEMOTH = ["41236", "47896", "63214", "69874", "89632", "23698", "87412", "21478"]
_add("Goldlewis", {
    # pattern order read from the logged inputs (verify_inputs.py): half circles in pairs,
    # 7 by elimination
    **{f"Swing_pattern{i}": (f"{p}H", f"Behemoth Typhoon ({p})", SPECIAL, "inputs")
       for i, p in enumerate(BEHEMOTH, 1)},
    **{f"AirSwing_pattern{i}": (f"j.{p}H", f"Air Behemoth Typhoon ({p})", SPECIAL, "inputs")
       for i, p in enumerate(BEHEMOTH, 1)},
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
    "Zansetu": ("632146H", "Zansetsu", SUPER, "inferred"),
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
    "HMC": ("214214H", "Heavy Mob Cemetery", SUPER, "dustloop"),
    "HMCExe": ("214214H", "Heavy Mob Cemetery", SUPER, "inferred"),
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
    "Lightning": ("214P", "Lightning Strike", SPECIAL, "inferred"),  # Ride the Lightning = RideTheLightning
    "SacredEdge": ("236236P", "Sacred Edge", SUPER, "proj"),
    "DragonInstall": ("214214H", "Dragon Install", SUPER, "nonattack"),
    "RideTheLightning": ("632146H", "Ride the Lightning", SUPER, "inferred"),
    "AirRideTheLightning": ("j.632146H", "Air Ride the Lightning", SUPER, "inferred"),
})
_add("Chipp", {
    "AlphaBlade": ("236K", "Alpha Blade (Diagonal)", SPECIAL, "inferred"),
    "AlphaBlade_yoko": ("236P", "Alpha Blade (Horizontal)", SPECIAL, "inferred"),
    "AlphaBlade_Cancel": ("236P/K", "Alpha Blade", SPECIAL),
    "AirAlphaBlade_yoko": ("j.236P", "Air Alpha Blade (Horizontal)", SPECIAL, "inferred"),
    "AirAlphaBlade_Cancel": ("j.236P/K", "Air Alpha Blade", SPECIAL),
    "AirAlphaBlade": ("j.236K", "Air Alpha Blade (Diagonal)", SPECIAL, "inferred"),
    "BetaBlade": ("623S", "Beta Blade", SPECIAL, "inferred"),
    "AirBetaBlade": ("j.623S", "Air Beta Blade", SPECIAL, "inferred"),
    "Genrouzan": ("63214S", "Genrouzan", SPECIAL, "grab inferred"),
    "GenrouzanExe": ("63214S", "Genrouzan", SPECIAL, "grab inferred"),
    "ZanseiRouga": ("632146H", "Zansei Rouga", SUPER, "inferred"),
    "AirZanseiRouga": ("j.632146H", "Air Zansei Rouga", SUPER, "inferred"),
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
    **{k: ("214H", "Tightrope", OTHER, "nonattack") for k in
       ("ThreadRun", "ThreadRun_Reverse", "ThreadRun_Reverse_Down")},
    "ThreadRun_S": ("214H~S", "Tightrope (S attack)", SPECIAL, "inferred"),
    **{k: ("", "Wall Run", OTHER, "nonattack") for k in ("WallRun", "WallRun_jump")},
})
_add("Ramlethal", {
    "BajonetoC": ("236S", "S Bajoneto", SPECIAL, "proj"),
    "BajonetoD": ("236H", "H Bajoneto", SPECIAL, "proj"),
    "Dauro": ("623P", "Dauro", SPECIAL),
    "Ondo": ("236K", "Ondo", SPECIAL),
    "ShirudoLand": ("214K", "Sildo Detruo", SPECIAL),
    "ShirudoAir": ("j.214K", "Air Sildo Detruo", SPECIAL, "inferred"),
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
    "GarudaImpact": ("214H", "Garuda Impact", SPECIAL, "inferred"),
    "Giganter": ("632146H", "Giganter Kai", SUPER, "inferred"),
})
_add("Faust", {
    "Tsukimasu": ("41236K", "Thrust", SPECIAL),
    "NiceShot_NoHasei": ("214H", "Hole In One", SPECIAL, "inferred"),
    "NiceShot": ("41236K~4~H", "Hole in One! (from Pull Back)", SPECIAL, "proj inferred"),
    "Hikimodoshi": ("41236K~4", "Pull Back", OTHER, "nonattack inferred"),
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
    "Genkai_Land": ("632146S", "Ultimate Fortissimo", SUPER, "dustloop"),
    "Genkai_Air": ("j.632146S", "Air Ultimate Fortissimo", SUPER, "inputs"),
    "Megalomania": ("632146H", "Megalomania", SUPER, "proj inferred"),
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
    "Kabari": ("41236H", "H Kabari", SPECIAL, "inputs"),
    "HimoKabari": ("41236S", "S Kabari", SPECIAL, "inputs"),
    "KabariHasei": ("41236H~H", "Kabari Followup", SPECIAL),
    "YouZanSen": ("j.236S", "Youzansen", SPECIAL),
    "Atemi": ("236P", "Hiiragi", SPECIAL),
    "TsuraneSanzuWatashi": ("236236S", "Tsurane Sanzu-watashi", SUPER),
    "TsuraneSanzuWatashiExe": ("236236S", "Tsurane Sanzu-watashi", SUPER),
    "KenjyuLand": ("214214P", "Kenjyu", SUPER, "proj"),
    "KenjyuAir": ("j.214214P", "Air Kenjyu", SUPER, "proj inferred"),
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
    "ByakueRenshou": ("236236H", "Sickle Storm", SUPER, "dustloop"),
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
_add("Giovanna", {  # Special1/3/4 read from the logged inputs (verify_inputs.py)
    "Special2_land": ("214S", "Sol Poente", SPECIAL, "inferred"),
    "Special2_air": ("j.214S", "Air Sol Poente", SPECIAL, "inferred"),
    "Special1": ("214K", "Sepultura", SPECIAL, "inputs"),
    "Special3": ("623S", "Sol Nascente", SPECIAL, "inputs"),
    "Special4": ("236K", "Trovao", SPECIAL, "inputs"),
    "Shervi": ("214H", "Chave", SPECIAL, "inferred"),
    "LandUltimate": ("632146H", "Ventania", SUPER),
    "AirUltimate": ("j.236236H", "Tempestade", SUPER),
    "NmlAtk6D2nd": ("6HH", "6H follow-up", CMD, "inferred btn:H stance:stand"),
})

# --- characters added 2026-10-24 from Dustloop (GGST/<Char>/Data move order + inputs) and the
# opener stats (distance, airborne, follow-up chains). All `inferred`; a few checked on strips.
_add("May", {
    "IrukasanYokoC": ("[4]6S", "S Mr. Dolphin Horizontal", SPECIAL, "inferred"),
    "IrukasanYokoD": ("[4]6H", "H Mr. Dolphin Horizontal", SPECIAL, "inferred"),
    "IrukasanTateC": ("[2]8S", "S Mr. Dolphin Vertical", SPECIAL, "inferred"),
    "IrukasanTateD": ("[2]8H", "H Mr. Dolphin Vertical", SPECIAL, "inferred"),
    "IrukasanJump": ("[4]6S/H~K", "Mr. Dolphin Split", OTHER, "nonattack inferred"),
    "IrukasanEnd": ("", "Mr. Dolphin end", OTHER, "nonattack inferred"),
    "OverHeadKiss": ("623K", "Overhead Kiss", SPECIAL, "grab inferred"),
    "Arisugawa": ("214P/K", "Arisugawa Sparkle", SPECIAL, "proj inferred"),
    "Yamada": ("236236S", "Great Yamada Attack", SUPER, "proj inferred"),
    "Goshogawara": ("632146H", "The Wonderful and Dynamic Goshogawara", SUPER, "dustloop"),
    "AirGoshogawara": ("j.632146H", "Air The Wonderful and Dynamic Goshogawara", SUPER, "inferred"),
})
_add("Zato-1", {  # EddieSummonA-D = P/K/S/H Eddie attacks (Pierce / That's a lot / Leap / Oppose)
    "InviteHell": ("22H", "Invite Hell", SPECIAL, "proj inferred"),
    "EddieSummonA": ("236P", "Pierce", SPECIAL, "proj inferred command"),
    "EddieSummonB": ("236K", "That's a lot", SPECIAL, "proj inferred command"),
    "EddieSummonC": ("236S", "Leap", SPECIAL, "proj inferred command"),
    "EddieSummonD": ("236H", "Oppose", SPECIAL, "proj inferred command"),
    "EddieCommand": ("236X (Eddie out)", "Eddie command (unresolved P/K/S/H)", SPECIAL, "proj inferred command"),
    "EddieSummonOnly": ("214H", "Summon Eddie", OTHER, "nonattack inferred"),
    "EddieEnd": ("214H", "Unsummon Eddie", OTHER, "nonattack inferred"),
    "DamnedFang": ("623S", "Damned Fang", SPECIAL, "grab inferred"),
    "DamnedFangExe": ("623S", "Damned Fang", SPECIAL, "grab inferred"),
    "BreakTheLaw": ("214K", "Break The Law", SPECIAL, "inferred"),
    "DrunkerdShade": ("214S", "Drunkard Shade", SPECIAL, "inferred"),
    "Fly": ("", "Flight", OTHER, "nonattack inferred"),
    "Amorphous": ("632146H", "Amorphous", SUPER, "proj inferred"),
    "SunVoid": ("632146S", "Sun Void", SUPER, "proj inferred"),
    "SunVoid_Air": ("j.632146S", "Air Sun Void", SUPER, "proj inferred"),
    "EddieSurround": ("214P", "Float", OTHER, "nonattack inputs dustloop"),
})
_add("Jack-O", {  # Ghost = Servant; the servants' hits are detached objects
    "SummonGhostA": ("236P", "Summon Servant", SPECIAL, "proj inferred"),
    "ThrowGhost": ("hs.6X", "Throw Servant", SPECIAL, "proj inferred command"),
    "AirThrowGhost": ("hs.j.6X", "Air Throw Servant", SPECIAL, "proj inferred command"),
    "AtkOrderGhost": ("214K", "Attack Command", SPECIAL, "proj inferred command"),
    "AirAtkOrderGhost": ("j.214K", "Air Attack Command", SPECIAL, "proj inferred command"),
    "AtemiGhost": ("214S", "Defend Command", SPECIAL, "proj inferred command"),
    "JibakuGhost": ("214H", "Countdown", SPECIAL, "proj inferred command"),
    "KickGhost": ("236K", "Servant Shoot", SPECIAL, "proj inferred command"),
    "ElysionDriver": ("632146P", "Forever Elysion Driver", SUPER, "inferred"),
    "ElysionDriverEx": ("632146P", "Forever Elysion Driver", SUPER, "inferred"),
    "PowerUp": ("236236S/H", "Cheer Servant On", SUPER, "nonattack inferred"),
    # hs. = holding a Servant (picked up with D)
    "ReleaseGhost": ("hs.D", "Release Servant", OTHER, "nonattack inferred"),
    "AirReleaseGhost": ("hs.j.D", "Air Release Servant", OTHER, "nonattack inferred"),
    "ReturnGhost": ("214P", "Recover Servant", OTHER, "nonattack inputs"),
    "PickAtkOrderGhost": ("hs.214K", "Attack Command (held Servant)", SPECIAL, "proj inferred command"),
    "PickKickGhost": ("hs.236K", "Servant Shoot (held Servant)", SPECIAL, "proj inferred command"),
})
_add("Happy Chaos", {
    "HoldGunAtk": ("5H H", "Fire (At the Ready)", SPECIAL, "proj inferred"),
    "Sniper": ("214S", "Steady Aim (Fire)", SPECIAL, "proj inferred"),
    "Reload": ("22P", "Reload", OTHER, "nonattack inferred"),
    "Routine": ("214P", "Focus", OTHER, "nonattack inferred"),
    "ForwardRoll": ("214K", "Roll", OTHER, "nonattack inferred"),
    "Decoy": ("236K", "Scapegoat", OTHER, "nonattack inferred"),
    "Mark": ("236P", "Curse", SPECIAL, "proj inferred"),
    "COS_Ultimate1": ("632146S", "Deus Ex Machina", SUPER, "proj inferred"),
    "Routine_Super": ("214214P", "Super Focus", SUPER, "nonattack inputs"),
    "HoldMiss": ("5H H", "Fire (empty)", OTHER, "nonattack inferred"),
    "SniperEnd": ("214S", "Steady Aim (end)", OTHER, "nonattack inferred"),
})
_add("Testament", {  # Grave Reaper = 1-4 (S/H ground, then air); 1/2/8/9 looked at on strips
    "TST_Special1": ("236S", "S Grave Reaper", SPECIAL, "proj inferred"),
    "TST_Special2": ("236H", "H Grave Reaper", SPECIAL, "proj inferred"),
    "TST_Special3": ("j.236S", "Air S Grave Reaper", SPECIAL, "proj inferred"),
    "TST_Special4": ("j.236H", "Air H Grave Reaper", SPECIAL, "proj inferred"),
    "TST_Special5": ("214K?", "Possession? (unresolved)", SPECIAL, "inferred"),
    "TST_Special6": ("", "Testament special 6 (unresolved)", SPECIAL, "inferred"),
    "TST_Special7": ("214P", "Unholy Diver", SPECIAL, "proj inferred command"),
    "TST_Special8": ("214S", "S Arbiter Sign", SPECIAL, "proj inferred"),
    "TST_Special9": ("214H", "H Arbiter Sign", SPECIAL, "proj inferred"),
    "TST_Ultimate1": ("236236P", "Nostrovia", SUPER, "inferred"),
    "TST_Ultimate2": ("236236K", "Calamity One", SUPER, "inferred"),
})
_add("Bridget", {
    "YOYO_236S_Land": ("236S", "Stop and Dash (S)", SPECIAL, "proj inferred"),
    "YOYO_236H_Land": ("236H", "Stop and Dash (H)", SPECIAL, "proj inferred"),
    "YOYO_236S_Air": ("j.236S", "Air Stop and Dash (S)", SPECIAL, "proj inferred"),
    "YOYO_236H_Air": ("j.236H", "Air Stop and Dash (H)", SPECIAL, "proj inferred"),
    "YOYO_214S_Land": ("214S", "Stop and Dash (214S)", SPECIAL, "proj inferred"),
    "YOYO_214H_Land": ("214H", "Stop and Dash (214H)", SPECIAL, "proj inferred"),
    "YOYO_214S_Air": ("j.214S", "Air Stop and Dash (214S)", SPECIAL, "proj inferred"),
    "YOYO_214H_Air": ("j.214H", "Air Stop and Dash (214H)", SPECIAL, "proj inferred"),
    "Rolling": ("214K", "Rolling Movement", SPECIAL, "inferred"),
    "StarShip": ("623P", "Starship", SPECIAL, "inferred"),
    "KSMH": ("236K", "Kick Start My Heart", SPECIAL, "inferred"),
    "KSMH_P": ("236K~P", "Kick Start My Heart - Brake", SPECIAL, "inferred"),
    "KSMH_K": ("236K~K", "Kick Start My Heart - Shoot", SPECIAL, "inferred"),
    "FallAtk": ("j.236K", "Roger Dive", SPECIAL, "inferred"),
    "RockTheBaby_Land": ("63214P", "Rock the Baby", SPECIAL, "grab inferred"),
    "RockTheBaby_Air": ("j.63214P", "Air Rock the Baby", SPECIAL, "grab inferred"),
    "RockTheBaby_LandExe": ("63214P", "Rock the Baby", SPECIAL, "grab inferred"),
    "KillMachine_Land": ("632146H", "Return of the Killing Machine", SUPER, "proj inferred"),
    "KillMachine_Air": ("j.632146H", "Air Return of the Killing Machine", SUPER, "proj inferred"),
})
_add("Sin", {
    "BeakDriver": ("236H", "Beak Driver", SPECIAL, "inferred"),
    "BeakDriver_Add": ("236H~H", "Beak Driver Follow-Up", SPECIAL, "inferred"),
    "HawkBaker": ("623S", "Hawk Baker", SPECIAL, "inferred"),
    "HoofStamp": ("214S", "Hoof Stomp", SPECIAL, "inferred"),
    "ElkHunt": ("236K", "Elk Hunt", SPECIAL, "inferred"),
    "GazelleStep": ("214K", "Gazelle Step", OTHER, "nonattack inferred"),
    "StillGrowing": ("63214P", "Still Growing", OTHER, "nonattack inferred"),
    "RTL_Land": ("632146H", "R.T.L", SUPER, "proj inferred"),
    "TyrantBarrel": ("236236P", "Tyrant Barrel", SUPER, "inferred"),
    "ElkHunt_Add": ("236K~K", "Elk Hunt Follow-Up", SPECIAL, "inferred"),
    "HawkBaker_Add": ("623S~S", "Hawk Baker Follow-Up", SPECIAL, "inferred"),
    "HoofStamp_Add": ("214S~S", "Hoof Stomp Follow-Up", SPECIAL, "inferred"),
    "TyrantBarrel_Add": ("236236P~P", "Tyrant Barrel Follow-Up", SUPER, "inferred"),
})
_add("Bedman?", {  # Special1-5 = Dustloop order 236P, 236S, 236H (ground only), 214P, 214K
    "Special1_Land": ("236P", "call 4BA", SPECIAL, "inferred"),
    "Special1_Air": ("j.236P", "Air call 4BA", SPECIAL, "inferred"),
    "Special2_Land": ("236S", "call 4B3", SPECIAL, "inferred"),
    "Special2_Air": ("j.236S", "Air call 4B3", SPECIAL, "inferred"),
    "Special3": ("236H", "call 4B9 (Malfunction)", SPECIAL, "inferred"),
    "Special4_Land": ("214P", "call 0x$0.20", SPECIAL, "inferred"),
    "Special4_Air": ("j.214P", "Air call 0x$0.20", SPECIAL, "inferred"),
    "Special5_Land": ("214K", "call 0x$1.00", SPECIAL, "inferred"),
    "Special5_Air": ("j.214K", "Air call 0x$1.00", SPECIAL, "inferred"),
    "Ultimate1": ("632146S", "call 13C", SUPER, "inferred"),
    "Ultimate2": ("632146H", "call 4CC", SUPER, "dustloop"),
    **{f"AirMove{d}": (f"j.{d} (float)" if d else "j. (float)", "air movement", OTHER, "nonattack inferred")
       for d in ("", "1", "2", "3", "4", "6", "7", "8", "9")},
})
_add("Johnny", {  # MistFiner_Attack_A/B/C = released P/K/S; Zweihander = Vault (looked at on strips)
    "MistFiner_Attack_A": ("214]P[", "Mist Finer (Upward)", SPECIAL, "inferred"),
    "MistFiner_Attack_B": ("214]K[", "Mist Finer (Horizontal)", SPECIAL, "inferred"),
    "MistFiner_Attack_C": ("214]S[", "Mist Finer (Downward)", SPECIAL, "inferred"),
    "AirMistFiner_Attack_A": ("j.214]P[", "Air Mist Finer (Upward)", SPECIAL, "inferred"),
    "AirMistFiner_Attack_B": ("j.214]K[", "Air Mist Finer (Horizontal)", SPECIAL, "inferred"),
    "AirMistFiner_Attack_C": ("j.214]S[", "Air Mist Finer (Downward)", SPECIAL, "inferred"),
    **{k: ("214[X]", "Mist Finer stance", OTHER, "nonattack inferred") for k in
       ("MistFiner_Stance", "AirMistFiner_Stance", "MistFiner_FDash", "AirMistFiner_FDash",
        "MistFiner_BDash", "MistFiner_BWalk", "MistFiner_FWalk", "MistFiner_Cancel")},
    "ThrowCard_A": ("236P", "Deal (P)", SPECIAL, "proj inferred"),
    "ThrowCard_B": ("236K", "Deal (K)", SPECIAL, "proj inferred"),
    "ThrowCard_C": ("236S", "Deal (S)", SPECIAL, "proj inferred"),
    "Zweihander_Land": ("236H", "Vault", SPECIAL, "inferred"),
    "Zweihander_Land_Add": ("236HH", "Vault Deal", SPECIAL, "inferred"),
    "Zweihander_Air": ("j.236H", "Air Deal", SPECIAL, "proj inferred"),
    "Ensenga": ("214H", "Ensenga", SPECIAL, "inferred"),
    "SoregaOrenonada": ("632146H", "That's My Name", SUPER, "inferred"),
    "JHN_Ultimate2": ("236236S", "Joker Trick", SUPER, "inferred"),
    "SoregaOrenonada_exe": ("632146H", "That's My Name", SUPER, "inferred"),
    "JHN_Ultimate2_exe": ("236236S", "Joker Trick", SUPER, "inferred"),
})
_add("Dizzy", {  # Special1-5 = Dustloop order (fish spike, chestnuts, fish summon, Michael Sword, Wings)
    "DZY_Special1": ("236S/H", "I used this to catch fish", SPECIAL, "proj inferred"),
    "DZY_Special1_Add": ("236S~6S/236H~6H", "Ice Field", SPECIAL, "proj inferred"),
    "DZY_Special2": ("236K", "For roasting chestnuts", SPECIAL, "proj inferred"),
    "DZY_Special3": ("214P/K", "We talked a lot together", SPECIAL, "proj inferred"),
    "DZY_Special4_S": ("214S", "S Michael Sword", SPECIAL, "proj inferred"),
    "DZY_Special4_HS": ("214H", "H Michael Sword", SPECIAL, "proj inferred"),
    "DZY_Special5": ("22H", "Wings of Light", SPECIAL, "proj inferred"),
    "DZY_ULT1": ("632146S", "Imperial Ray", SUPER, "proj inferred"),
    "DZY_ULT1_exe": ("632146S", "Imperial Ray", SUPER, "proj inferred"),
    "DZY_ULT2": ("632146H", "Gamma Ray", SUPER, "proj inferred"),
})
_add("Venom", {  # 01/02 dmg 0 = ball set; 04 has Land/Air = Carcass Raid; 05 long range = Stinger Aim
    "VEN_Special_01": ("214X", "Ball Set", SPECIAL, "nonattack inferred"),
    "VEN_Special_02": ("41236K", "Ball Set (Multiple)", SPECIAL, "nonattack inferred"),
    "VEN_Special_03": ("623X", "QV", SPECIAL, "proj inferred"),
    "VEN_Special_04_S_Land": ("[2]8S", "S Carcass Raid", SPECIAL, "proj inferred"),
    "VEN_Special_04_HS_Land": ("[2]8H", "H Carcass Raid", SPECIAL, "proj inferred"),
    "VEN_Special_04_S_Air": ("j.236S", "Air S Carcass Raid", SPECIAL, "proj inferred"),
    "VEN_Special_04_HS_Air": ("j.236H", "Air H Carcass Raid", SPECIAL, "proj inferred"),
    "VEN_Special_05_S": ("[4]6S", "S Stinger Aim", SPECIAL, "proj inferred"),
    "VEN_Special_05_HS": ("[4]6H", "H Stinger Aim", SPECIAL, "proj inferred"),
    "VEN_ULT1": ("632146S", "Dark Angel", SUPER, "proj inferred"),
})
_add("Unika", {  # names follow Ky's: Edge=Stun Edge (beam), Thrust=Vapor Thrust, Dipper, Sever=Foudre Arc
    "UNI_SP00_Edge": ("236H", "A.M.T.G. Streak", SPECIAL, "proj inferred"),
    "UNI_SP01_AirEdge": ("j.236H", "A.A.T.G. Blitz", SPECIAL, "proj inferred"),
    "UNI_SP02_Thrust": ("623S", "A.C.T.G. Blast-Off", SPECIAL, "dustloop"),
    "UNI_SP03_Dipper": ("236K", "A.T.T.G. Penetrate", SPECIAL, "inferred"),
    "UNI_SP04_Sever": ("214K", "A.T.T.G. Top-Attack", SPECIAL, "inferred"),
    "UNI_SP05_Slash": ("214S", "Blaze a Trail", SPECIAL, "inferred"),
    "UNI_SpecialCancel_FDash": ("", "special cancel dash", OTHER, "nonattack"),
    "UNI_ULT00_MDBuster": ("632146S", "Megadeath Buster", SUPER, "inferred"),
    "NmlAtkC_5C": ("~5S", "5S Follow-Up (Arbalest)", CMD, "proj inferred btn:S stance:stand"),
    "NmlAtkC_2C": ("~2S", "2S Follow-Up (Arbalest)", CMD, "proj inferred btn:S stance:crouch"),
    "NmlAtkD_5D": ("~5H", "5H Follow-Up (Arbalest)", CMD, "inferred btn:H stance:stand"),
    "NmlAtkD_2D": ("~2H", "2H Follow-Up (Arbalest)", CMD, "inferred btn:H stance:crouch"),
})
_add("Jam", {  # 01-03 have air versions = Ryuujin/Gekirin/Ryuutsui; 05->07, 09->10 chains; 08 -> air specials
    "USG_Special_01": ("236K", "Ryuujin", SPECIAL, "inferred"),
    "USG_Special_01_Air": ("j.236K", "Air Ryuujin", SPECIAL, "inferred"),
    "USG_Special_02": ("236S", "Gekirin", SPECIAL, "inferred"),
    "USG_Special_02_Air": ("j.236S", "Air Gekirin", SPECIAL, "inferred"),
    "USG_Special_03": ("236H", "Ryuutsui", SPECIAL, "inferred"),
    "USG_Special_03_Air": ("j.236H", "Air Ryuutsui", SPECIAL, "inferred"),
    "USG_Special_04": ("22K/S/H", "Asanagi no Kokyuu", OTHER, "nonattack inferred"),
    "USG_Special_05": ("214S", "Bakushuu", SPECIAL, "inferred"),
    "USG_Special_06": ("214S~P/K/S/D", "Bakushuu Break", OTHER, "nonattack inferred"),
    "USG_Special_07": ("214S~H", "Hyappo Shinshou", SPECIAL, "inferred"),
    "USG_Special_08": ("214K", "Choujin", OTHER, "nonattack inferred"),
    "USG_Special_09": ("236P", "Hochifu", SPECIAL, "inferred"),
    "USG_Special_10": ("236P~6P", "Goushao", SPECIAL, "inferred"),
    "USG_ULT_01": ("632146H", "Choukyaku Kouryumon", SUPER, "inferred"),
    "USG_ULT_01_exe": ("632146H", "Choukyaku Kouryumon", SUPER, "inferred"),
    "USG_ULT_02": ("632146P", "Asanagi no Kokyuu: Shinki Kaiten", SUPER, "nonattack inputs dustloop"),
})
_add("Robo-Ky", {  # ABS_Special_01-06 = Dustloop order; 02/06 have air versions
    "ABS_Special_01": ("22H", "Gimmic-KY", SPECIAL, "proj inferred"),
    "ABS_Special_02_Land": ("236S", "Jun-KY Bargain", SPECIAL, "inferred"),
    "ABS_Special_02_Air": ("j.236S", "Air Jun-KY Bargain", SPECIAL, "inferred"),
    "ABS_Special_03": ("214S", "KY-zen Blitz", SPECIAL, "inferred"),
    "ABS_Special_04": ("63214K", "Spar-KY Surprise", SPECIAL, "grab inferred"),
    "ABS_Special_05": ("214H", "Luc-KY Launcher", SPECIAL, "inferred"),
    "ABS_Special_06_Land": ("236H", "Beautiful S-KY Today", SPECIAL, "inferred"),
    "ABS_Special_06_Air": ("j.236H", "Air Beautiful S-KY Today", SPECIAL, "inferred"),
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


# Objects whose move the attacker's action doesn't show: Happy Chaos fires during Roll
# and while moving in At the Ready (Dustloop), his action staying Roll / dash.
OBJECT_MOVES = {("Happy Chaos", "ShootingObj"): "HoldGunAtk"}


def classify(row: dict) -> dict:
    """Clean move fields for one opener row (build.py extract_openers + attacker_char).

    With the logged hit source (hit_logged): an object hit takes the move that made the object
    (category 'projectile' once the attacker has moved on), a direct hit the current action
    unless that is not an attack.
    Older logs: the hitting move is the attacker's current action unless that is not an attack
    (moving, guarding, Asuka drawing cards, Leo backturn walk ...): then a projectile/object
    from the attacker's last attack hit, and the move is that attack (category 'projectile')."""
    char = row.get("attacker_char", "")
    act = row["move_act"]
    mi = move_info(char, act)
    src = "action"
    spawn = OBJECT_MOVES.get((char, row.get("hit_object"))) or row.get("hit_object_spawn_act")
    if spawn:
        # logged hit source (logger p*_hitby): a pool object of the attacker hit, made by
        # `spawn`; a projectile when the attacker is no longer in that move
        # a summon that stays out (Servant, Eddie, Testament's crow) attacks on its owner's
        # order: the newest `command` move since it appeared made the hit
        cmd = next((c for c in row.get("hit_object_cmd_acts") or []
                    if (m := move_info(char, c)) and "command" in m[3]), None)
        cands = ([cmd] if cmd else []) + [spawn] + list(row.get("hit_object_prior_acts") or [])
        spawn = next((c for c in cands if (m := move_info(char, c)) and "nonattack" not in m[3]), spawn)
        smi = move_info(char, spawn)
        if smi:
            mi, act, src = smi, spawn, "object"
            nota, name, cat, flags = smi
            # a special's object is a projectile even while its move lasts (Venom's struck
            # ball, Dizzy's chestnut); a normal's detached hitbox (j.D) only once it is over
            if cat == SPECIAL or (spawn != row.get("attacker_act") and cat not in (SUPER, BURST, THROW)):
                cat = PROJ
            return _fields(char, act, src, (nota, name, cat, flags))
    # a direct hit from a non-attack (dash, crouch-to-stand) is still credited to the last
    # attack: the hit-by field is set on the HP-drop frame, which can trail the move
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
    elif ("proj" in flags and cat == SPECIAL and not row.get("hit_logged") and row.get("projectile_likely")
          and (row.get("attacker_act_age") or 0) > 45):
        cat = PROJ                       # the special's object hit long after the cast
    return _fields(char, act, src, (nota, name, cat, flags))


def _fields(char, act, src, mi) -> dict:
    nota, name, cat, flags = mi
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
