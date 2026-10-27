"""
UFC fighter archetype classifier.

Reads the UFCStats mirror (github.com/Greco1899/scrape_ufc_stats), cleans and validates
round-level fight stats, builds recency-weighted, sample-size-shrunk per-fighter features
with division baselines, then scores every active fighter on 10 archetypes from
MMA_Classifier.pdf and assigns a primary and secondary archetype. The PDF's "One Punch Man" is not a style
(power sits on top of any style), so it is scored separately as a 0-100 power rating.

Usage:  python archetypes.py <path-to-scrape_ufc_stats> <out_dir> [--as-of YYYY-MM-DD] [--active-months 24]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

HALF_LIFE_YEARS = 4.0          # recency weighting of fights (style evolves)
PRIOR_MIN = 25.0               # minutes of division-average "ghost" fight time added to rate stats
PRIOR_SIG_ATT = 60.0           # attempts added to striking percentages
PRIOR_TD_ATT = 8.0             # attempts added to takedown percentages
PRIOR_WINS = 3.0               # wins added to win-method shares
EARLY_HALF_LIFE_MIN = 5.0      # a KO's "early" credit halves for every 5 fight minutes elapsed (one round)
MIN_POOL = 20                  # divisions smaller than this borrow a neighbour for baselines

DIVISIONS = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
             "Middleweight", "Light Heavyweight", "Heavyweight",
             "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight"]
DIV_LIMIT = {"Flyweight": 125, "Bantamweight": 135, "Featherweight": 145, "Lightweight": 155,
             "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 265,
             "Women's Strawweight": 115, "Women's Flyweight": 125, "Women's Bantamweight": 135,
             "Women's Featherweight": 145}
BASELINE_FALLBACK = {"Women's Featherweight": "Women's Bantamweight"}

ARCHETYPES = ["Brawler", "Pressure Fighter", "Point Fighter", "Sniper",
              "Defensive Technician", "Submission Threat", "Grappler / Wrestler",
              "Point-Fighting Wrestler", "Ground-and-Pound Wrestler", "All-in-One"]
WRESTLER_FAMILY = {"Grappler / Wrestler", "Point-Fighting Wrestler", "Ground-and-Pound Wrestler"}

issues = []  # validation log

REF = Path(__file__).parent / "ref"
ROSTER_DATE = pd.Timestamp("2026-09-05")   # "as of" date of the roster snapshot


def norm_name(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s).replace("ı", "i").replace("ł", "l")).encode("ascii", "ignore").decode()
    s = s.lower().replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


ROSTER, ROSTER_ALIASES = {}, {}
if (REF / "wiki_roster_partial.txt").exists():
    for line in open(REF / "wiki_roster_partial.txt", encoding="utf-8"):
        if ":" in line:
            d, names = line.split(":", 1)
            for n in names.split(","):
                if n.strip():
                    ROSTER[norm_name(n)] = d.strip()
if (REF / "roster_aliases.json").exists():
    ROSTER_ALIASES = json.load(open(REF / "roster_aliases.json"))


def log(kind, n, note):
    issues.append({"check": kind, "rows": int(n), "note": note})


# ---------------------------------------------------------------- parsing helpers
def of_split(s):
    s = s.fillna("0 of 0").astype(str).str.strip()
    parts = s.str.extract(r"(\d+)\s*of\s*(\d+)")
    return parts[0].fillna(0).astype(int), parts[1].fillna(0).astype(int)


def mmss(s):
    s = s.fillna("--").astype(str).str.strip()
    m = s.str.extract(r"(\d+):(\d+)")
    return (m[0].astype(float) * 60 + m[1].astype(float)).fillna(0.0)


def division_of(wc):
    wc = str(wc)
    for d in sorted(DIVISIONS, key=len, reverse=True):
        if d.lower() in wc.lower():
            if not d.startswith("Women") and "women" in wc.lower():
                continue
            return d
    return None


def round_lengths(fmt):
    m = re.search(r"\(([\d\-]+)\)", str(fmt))
    if not m:
        return [5] * 5
    return [int(x) for x in m.group(1).split("-")]


def height_in(s):
    m = re.match(r"(\d+)'\s*(\d+)", str(s))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def inches(s):
    m = re.match(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else np.nan


# ---------------------------------------------------------------- load + clean
def load(src: Path):
    ev = pd.read_csv(src / "ufc_event_details.csv")
    res = pd.read_csv(src / "ufc_fight_results.csv")
    st = pd.read_csv(src / "ufc_fight_stats.csv")
    fd = pd.read_csv(src / "ufc_fighter_details.csv")
    tott = pd.read_csv(src / "ufc_fighter_tott.csv")
    for df in (ev, res, st, fd, tott):
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].str.strip()
        df.columns = [c.strip() for c in df.columns]

    ev["DATE"] = pd.to_datetime(ev["DATE"], format="%B %d, %Y", errors="coerce")
    # some events are renamed between files (e.g. "UFC Fight Night: X vs. Y" vs "Noche UFC: X vs. Y"):
    # repair by matching the headline after the colon when it is unique
    known = set(ev["EVENT"])
    tail = ev["EVENT"].str.split(":", n=1).str[-1].str.strip()
    uniq = tail[~tail.duplicated(keep=False)]
    tail_map = dict(zip(uniq, ev.loc[uniq.index, "EVENT"]))
    fix = {}
    for name in set(res["EVENT"]) - known:
        t = name.split(":", 1)[-1].strip()
        if t in tail_map:
            fix[name] = tail_map[t]
    log("event names repaired via headline match", res["EVENT"].isin(fix).sum(), str(fix))
    res["EVENT"] = res["EVENT"].replace(fix)
    st["EVENT"] = st["EVENT"].replace(fix)
    res = res.merge(ev[["EVENT", "DATE"]], on="EVENT", how="left")
    log("results missing event date", res.DATE.isna().sum(), "dropped")
    res = res.dropna(subset=["DATE"])
    res["BOUT"] = res["BOUT"].str.replace(r"\s+", " ", regex=True)
    st["BOUT"] = st["BOUT"].str.replace(r"\s+", " ", regex=True)

    dup = res.duplicated(subset=["URL"]).sum()
    log("duplicate fight URLs in results", dup, "dropped")
    res = res.drop_duplicates(subset=["URL"])

    n0 = len(st)
    st = st.dropna(subset=["FIGHTER", "ROUND"])
    log("stat rows with no fighter/round", n0 - len(st), "dropped")
    st = st.drop_duplicates(subset=["EVENT", "BOUT", "ROUND", "FIGHTER"])
    return ev, res, st, fd, tott


def build_fighter_ids(fd, tott):
    fd = fd.copy()
    fd["NAME"] = (fd["FIRST"].fillna("") + " " + fd["LAST"].fillna("")).str.strip().str.replace(r"\s+", " ", regex=True)
    tott = tott.rename(columns={"FIGHTER": "NAME_T"})
    f = fd.merge(tott, on="URL", how="left")
    f["ID"] = f["URL"].str.extract(r"fighter-details/(\w+)")
    f["WEIGHT_LBS"] = f["WEIGHT"].map(inches)
    f["HEIGHT_IN"] = f["HEIGHT"].map(height_in)
    f["REACH_IN"] = f["REACH"].map(inches)
    f["DOB_DT"] = pd.to_datetime(f["DOB"], format="%b %d, %Y", errors="coerce")
    return f


# stats/results spellings that differ from the fighter-profile name (validated by hand)
NAME_ALIASES = {"Kai Kamaka": "Kai Kamaka III", "Bibulatov Magomed": "Magomed Bibulatov",
                "Patricio Freire": "Patricio Pitbull", "Rafael Cerquiera": "Rafael Cerqueira",
                "Nick Galanti": "Nicholas Galanti"}


def resolve_name(name, division, f_by_name):
    name = NAME_ALIASES.get(name, name)
    cands = f_by_name.get(name)
    if cands is None:
        return None
    if len(cands) == 1:
        return cands[0][0]
    lim = DIV_LIMIT.get(division, np.nan)
    best = min(cands, key=lambda c: abs((c[1] if not np.isnan(c[1]) else 999) - (lim if not np.isnan(lim) else 0)))
    return best[0]


def fighter_rounds(res, st, fighters):
    """Round-level rows: one row per fighter per round with opponent stats attached."""
    r = res.copy()
    r["DIV"] = r["WEIGHTCLASS"].map(division_of)
    r["N_ROUNDS_FOUGHT"] = pd.to_numeric(r["ROUND"], errors="coerce")
    r["END_SEC"] = mmss(r["TIME"])
    r["RLEN"] = r["TIME FORMAT"].map(round_lengths)

    s = st.copy()
    for col in ["SIG.STR.", "TOTAL STR.", "TD", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]:
        s[col + "_L"], s[col + "_A"] = of_split(s[col])
    bad = (s["SIG.STR._L"] > s["SIG.STR._A"]).sum() + (s["TD_L"] > s["TD_A"]).sum()
    log("landed > attempted", bad, "kept (none expected)")
    s["CTRL_S"] = mmss(s["CTRL"])
    for c in ["KD", "SUB.ATT", "REV."]:
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0)
    s["RND"] = s["ROUND"].str.extract(r"(\d+)").astype(int)

    tgt = s["HEAD_L"] + s["BODY_L"] + s["LEG_L"]
    pos = s["DISTANCE_L"] + s["CLINCH_L"] + s["GROUND_L"]
    log("target split != sig landed", (tgt != s["SIG.STR._L"]).sum(), "kept; sig total used as truth")
    log("position split != sig landed", (pos != s["SIG.STR._L"]).sum(), "kept; sig total used as truth")

    s = s.merge(r[["EVENT", "BOUT", "URL", "DATE", "DIV", "WEIGHTCLASS", "N_ROUNDS_FOUGHT", "END_SEC", "RLEN"]],
                on=["EVENT", "BOUT"], how="inner")
    log("stat rows unmatched to a result", len(st) - len(s), "dropped")

    def rsec(row):
        rl = row.RLEN
        L = (rl[row.RND - 1] if row.RND - 1 < len(rl) else rl[-1]) * 60
        return row.END_SEC if row.RND == row.N_ROUNDS_FOUGHT else L
    s["RSEC"] = s.apply(rsec, axis=1).astype(float)
    over = s["CTRL_S"] > s["RSEC"] + 1
    log("control time > round time", over.sum(), "clipped to round time")
    s.loc[over, "CTRL_S"] = s.loc[over, "RSEC"]
    zero = s["RSEC"] <= 0
    log("rounds with zero/unknown duration", zero.sum(), "dropped")
    s = s[~zero]

    # exactly two fighters per round
    cnt = s.groupby(["URL", "RND"])["FIGHTER"].transform("nunique")
    log("rounds without exactly 2 fighters", (cnt != 2).sum(), "dropped")
    s = s[cnt == 2]

    keep = ["URL", "RND", "FIGHTER", "KD", "SIG.STR._L", "SIG.STR._A", "TOTAL STR._L", "TOTAL STR._A",
            "TD_L", "TD_A", "SUB.ATT", "REV.", "CTRL_S", "HEAD_L", "BODY_L", "LEG_L", "DISTANCE_L",
            "DISTANCE_A", "CLINCH_L", "CLINCH_A", "GROUND_L", "GROUND_A", "LEG_A"]
    opp = s[keep].rename(columns={c: "O_" + c for c in keep if c not in ("URL", "RND")})
    s = s.merge(opp, on=["URL", "RND"])
    s = s[s["FIGHTER"] != s["O_FIGHTER"]]

    f_by_name = {}
    for _, x in fighters.iterrows():
        f_by_name.setdefault(x["NAME"], []).append((x["ID"], x["WEIGHT_LBS"]))
    key = s[["FIGHTER", "DIV"]].drop_duplicates()
    key["FID"] = [resolve_name(n, d, f_by_name) for n, d in zip(key.FIGHTER, key.DIV)]
    s = s.merge(key, on=["FIGHTER", "DIV"], how="left")
    log("stat rows whose fighter name has no profile", s["FID"].isna().sum(), "dropped")
    return s.dropna(subset=["FID"]), r


def fight_results_long(r, fighters):
    """One row per fighter per fight with result + method (covers fights without round stats)."""
    f_by_name = {}
    for _, x in fighters.iterrows():
        f_by_name.setdefault(x["NAME"], []).append((x["ID"], x["WEIGHT_LBS"]))
    rows = []
    for _, x in r.iterrows():
        parts = str(x["BOUT"]).split(" vs. ")
        if len(parts) != 2:
            continue
        oc = str(x["OUTCOME"]).split("/")
        if len(oc) != 2:
            continue
        method = str(x["METHOD"])
        det = str(x["DETAILS"])
        # elapsed fight time at the finish (seconds from the opening bell)
        er, rl = x["N_ROUNDS_FOUGHT"], x["RLEN"]
        fin = (sum((rl + rl[-1:] * 10)[:int(er) - 1]) * 60 + x["END_SEC"]) if pd.notna(er) and x["END_SEC"] > 0 else np.nan
        for i in (0, 1):
            rows.append({
                "URL": x["URL"], "DATE": x["DATE"], "EVENT": x["EVENT"], "DIV": x["DIV"],
                "WEIGHTCLASS": x["WEIGHTCLASS"], "FIGHTER": parts[i], "OPP": parts[1 - i],
                "RES": oc[i], "METHOD": method, "DETAILS": det,
                "END_ROUND": x["N_ROUNDS_FOUGHT"], "FIN_SEC": fin, "TITLE": "Title" in str(x["WEIGHTCLASS"]),
                "FID": resolve_name(parts[i], x["DIV"], f_by_name),
                "OPP_FID": resolve_name(parts[1 - i], x["DIV"], f_by_name),
            })
    L = pd.DataFrame(rows)
    log("result rows with unresolvable fighter", L["FID"].isna().sum(), "dropped")
    L = L.dropna(subset=["FID"])
    m = L["METHOD"]
    L["IS_KO"] = m.str.contains("KO/TKO|Doctor", regex=True)
    L["IS_SUB"] = m.str.startswith("Submission")
    L["IS_DEC"] = m.str.startswith("Decision")
    # ground-and-pound finish: KO/TKO whose details say the strikes came from a ground position
    L["IS_GNP_KO"] = L["IS_KO"] & L["DETAILS"].str.contains(r"From (?:Mount|Half Guard|Guard|Back Control|Side Control|North-South|Crucifix|Ground|Top)|Ground", regex=True)
    # UFCStats has no KO vs TKO field; the finishing technique is singular for a one-shot KO ("Punch to Head")
    # and plural for a flurry/TKO ("Punches to Head From Mount"). Cuts, injuries, corner and doctor stoppages are neither.
    ko = m.eq("KO/TKO")
    L["IS_STRIKE_KO"] = ko & L["DETAILS"].str.match(r"(?:Punch(?:es)?|Kicks?|Knees?|Elbows?|Flying|Spinning|Headbutts?|Slam)\b")
    L["IS_CLEAN_KO"] = ko & L["DETAILS"].str.match(r"(?:Punch|Kick|Knee|Elbow|Flying|Spinning|Headbutt|Slam)\b")
    # early credit: 1.0 at the opening bell, halving every EARLY_HALF_LIFE_MIN minutes (round midpoint if time missing)
    fin_min = (L["FIN_SEC"] / 60).fillna((L["END_ROUND"] - 0.5) * 5)
    L["EARLY"] = 0.5 ** (fin_min / EARLY_HALF_LIFE_MIN)
    return L


# ---------------------------------------------------------------- features
def wavg_sum(df, cols, w):
    return (df[cols].multiply(w, axis=0)).sum()


def build_features(rounds, L, fighters, as_of, active_months):
    as_of = pd.Timestamp(as_of)
    rounds = rounds.copy()
    rounds["W"] = 0.5 ** (((as_of - rounds["DATE"]).dt.days / 365.25) / HALF_LIFE_YEARS)
    L = L.copy()
    L["W"] = 0.5 ** (((as_of - L["DATE"]).dt.days / 365.25) / HALF_LIFE_YEARS)

    # ---- division: most recent non-catch/open-weight bout
    Ld = L.dropna(subset=["DIV"]).sort_values("DATE")
    div = Ld.groupby("FID")["DIV"].last()

    # ---- active roster
    # 1) official roster snapshot where we have one (ref/wiki_roster_partial.txt) -> authoritative for those divisions,
    #    plus anyone who has fought since the snapshot date (new signings).
    # 2) other divisions: anyone who fought in the last 24 months (~0.98 recall on the snapshot divisions);
    #    a loss in the last fight more than 12 months ago is flagged as a possible release rather than dropped.
    ufc = L[~L["EVENT"].str.startswith("Road to UFC")].sort_values("DATE")
    g = ufc.groupby("FID")
    st = pd.DataFrame({"last": g["DATE"].max(), "last_res": g["RES"].last(),
                       "name": g["FIGHTER"].last()})
    st["div"] = div
    st["mo"] = (as_of - st["last"]).dt.days / 30.44
    st["n"] = st["name"].map(norm_name)
    st["n"] = st["n"].map(lambda x: ROSTER_ALIASES.get(x, x))
    roster = {ROSTER_ALIASES.get(k, k): v for k, v in ROSTER.items()}
    covered = set(roster.values())
    on_roster = st["n"].isin(roster)
    since_snap = st["last"] >= ROSTER_DATE
    recent = st["mo"] <= 24
    likely_cut = (st["mo"] > 12) & (st["last_res"] == "L")
    st["roster_source"] = np.where(on_roster, "official roster",
                          np.where(st["div"].isin(covered), np.where(since_snap, "fought since roster snapshot", ""),
                                   np.where(recent & likely_cut, "recent activity (may be released)",
                                            np.where(recent, "recent activity", ""))))
    st = st[st["roster_source"] != ""]
    active_ids = st.index
    roster_src = st["roster_source"]

    # ---- round-level aggregates
    R = rounds.copy()
    R["OPEN_S"] = (R["RSEC"] - R["CTRL_S"] - R["O_CTRL_S"]).clip(lower=0)
    stat_cols = ["RSEC", "OPEN_S", "KD", "SIG.STR._L", "SIG.STR._A", "TOTAL STR._L", "TOTAL STR._A",
                 "TD_L", "TD_A", "SUB.ATT", "REV.", "CTRL_S", "HEAD_L", "BODY_L", "LEG_L", "LEG_A",
                 "DISTANCE_L", "DISTANCE_A", "CLINCH_L", "CLINCH_A", "GROUND_L", "GROUND_A",
                 "O_KD", "O_SIG.STR._L", "O_SIG.STR._A", "O_TD_L", "O_TD_A", "O_CTRL_S", "O_SUB.ATT",
                 "O_DISTANCE_L", "O_DISTANCE_A", "O_GROUND_L", "O_LEG_L"]
    Rw = R[stat_cols].multiply(R["W"], axis=0)
    Rw["FID"] = R["FID"]
    T = Rw.groupby("FID").sum()
    T["RAW_MIN"] = R.groupby("FID")["RSEC"].sum() / 60
    T["STAT_FIGHTS"] = R.groupby("FID")["URL"].nunique()

    # pace retention: sig attempts per minute in rounds 2+ vs round 1 (only fights reaching round 2)
    reach2 = R.groupby(["FID", "URL"])["RND"].transform("max") >= 2
    Rp = R[reach2]
    r1 = Rp[Rp.RND == 1].groupby("FID").apply(lambda d: (d["SIG.STR._A"] * d.W).sum() / max((d.RSEC * d.W).sum() / 60, 1e-9), include_groups=False)
    rl = Rp[Rp.RND >= 2].groupby("FID").apply(lambda d: (d["SIG.STR._A"] * d.W).sum() / max((d.RSEC * d.W).sum() / 60, 1e-9), include_groups=False)
    late_min = Rp[Rp.RND >= 2].groupby("FID")["RSEC"].sum() / 60
    T["LATE_RATIO_RAW"] = (rl / r1.replace(0, np.nan))
    T["LATE_MIN"] = late_min

    # ---- result aggregates (all UFC fights, unweighted counts + weighted shares)
    g = L.groupby("FID")
    Rz = pd.DataFrame({
        "UFC_FIGHTS": g.size(),
        "W": g.apply(lambda d: (d.RES == "W").sum(), include_groups=False),
        "L": g.apply(lambda d: (d.RES == "L").sum(), include_groups=False),
        "D": g.apply(lambda d: (d.RES == "D").sum(), include_groups=False),
        "NC": g.apply(lambda d: (d.RES == "NC").sum(), include_groups=False),
        "KO_W": g.apply(lambda d: ((d.RES == "W") & d.IS_KO).sum(), include_groups=False),
        "SUB_W": g.apply(lambda d: ((d.RES == "W") & d.IS_SUB).sum(), include_groups=False),
        "DEC_W": g.apply(lambda d: ((d.RES == "W") & d.IS_DEC).sum(), include_groups=False),
        "GNP_KO_W": g.apply(lambda d: ((d.RES == "W") & d.IS_GNP_KO).sum(), include_groups=False),
        "KO_L": g.apply(lambda d: ((d.RES == "L") & d.IS_KO).sum(), include_groups=False),
        "SUB_L": g.apply(lambda d: ((d.RES == "L") & d.IS_SUB).sum(), include_groups=False),
        "DEC_L": g.apply(lambda d: ((d.RES == "L") & d.IS_DEC).sum(), include_groups=False),
        "R1_FIN_W": g.apply(lambda d: ((d.RES == "W") & (d.IS_KO | d.IS_SUB) & (d.END_ROUND == 1)).sum(), include_groups=False),
        "WW": g.apply(lambda d: (d.W * (d.RES == "W")).sum(), include_groups=False),
        "WKO": g.apply(lambda d: (d.W * ((d.RES == "W") & d.IS_KO)).sum(), include_groups=False),
        "WSUB": g.apply(lambda d: (d.W * ((d.RES == "W") & d.IS_SUB)).sum(), include_groups=False),
        "WDEC": g.apply(lambda d: (d.W * ((d.RES == "W") & d.IS_DEC)).sum(), include_groups=False),
        "WF": g.apply(lambda d: d.W.sum(), include_groups=False),
        # knockout power (recency-weighted): clean one-shot KOs worth 1 + early credit, strike KO/TKOs worth their early credit
        "WCKO": g.apply(lambda d: (d.W * ((d.RES == "W") & d.IS_CLEAN_KO) * (1 + d.EARLY)).sum(), include_groups=False),
        "WEKO": g.apply(lambda d: (d.W * ((d.RES == "W") & d.IS_STRIKE_KO) * d.EARLY).sum(), include_groups=False),
        "WSKO": g.apply(lambda d: (d.W * ((d.RES == "W") & d.IS_STRIKE_KO)).sum(), include_groups=False),
        "WDECF": g.apply(lambda d: (d.W * d.IS_DEC).sum(), include_groups=False),
        "FIRST_UFC": g["DATE"].min(), "LAST_UFC": g["DATE"].max(),
        "LAST5": g.apply(lambda d: "".join(d.sort_values("DATE", ascending=False).RES.str[0].head(5)), include_groups=False),
        "TITLE_FIGHTS": g["TITLE"].sum(),
    })

    F = fighters.set_index("ID")[["NAME", "NICKNAME", "HEIGHT_IN", "REACH_IN", "WEIGHT_LBS", "STANCE", "DOB_DT"]]
    D = F.join(Rz, how="inner").join(T, how="left")
    D["DIV"] = div
    D["ROSTER_SOURCE"] = roster_src
    D = D.loc[D.index.intersection(active_ids)]
    D = D[D["DIV"].notna()]
    D["AGE"] = ((as_of - D["DOB_DT"]).dt.days / 365.25).round(1)
    D["APE"] = D["REACH_IN"] - D["HEIGHT_IN"]
    for c in T.columns:
        if c not in ("LATE_RATIO_RAW",):
            D[c] = D[c].fillna(0)
    D["BASE_DIV"] = D["DIV"]
    for d in D["DIV"].unique():
        if (D["DIV"] == d).sum() < MIN_POOL and d in BASELINE_FALLBACK:
            D.loc[D["DIV"] == d, "BASE_DIV"] = BASELINE_FALLBACK[d]
    return D


def shrink_rate(D, num, den_min, per=1.0):
    """rate per minute*per, shrunk toward division pooled rate with PRIOR_MIN ghost minutes."""
    out = pd.Series(index=D.index, dtype=float)
    for bd, grp in D.groupby("BASE_DIV"):
        pool = D[D["BASE_DIV"] == bd]
        prior = pool[num].sum() / max(pool[den_min].sum(), 1e-9)
        out[grp.index] = (grp[num] + prior * PRIOR_MIN * 60) / (grp[den_min] + PRIOR_MIN * 60)
    return out * 60 * per


def shrink_pct(D, num, den, k):
    out = pd.Series(index=D.index, dtype=float)
    for bd, grp in D.groupby("BASE_DIV"):
        pool = D[D["BASE_DIV"] == bd]
        prior = pool[num].sum() / max(pool[den].sum(), 1e-9)
        out[grp.index] = (grp[num] + prior * k) / (grp[den] + k)
    return out


def metrics(D):
    M = pd.DataFrame(index=D.index)
    sec = "RSEC"
    M["slpm"] = shrink_rate(D, "SIG.STR._L", sec)
    M["sapm"] = shrink_rate(D, "O_SIG.STR._L", sec)
    M["sig_att_pm"] = shrink_rate(D, "SIG.STR._A", sec)
    M["opp_sig_att_pm"] = shrink_rate(D, "O_SIG.STR._A", sec)
    M["total_str_pm"] = shrink_rate(D, "TOTAL STR._A", sec)
    M["sig_acc"] = shrink_pct(D, "SIG.STR._L", "SIG.STR._A", PRIOR_SIG_ATT)
    D["_O_MISS"] = D["O_SIG.STR._A"] - D["O_SIG.STR._L"]
    M["sig_def"] = shrink_pct(D, "_O_MISS", "O_SIG.STR._A", PRIOR_SIG_ATT)
    M["str_diff"] = M["slpm"] - M["sapm"]
    # distance striking normalised by open (non-control) time -> wrestlers aren't flattered/penalised
    M["dist_att_open_pm"] = shrink_rate(D, "DISTANCE_A", "OPEN_S")
    M["dist_land_open_pm"] = shrink_rate(D, "DISTANCE_L", "OPEN_S")
    M["dist_abs_open_pm"] = shrink_rate(D, "O_DISTANCE_L", "OPEN_S")
    M["dist_acc"] = shrink_pct(D, "DISTANCE_L", "DISTANCE_A", PRIOR_SIG_ATT)
    D["_O_DMISS"] = D["O_DISTANCE_A"] - D["O_DISTANCE_L"]
    M["dist_def"] = shrink_pct(D, "_O_DMISS", "O_DISTANCE_A", PRIOR_SIG_ATT)
    M["kd15"] = shrink_rate(D, "KD", sec, 15)
    M["kd_abs15"] = shrink_rate(D, "O_KD", sec, 15)
    M["kd_per_100_landed"] = shrink_pct(D, "KD", "SIG.STR._L", 150) * 100
    M["td15"] = shrink_rate(D, "TD_L", sec, 15)
    M["td_att15"] = shrink_rate(D, "TD_A", sec, 15)
    M["td_acc"] = shrink_pct(D, "TD_L", "TD_A", PRIOR_TD_ATT)
    D["_O_TDMISS"] = D["O_TD_A"] - D["O_TD_L"]
    M["td_def"] = shrink_pct(D, "_O_TDMISS", "O_TD_A", PRIOR_TD_ATT)
    M["opp_td_att15"] = shrink_rate(D, "O_TD_A", sec, 15)
    M["sub15"] = shrink_rate(D, "SUB.ATT", sec, 15)
    M["opp_sub15"] = shrink_rate(D, "O_SUB.ATT", sec, 15)
    M["rev15"] = shrink_rate(D, "REV.", sec, 15)
    M["ctrl_pct"] = shrink_rate(D, "CTRL_S", sec) / 60
    M["ctrl_against_pct"] = shrink_rate(D, "O_CTRL_S", sec) / 60
    M["gnd_land15"] = shrink_rate(D, "GROUND_L", sec, 15)
    M["gnd_land_per_ctrl_min"] = shrink_pct(D, "GROUND_L", "CTRL_S", 300) * 60
    M["clinch_share"] = shrink_pct(D, "CLINCH_L", "SIG.STR._L", 40)
    M["ground_share"] = shrink_pct(D, "GROUND_L", "SIG.STR._L", 40)
    M["leg_pm"] = shrink_rate(D, "LEG_L", sec)
    M["leg_share"] = shrink_pct(D, "LEG_L", "SIG.STR._L", 40)
    M["body_share"] = shrink_pct(D, "BODY_L", "SIG.STR._L", 40)
    M["head_share"] = shrink_pct(D, "HEAD_L", "SIG.STR._L", 40)
    M["open_time_pct"] = shrink_pct(D, "OPEN_S", "RSEC", 600)
    # pace retention (shrunk toward 1.0 with 15 late minutes)
    lr = D["LATE_RATIO_RAW"].fillna(1.0).clip(0.2, 3.0)
    M["late_output_ratio"] = (lr * D["LATE_MIN"] + 1.0 * 15) / (D["LATE_MIN"] + 15)
    # results
    M["ko_win_share"] = shrink_pct(D, "WKO", "WW", PRIOR_WINS)
    M["sub_win_share"] = shrink_pct(D, "WSUB", "WW", PRIOR_WINS)
    M["dec_win_share"] = shrink_pct(D, "WDEC", "WW", PRIOR_WINS)
    M["finish_win_share"] = M["ko_win_share"] + M["sub_win_share"]
    M["decision_fight_rate"] = shrink_pct(D, "WDECF", "WF", 3)
    M["r1_finish_rate"] = (D["R1_FIN_W"] + 0.15 * 2) / (D["W"] + 2)
    M["avg_fight_min"] = (D["RAW_MIN"] + 11 * 2) / (D["STAT_FIGHTS"] + 2)
    M["ko_loss_rate"] = (D["KO_L"] + 0.12 * 3) / (D["UFC_FIGHTS"] + 3)
    M["clean_ko_pts"] = shrink_pct(D, "WCKO", "WF", 3)              # per fight; a 1st-minute one-shot KO ~2, a late one ~1
    M["early_ko_rate"] = shrink_pct(D, "WEKO", "WF", 3)             # per fight; strike KO/TKOs discounted by fight time elapsed
    M["ko_per_100_landed"] = shrink_pct(D, "WSKO", "SIG.STR._L", 150) * 100
    M["ko_wins"] = D["KO_W"]
    M["sub_wins"] = D["SUB_W"]
    M["gnp_ko_wins"] = D["GNP_KO_W"]
    M["gnp_ko_share"] = (D["GNP_KO_W"] + 0.1 * 2) / (D["KO_W"] + 2)
    return M


# ---------------------------------------------------------------- archetype scoring
def pct_by_div(M, D):
    P = pd.DataFrame(index=M.index)
    for c in M.columns:
        P[c] = M.groupby(D["BASE_DIV"])[c].rank(pct=True)
    return P


def power_score(P):
    """Knockout power, separate from style: clean one-shot KOs (bonus if early), early KO/TKOs, knockdowns,
    KOs per strike landed, short fights. Returned as a 0-100 rank against all active fighters."""
    raw = (.30 * P.clean_ko_pts + .25 * P.early_ko_rate + .20 * P.kd15 + .15 * P.ko_per_100_landed
           + .10 * (1 - P.avg_fight_min))
    return raw.rank(pct=True) * 100


def score(P):
    I = 1 - P
    G = (P.td15 + P.ctrl_pct + P.sub15) / 3            # grappling emphasis
    S = 1 - G                                           # striking emphasis
    sc = pd.DataFrame(index=P.index)
    sc["Brawler"] = (.26 * P.sig_att_pm + .14 * P.slpm + .18 * P.sapm + .10 * I.sig_def + .08 * P.kd15
                     + .06 * P.ko_win_share + .18 * S)
    sc["Pressure Fighter"] = (.22 * P.slpm + .14 * P.sig_att_pm + .20 * P.late_output_ratio + .10 * P.total_str_pm
                              + .10 * P.str_diff + .08 * P.clinch_share + .08 * P.td_att15 + .08 * P.avg_fight_min)
    sc["Point Fighter"] = (.18 * P.slpm + .13 * P.sig_acc + .18 * P.str_diff + .09 * I.sapm + .09 * P.sig_def
                           + .13 * P.dec_win_share + .08 * I.ko_win_share + .12 * S)
    sc["Sniper"] = (.26 * P.dist_acc + .18 * P.str_diff + .14 * P.kd_per_100_landed + .10 * I.sapm
                    + .08 * P.ko_win_share + .10 * I.sig_att_pm + .14 * S)
    sc["Defensive Technician"] = (.30 * P.dist_def + .22 * I.dist_abs_open_pm + .14 * P.str_diff + .08 * P.sig_acc
                                  + .10 * I.kd_abs15 + .06 * P.opp_sig_att_pm + .10 * S)
    sc["Submission Threat"] = (.38 * P.sub15 + .20 * P.sub_win_share + .18 * P.sub_wins + .10 * P.td15
                               + .08 * P.ctrl_pct + .06 * P.rev15)
    sc["Grappler / Wrestler"] = (.28 * P.td15 + .10 * P.td_acc + .28 * P.ctrl_pct + .12 * P.td_att15
                                 + .10 * P.td_def + .06 * P.sub15 + .06 * I.ctrl_against_pct)
    sc["Point-Fighting Wrestler"] = (.24 * P.td15 + .28 * P.ctrl_pct + .20 * P.dec_win_share + .10 * I.finish_win_share
                                     + .08 * P.td_acc + .10 * I.gnd_land_per_ctrl_min)
    sc["Ground-and-Pound Wrestler"] = (.22 * P.td15 + .18 * P.ctrl_pct + .26 * P.gnd_land15 + .12 * P.gnd_land_per_ctrl_min
                                       + .12 * P.gnp_ko_wins + .10 * P.ko_win_share)
    # wrestler archetypes require actual wrestling: gate on takedown rate + control share, so scramble-heavy
    # strikers (ground strikes in finishing sequences) or decision-winning strikers don't read as wrestlers
    Wcore = (P.td15 + P.ctrl_pct + P.td_att15) / 3
    gate = 0.35 + 0.65 * ((Wcore - 0.45) / 0.35).clip(0, 1)
    for a in WRESTLER_FAMILY:
        sc[a] = sc[a] * gate
    cats = pd.DataFrame({
        "strike": (P.sig_acc + P.sig_def + P.str_diff) / 3,
        "wrestle": (P.td15 + P.td_acc) / 2,
        "tdd": P.td_def,
        "grapple": (P.sub15 + P.ctrl_pct) / 2,
    })
    sc["All-in-One"] = .5 * cats.mean(axis=1) + .5 * cats.min(axis=1)
    return sc, cats


def classify(sc, cats, D):
    Z = (sc - sc.mean()) / sc.std()                # put archetypes on a common scale
    fit = sc.rank(pct=True) * 100                  # 0-100 fit vs all active fighters
    aio_ok = (cats.min(axis=1) >= 0.55) & (cats.mean(axis=1) >= 0.65)
    out = []
    for fid in sc.index:
        z = Z.loc[fid].copy()
        if not aio_ok[fid]:
            z["All-in-One"] = -np.inf
        order = z.sort_values(ascending=False)
        prim = order.index[0]
        sec = None
        for a in order.index[1:]:
            if prim in WRESTLER_FAMILY and a in WRESTLER_FAMILY:
                continue
            sec = a
            break
        margin = order.iloc[0] - order.iloc[1]
        n, mins = D.loc[fid, "UFC_FIGHTS"], D.loc[fid, "RAW_MIN"]
        if n < 3 or mins < 15:
            conf = "Low"
        elif n >= 5 and mins >= 50 and margin >= 0.25:
            conf = "High"
        else:
            conf = "Medium"
        out.append({"FID": fid, "primary": prim, "secondary": sec, "primary_z": round(order.iloc[0], 2),
                    "margin_z": round(margin, 2), "confidence": conf, "all_in_one_flag": bool(aio_ok[fid])})
    return pd.DataFrame(out).set_index("FID"), fit


LABELS = {
    "slpm": "Sig. strikes landed/min", "sapm": "Low damage absorbed", "sig_acc": "Striking accuracy",
    "sig_def": "Striking defense", "str_diff": "Striking differential", "kd15": "Knockdowns/15",
    "kd_per_100_landed": "Power per landed strike", "td15": "Takedowns/15", "td_acc": "Takedown accuracy",
    "td_def": "Takedown defense", "sub15": "Sub attempts/15", "ctrl_pct": "Control time share",
    "gnd_land15": "Ground strikes/15", "late_output_ratio": "Late-round output retention",
    "ko_loss_rate": "Durability (rarely KO'd)", "kd_abs15": "Rarely knocked down", "ctrl_against_pct": "Hard to control",
    "sig_att_pm": "Striking volume", "leg_pm": "Leg kicks landed/min", "clinch_share": "Clinch striking share",
}
GOOD_LOW = {"sapm", "ko_loss_rate", "kd_abs15", "ctrl_against_pct"}


WEAK = {"sapm": "Absorbs heavy volume", "ko_loss_rate": "Has been KO'd often", "kd_abs15": "Gets knocked down",
        "ctrl_against_pct": "Gets controlled", "late_output_ratio": "Output fades late", "kd15": "Little knockdown power",
        "kd_per_100_landed": "Little power per strike", "sig_att_pm": "Low striking volume"}


def weak_label(k):
    return WEAK.get(k, "Low " + LABELS[k][0].lower() + LABELS[k][1:])


def strengths_weaknesses(P):
    sw = {}
    for fid, row in P.iterrows():
        vals = {}
        for k in LABELS:
            v = row[k]
            vals[k] = 1 - v if k in GOOD_LOW else v
        s = sorted(vals.items(), key=lambda x: -x[1])
        sw[fid] = ("; ".join(LABELS[k] for k, v in s[:3] if v >= .7),
                   "; ".join(weak_label(k) for k, v in s[::-1][:3] if v <= .3))
    return sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--active-months", type=int, default=24)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ev, res, st, fd, tott = load(src)
    fighters = build_fighter_ids(fd, tott)
    rounds, r = fighter_rounds(res, st, fighters)
    L = fight_results_long(r, fighters)
    D = build_features(rounds, L, fighters, a.as_of, a.active_months)
    M = metrics(D)
    P = pct_by_div(M, D)
    sc, cats = score(P)
    C, fit = classify(sc, cats, D)
    C["power"] = power_score(P).round(1)
    sw = strengths_weaknesses(P)

    base = D[["NAME", "NICKNAME", "DIV", "ROSTER_SOURCE", "AGE", "HEIGHT_IN", "REACH_IN", "APE", "STANCE", "UFC_FIGHTS", "W", "L", "D", "NC",
              "KO_W", "SUB_W", "DEC_W", "GNP_KO_W", "KO_L", "SUB_L", "DEC_L", "R1_FIN_W", "TITLE_FIGHTS",
              "FIRST_UFC", "LAST_UFC", "LAST5", "RAW_MIN", "STAT_FIGHTS"]].copy()
    base["RAW_MIN"] = base["RAW_MIN"].round(1)
    base = base.join(C)
    base["strengths"] = [sw[i][0] for i in base.index]
    base["weaknesses"] = [sw[i][1] for i in base.index]
    full = base.join(M.round(3)).join(P.add_prefix("pct_").round(3)).join(fit.add_prefix("fit_").round(1))
    full.index.name = "fighter_id"
    full = full.sort_values(["DIV", "NAME"])
    full.to_csv(out / "fighter_archetypes.csv")
    pd.DataFrame(issues).to_csv(out / "validation_log.csv", index=False)
    rounds.to_csv(out / "round_stats_clean.csv.gz", index=False, compression="gzip")
    L.to_csv(out / "fight_results_clean.csv", index=False)
    print(pd.DataFrame(issues).to_string())
    print(len(full), "active fighters classified")
    print(full.primary.value_counts())


if __name__ == "__main__":
    main()
