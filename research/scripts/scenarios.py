"""A scenario file: named matching cases with known answers, and a scorecard.

    python research/scripts/scenarios.py build      # writes examples/scenarios/
    python research/scripts/scenarios.py score [variant ...]   # trains custmatch, scores each scenario
                                                      # writes research/results/scenarios.json

The file is a US customer file in two sources (crm, billing). It holds:

  background   1,000 people, 2 to 3 records each, ordinary noise (typos, nicknames, missing
               fields, format differences), in households: couples sharing an address and
               often a phone or a family email, some with grown children. custmatch is
               trained on labels from these records only.
  scenarios    small groups of records, each built to test one thing that goes wrong in
               customer matching, with the answer for every pair: same person or not. They
               are held out of training, so the scorecard shows what the model generalises to.

Every scenario appears REPEATS times with different people, so a pass is not luck on one
name. A scenario passes a repeat when every expected pair comes out right (same customer, or
different customers); the scorecard gives the share of repeats passed.
"""
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "examples/scenarios"
RESULTS = ROOT / "research/results/scenarios.json"
BACKGROUND = 1000
REPEATS = 10
SEEDS = [1, 2, 3]
VARIANTS = {"default": {}, "twin_guard": {"twin_guard": True},
            "match_weight": {"match_weight_feature": True}, "active_match_weight": {"match_weight_feature": True, "_active": True},
            # 500 random labels, then 1,000 chosen by active learning, 100 at a time
            "active": {"_active": True}, "twin_guard_active": {"twin_guard": True, "_active": True}}

FIRST_M = ["james", "john", "robert", "michael", "william", "david", "richard", "joseph", "thomas", "charles",
           "christopher", "daniel", "matthew", "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
           "kenneth", "kevin", "brian", "george", "timothy", "ronald", "edward", "jason", "jeffrey", "ryan"]
FIRST_F = ["mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", "jessica", "sarah", "karen",
           "lisa", "nancy", "betty", "margaret", "sandra", "ashley", "kimberly", "emily", "donna", "michelle",
           "carol", "amanda", "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia", "kathleen"]
NICK = {"william": "bill", "robert": "bob", "richard": "dick", "james": "jim", "michael": "mike", "joseph": "joe",
        "thomas": "tom", "charles": "chuck", "christopher": "chris", "daniel": "dan", "matthew": "matt",
        "anthony": "tony", "elizabeth": "liz", "patricia": "pat", "jennifer": "jen", "margaret": "peggy",
        "kimberly": "kim", "deborah": "deb", "stephanie": "steph", "rebecca": "becky", "kathleen": "kathy",
        "steven": "steve", "andrew": "andy", "kenneth": "ken", "edward": "ed", "timothy": "tim", "donald": "don"}
LAST = ["smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis", "rodriguez", "martinez",
        "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas", "taylor", "moore", "jackson", "martin",
        "lee", "perez", "thompson", "white", "harris", "sanchez", "clark", "ramirez", "lewis", "robinson",
        "walker", "young", "allen", "king", "wright", "scott", "torres", "nguyen", "hill", "flores", "green",
        "adams", "nelson", "baker", "hall", "rivera", "campbell", "mitchell", "carter", "roberts", "okafor",
        "kowalski", "lindqvist", "haddad", "fitzgerald", "obrien", "delacroix", "van der berg", "mcallister"]
STREETS = ["main st", "oak ave", "maple dr", "cedar ln", "pine st", "elm st", "washington ave", "lake rd",
           "hill st", "park ave", "sunset blvd", "river rd", "church st", "spring st", "highland ave",
           "franklin st", "jefferson ave", "madison ave", "lincoln way", "jackson st", "walnut st", "chestnut st"]
CITIES = [("seattle", "wa", "981"), ("portland", "or", "972"), ("austin", "tx", "787"), ("denver", "co", "802"),
          ("boston", "ma", "021"), ("chicago", "il", "606"), ("atlanta", "ga", "303"), ("phoenix", "az", "850")]
# the reserved .example domain (RFC 2606): no generated address can be someone's real inbox
DOMAINS = ["gmail.example", "yahoo.example", "outlook.example", "hotmail.example", "icloud.example", "comcast.example"]


def unlikely_inbox(*parts):
    """Six characters derived from the person, appended to the user name of a real Gmail address
    (the Gmail scenario must be on gmail.com, where dots and +tags are ignored): a real inbox of
    that name is vanishingly unlikely. A hash, not the random generator, so no other record moves."""
    import hashlib
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:6]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.used_phones = set()

    def phone(self):
        while True:
            p = f"{self.r.choice(['206', '503', '512', '303', '617', '312', '404', '602'])}" \
                f"{self.r.randint(200, 999)}{self.r.randint(1000, 9999)}"
            if p not in self.used_phones:
                self.used_phones.add(p)
                return p

    def dob(self, lo=1940, hi=2004):
        return f"{self.r.randint(lo, hi)}-{self.r.randint(1, 12):02d}-{self.r.randint(1, 28):02d}"

    def address(self):
        city, st, z = self.r.choice(CITIES)
        return {"address": f"{self.r.randint(1, 9999)} {self.r.choice(STREETS)}", "city": city, "state": st,
                "zip": f"{z}{self.r.randint(10, 99)}"}

    def email(self, first, last):
        style = self.r.randint(0, 3)
        user = [f"{first}.{last}", f"{first[0]}{last}", f"{first}{last}{self.r.randint(1, 99)}", f"{last}.{first}"][style]
        return f"{user.replace(' ', '')}@{self.r.choice(DOMAINS)}"

    def person(self, sex=None, last=None, dob=None, addr=None):
        sex = sex or self.r.choice("mf")
        first = self.r.choice(FIRST_M if sex == "m" else FIRST_F)
        last = last or self.r.choice(LAST)
        return {"first": first, "last": last, "dob": dob or self.dob(), "phone": self.phone(),
                "email": self.email(first, last), **(addr or self.address())}

    # formats differ by source, like real systems
    def fmt_phone(self, p, source):
        return f"({p[:3]}) {p[3:6]}-{p[6:]}" if source == "billing" else p

    def fmt_dob(self, d, source):
        y, m, dd = d.split("-")
        return f"{m}/{dd}/{y}" if source == "billing" else d

    def bad_dob(self, d):
        """Day and month swapped, a year typo, or one digit wrong - the usual keying errors."""
        y, m, dd = d.split("-")
        kind = self.r.randint(0, 2)
        if kind == 0 and int(dd) <= 12:
            return f"{y}-{dd}-{m}"
        if kind == 1:
            return f"{int(y) + self.r.choice([-1, 1, -10, 10])}-{m}-{dd}"
        return f"{y}-{m}-{(int(dd) % 28) + 1:02d}"

    def noisy(self, s, rate=0.15):
        if not s or self.r.random() > rate or len(s) < 4:
            return s
        i = self.r.randrange(1, len(s) - 1)
        op = self.r.randint(0, 2)
        if op == 0:
            return s[:i] + s[i + 1:]
        if op == 1:
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        return s[:i] + self.r.choice("aeiourstln") + s[i + 1:]


def record(g, rid, source, p, pid, hh, first=None, last=None, drop=(), **over):
    """One record of person p in a source, with that source's formats; `drop` blanks fields."""
    rec = {"id": rid, "first_name": first if first is not None else p["first"],
           "last_name": last if last is not None else p["last"],
           "dob": g.fmt_dob(p["dob"], source), "address": f"{p['address']}, {p['city']} {p['state']} {p['zip']}",
           "email": p["email"], "phone": g.fmt_phone(p["phone"], source), "person_id": pid, "household_id": hh}
    rec |= over
    for f in drop:
        rec[f] = ""
    rec["_source"] = source
    return rec


def background(g, n):
    recs, pid, hh = [], 0, 0
    while pid < n:
        addr = g.address()
        a = g.person(addr=addr)
        members = [a]
        if g.r.random() < 0.5:                                  # a couple
            b = g.person(sex="f" if a["first"] in FIRST_M else "m", last=a["last"] if g.r.random() < 0.7 else None,
                         dob=g.dob(), addr=addr)
            if g.r.random() < 0.5:
                b["phone"] = a["phone"]                         # a shared landline
            if g.r.random() < 0.3:
                fam = f"the{a['last'].replace(' ', '')}s@{g.r.choice(DOMAINS)}"
                a["email"] = b["email"] = fam                   # a family email
            members.append(b)
            if g.r.random() < 0.3:                              # a grown child at home
                members.append(g.person(last=a["last"], dob=g.dob(1990, 2004), addr=addr))
        for p in members:
            k = g.r.choice([2, 2, 3])
            for j in range(k):
                source = ["crm", "billing", "crm"][j]
                first = p["first"]
                if g.r.random() < 0.15 and first in NICK:
                    first = NICK[first]
                drop = [f for f in ("dob", "email", "phone", "address") if g.r.random() < 0.2]
                over = {}
                if g.r.random() < 0.06:                         # a birth date keyed wrongly
                    over["dob"] = g.fmt_dob(g.bad_dob(p["dob"]), source)
                recs.append(record(g, f"bg{pid}_{j}", source, p, pid, hh, first=g.noisy(first),
                                   last=g.noisy(p["last"]), drop=drop, **over))
            pid += 1
        hh += 1
    return recs, pid, hh


# ------------------------------------------------------------------ the scenarios
# each returns (records, expected pairs [(record id, record id, "same" | "different")]) for one repeat
def scenarios():
    S = {}

    def same_pair(name, desc, change):
        def make(g, k, pid, hh):
            p = g.person()
            a = record(g, f"{name}{k}a", "crm", p, pid, hh)
            b = record(g, f"{name}{k}b", "billing", p, pid, hh)
            change(g, p, a, b)
            return [a, b], [(a["id"], b["id"], "same")]
        S[name] = (desc, "same", make)

    same_pair("typo_first", "A typo in the first name (Jonathon / Jonathan)",
              lambda g, p, a, b: b.update(first_name=g.noisy(p["first"], 1.0)))
    same_pair("nickname", "A nickname (Bill / William)",
              lambda g, p, a, b: b.update(first_name=NICK.get(p["first"], p["first"][:3])))
    same_pair("name_swapped", "First and last name swapped",
              lambda g, p, a, b: b.update(first_name=p["last"], last_name=p["first"]))
    same_pair("married_name", "A new surname after marriage; same date of birth and email",
              lambda g, p, a, b: b.update(last_name=g.r.choice(LAST), address=f"{g.address()['address']}, {p['city']}"))
    same_pair("moved", "Moved house: a new address and phone; same name, date of birth and email",
              lambda g, p, a, b: b.update(address=f"{g.address()['address']}, {g.r.choice(CITIES)[0]}",
                                          phone=g.fmt_phone(g.phone(), "billing")))
    same_pair("gmail_variants", "Gmail with dots and a +tag (j.smith+shop@gmail.com / jsmith@gmail.com)",
              lambda g, p, a, b: (a.update(email=f"{p['first']}.{p['last'].replace(' ', '')}.{unlikely_inbox(a['id'])}+shop@gmail.com"),
                                  b.update(email=f"{p['first']}{p['last'].replace(' ', '')}{unlikely_inbox(a['id'])}@gmail.com"),
                                  b.update(phone="", dob="")))
    same_pair("phone_formats", "One phone in different formats (+1 206 555 1234 / (206) 555-1234); no email",
              lambda g, p, a, b: (a.update(phone=f"+1 {p['phone'][:3]} {p['phone'][3:6]} {p['phone'][6:]}", email=""),
                                  b.update(email="")))
    same_pair("dob_day_month_swapped", "Day and month of birth swapped",
              lambda g, p, a, b: b.update(dob="/".join([p["dob"][8:10], p["dob"][5:7], p["dob"][:4]])))
    same_pair("no_surname_email", "One record has only a first name and email (a web sign-up)",
              lambda g, p, a, b: b.update(last_name="", dob="", address="", phone=""))
    same_pair("initial_only", "First initial only, with the same email",
              lambda g, p, a, b: b.update(first_name=p["first"][0]))
    same_pair("hyphenated", "A double-barrelled surname on one record (Garcia-Lopez / Garcia)",
              lambda g, p, a, b: a.update(last_name=f"{p['last']}-{g.r.choice(LAST)}"))
    same_pair("punctuation_case", "Case, spaces and punctuation (O'BRIEN  JOHN / obrien john)",
              lambda g, p, a, b: (a.update(first_name=p["first"].upper() + " ", last_name="O'" + p["last"].upper()),
                                  b.update(last_name="o" + p["last"])))
    same_pair("accents", "Accents on one record (José / Jose, Zoë / Zoe)",
              lambda g, p, a, b: a.update(first_name=p["first"].replace("e", "é", 1).replace("a", "á", 1)))
    same_pair("apartment_forms", "Apartment written differently (12 Main St Apt 4 / 12 Main St #4)",
              lambda g, p, a, b: (a.update(address=f"{p['address']} apt 4, {p['city']}"),
                                  b.update(address=f"{p['address']} #4, {p['city']}", email="")))
    same_pair("sparse_pair", "Both records sparse: first name, surname and phone only",
              lambda g, p, a, b: (a.update(email="", dob="", address=""), b.update(email="", dob="", address="")))

    def different(name, desc, make_people):
        def make(g, k, pid, hh):
            recs, exp = make_people(g, k, pid, hh)
            return recs, exp
        S[name] = (desc, "different", make)

    def twins(g, k, pid, hh):
        a = g.person(sex="m")
        b = g.person(sex="m", last=a["last"], dob=a["dob"], addr={x: a[x] for x in ("address", "city", "state", "zip")})
        b["phone"] = a["phone"]
        while b["first"] == a["first"]:                         # twins never share a first name
            b["first"] = g.r.choice(FIRST_M)
        ra, rb = record(g, f"twins{k}a", "crm", a, pid, hh), record(g, f"twins{k}b", "billing", b, pid + 1, hh)
        return [ra, rb], [(ra["id"], rb["id"], "different")]
    different("twins", "Twins: same surname, birthday, address and phone; different first names", twins)

    def jr_sr(g, k, pid, hh):
        a = g.person(sex="m", dob=g.dob(1950, 1965))
        b = dict(a, dob=g.dob(1980, 1995), phone=g.phone(), email=g.email(a["first"], a["last"]))
        ra = record(g, f"jrsr{k}a", "crm", a, pid, hh)
        rb = record(g, f"jrsr{k}b", "billing", b, pid + 1, hh)
        return [ra, rb], [(ra["id"], rb["id"], "different")]
    different("father_son_same_name", "Father and son with the same name and address (Sr / Jr); different birth years", jr_sr)

    def spouses(g, k, pid, hh):
        a = g.person(sex="m")
        b = g.person(sex="f", last=a["last"], addr={x: a[x] for x in ("address", "city", "state", "zip")})
        fam = f"the{a['last'].replace(' ', '')}family@gmail.example"
        ra = record(g, f"spouses{k}a", "crm", a, pid, hh, email=fam)
        rb = record(g, f"spouses{k}b", "billing", b, pid + 1, hh, email=fam, phone=g.fmt_phone(a["phone"], "billing"))
        return [ra, rb], [(ra["id"], rb["id"], "different")]
    different("spouses_shared_contacts", "A couple sharing surname, address, family email and phone", spouses)

    def common_name(g, k, pid, hh):
        a = g.person(sex="m", last="smith")
        a["first"] = "john"
        b = g.person(sex="m", last="smith")
        b["first"] = "john"
        ra = record(g, f"johnsmith{k}a", "crm", a, pid, hh, email="")
        rb = record(g, f"johnsmith{k}b", "billing", b, pid + 1, hh + 1, email="")
        return [ra, rb], [(ra["id"], rb["id"], "different")]
    different("common_name_strangers", "Two John Smiths: same name, different birthdays, addresses and phones", common_name)

    def recycled(g, k, pid, hh):
        a, b = g.person(), g.person()
        b["phone"] = a["phone"]
        ra = record(g, f"recycled{k}a", "crm", a, pid, hh, email="")
        rb = record(g, f"recycled{k}b", "billing", b, pid + 1, hh + 1, email="")
        return [ra, rb], [(ra["id"], rb["id"], "different")]
    different("reissued_phone", "Strangers with the same mobile (a reissued number)", recycled)

    def junk(g, k, pid, hh):
        recs = []
        for j in range(4):
            p = g.person()
            recs.append(record(g, f"junk{k}_{j}", "crm" if j % 2 else "billing", p, pid + j, hh + j,
                               email="test@test.com", phone="(555) 555-5555" if j % 2 else "0000000000"))
        return recs, [(recs[i]["id"], recs[j]["id"], "different") for i in range(4) for j in range(i + 1, 4)]
    different("junk_contacts", "Strangers who all typed test@test.com and 555-555-5555 / 0000000000", junk)

    def roommates(g, k, pid, hh):
        a = g.person()
        b = g.person(addr={x: a[x] for x in ("address", "city", "state", "zip")})
        ra = record(g, f"roommates{k}a", "crm", a, pid, hh)
        rb = record(g, f"roommates{k}b", "billing", b, pid + 1, hh)
        return [ra, rb], [(ra["id"], rb["id"], "different")]
    different("roommates", "Unrelated people at one address", roommates)

    def chain(g, k, pid, hh):
        # two different people, and a third record that half-matches each: it must not join them
        a, b = g.person(sex="m"), g.person(sex="m")
        b["last"] = a["last"]
        ra = record(g, f"chain{k}a", "crm", a, pid, hh)
        rb = record(g, f"chain{k}b", "crm", b, pid + 1, hh + 1)
        mid = record(g, f"chain{k}m", "billing", a, pid, hh, email=b["email"], phone=g.fmt_phone(b["phone"], "billing"),
                     dob="", address="")
        return [ra, rb, mid], [(ra["id"], rb["id"], "different")]
    different("no_chaining", "A record with one person's name and another's email and phone must not join the two", chain)

    def siblings(g, k, pid, hh):
        a = g.person(sex="f", dob=g.dob(1985, 1990))
        b = g.person(sex="f", last=a["last"], dob=g.dob(1991, 1996),
                     addr={x: a[x] for x in ("address", "city", "state", "zip")})
        b["first"] = a["first"][0] + b["first"][1:] if b["first"][0] != a["first"][0] else b["first"]
        ra = record(g, f"siblings{k}a", "crm", a, pid, hh, first=a["first"][0])
        rb = record(g, f"siblings{k}b", "billing", b, pid + 1, hh, first=b["first"][0], email="", phone="")
        return [ra, rb], [(ra["id"], rb["id"], "different")]
    different("siblings_same_initial", "Sisters at one address, both recorded by first initial only", siblings)

    def fake_mobile(g, k, pid, hh):
        # nameless sign-ups, each with a real email, all typing the same made-up mobile: people reuse
        # the same fakes ("867-5309"), which then collect many different emails
        phone = f"({g.r.choice(['312', '212', '415', '206', '713', '602', '305', '404', '617', '503'])}) 867-5309"
        recs = [record(g, f"fakemobile{k}_{j}", "crm" if j % 2 else "billing", g.person(), pid + j, hh + j,
                       phone=phone, drop=("first_name", "last_name", "dob", "address")) for j in range(10)]
        return recs, [(recs[i]["id"], recs[j]["id"], "different") for i in range(10) for j in range(i + 1, 10)]
    different("fake_mobile_many_emails", "Nameless sign-ups, each with their own email, all typing one made-up mobile",
              fake_mobile)

    golds = []

    def gold(g, p, a, b):
        # a bought "gold" number looks made up but is hers alone: the sign-up has nothing else to link by
        if not golds:
            golds.extend(f"({area}) {d * 3}-{d * 4}" for area in ("212", "415", "206", "713", "602") for d in "2346789")
            g.r.shuffle(golds)
        phone = golds.pop()
        a.update(phone=phone)
        b.update(last_name="", dob="", address="", email="", phone=phone)
    same_pair("gold_number", "A bought gold mobile ((212) 222-2222): a sign-up with only a first name and it",
              gold)
    return S


def build(seed=0):
    g = Gen(seed)
    recs, pid, hh = background(g, BACKGROUND)
    expected = []
    for name, (desc, kind, make) in scenarios().items():
        for k in range(REPEATS):
            rs, exp = make(g, k, pid, hh)
            pid, hh = pid + 5, hh + 5
            for r in rs:
                r["scenario"] = name
            recs += rs
            expected += [{"scenario": name, "repeat": k, "a": a, "b": b, "expected": e} for a, b, e in exp]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs).fillna("")
    for source in ("crm", "billing"):
        d = df[df._source == source].drop(columns=["_source"])
        d.drop(columns=["person_id", "household_id", "scenario"]).to_csv(OUT_DIR / f"{source}.csv", index=False)
    key = df._source + ":" + df.id
    pd.DataFrame({"record": key, "person_id": df.person_id, "household_id": df.household_id,
                  "scenario": df.get("scenario", "").replace("", "background")}).to_csv(OUT_DIR / "truth.csv", index=False)
    src = dict(zip(df.id, df._source))
    exp = pd.DataFrame(expected)
    exp["a"], exp["b"] = exp.a.map(lambda x: f"{src[x]}:{x}"), exp.b.map(lambda x: f"{src[x]}:{x}")
    exp.to_csv(OUT_DIR / "expected.csv", index=False)
    cols = {"record_id": "id", "first_name": "first_name", "last_name": "last_name", "date_of_birth": "dob",
            "address": "address", "email": "email", "phone": "phone"}
    cfg = {"sources": [{"name": s, "path": f"{s}.csv", "columns": cols} for s in ("crm", "billing")],
           "country": "US", "source_priority": ["crm", "billing"]}
    (OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=1))
    (OUT_DIR / "scenarios.json").write_text(json.dumps(
        {n: {"description": d, "expected": k} for n, (d, k, _) in scenarios().items()}, indent=1))
    return len(df), len(exp)


def score(variant="default", seeds=SEEDS, labels_n=1500):
    from custmatch import labels, pipeline
    truth = pd.read_csv(OUT_DIR / "truth.csv", dtype=str)
    pid = dict(zip(truth.record, truth.person_id))
    scen = dict(zip(truth.record, truth.scenario))
    exp = pd.read_csv(OUT_DIR / "expected.csv", dtype=str)
    meta = json.loads((OUT_DIR / "scenarios.json").read_text())
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = VARIANTS[variant]
        cfg = json.loads((OUT_DIR / "config.json").read_text()) | {k: v for k, v in spec.items() if not k.startswith("_")}
        for src in cfg["sources"]:
            src["path"] = str(OUT_DIR / src["path"])
        cp = tmp / "config.json"
        cp.write_text(json.dumps(cfg))
        # labels are drawn from the background records only: a copy of the file without the
        # scenarios, so neither random nor active sampling can pick a scenario pair
        bg_cfg = json.loads(json.dumps(cfg))
        for src in bg_cfg["sources"]:
            d = pd.read_csv(src["path"], dtype=str, keep_default_na=False)
            d = d[(src["name"] + ":" + d.id).map(scen) == "background"]
            src["path"] = str(tmp / f"bg_{src['name']}.csv")
            d.to_csv(src["path"], index=False)
        bp = tmp / "background.json"
        bp.write_text(json.dumps(bg_cfg))
        for seed in seeds:
            lp = tmp / f"labels{seed}.csv"
            labels.sample(bp, labels_n * 2, lp, seed=seed)
            lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
            background_only = lambda d: d[(d.a.map(scen) == "background") & (d.b.map(scen) == "background")]  # noqa: E731
            lab = background_only(lab).head(500 if spec.get("_active") else labels_n)
            lab["label"] = [int(pid[a] == pid[b]) for a, b in zip(lab.a, lab.b)]
            lab.to_csv(lp, index=False)
            while spec.get("_active") and len(lab) < labels_n:
                nxt = tmp / "next.csv"
                labels.sample_active(bp, lp, 100, nxt, seed=seed)
                more = background_only(pd.read_csv(nxt, dtype=str, keep_default_na=False))
                more["label"] = [int(pid[a] == pid[b]) for a, b in zip(more.a, more.b)]
                if not len(more):
                    break                                       # nothing left to ask about
                lab = pd.concat([lab, more], ignore_index=True)
                lab.to_csv(lp, index=False)
            out = tmp / f"run{seed}"
            s = pipeline.run(cp, lp, out, seed=seed, log=lambda *_: None)
            c = pd.read_csv(out / "clusters.csv", dtype=str)
            cust = dict(zip(c.source + ":" + c.record_id, c.customer_id))
            got = np.where([cust[a] == cust[b] for a, b in zip(exp.a, exp.b)], "same", "different")
            ok = pd.Series(got == exp.expected.to_numpy())
            per = ok.groupby([exp.scenario, exp.repeat]).all().groupby(level=0).mean()
            bg = truth[truth.scenario == "background"].record
            from custmatch.core import partition_scores
            ids = np.array([pid[k] for k in bg])
            _, n = np.unique(ids, return_counts=True)
            f1 = partition_scores(np.array([cust[k] for k in bg]), ids, float(np.sum(n * (n - 1) / 2)))["f1"]
            runs.append({"seed": seed, "labels": int(len(lab)), "background_f1": f1, "threshold": s["threshold"],
                         "scenarios": {k: round(100 * float(v), 1) for k, v in per.items()}})
            print(variant, seed, f1, {k: v for k, v in runs[-1]["scenarios"].items() if v < 100}, flush=True)
    summary = {name: {"description": m["description"], "expected": m["expected"],
                      "passed_pct": round(float(np.mean([r["scenarios"][name] for r in runs])), 1)}
               for name, m in meta.items()}
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res |= {"repeats": REPEATS, "background_people": BACKGROUND}
    res[variant] = {"runs": runs, "scenarios": summary,
                    "background_f1": round(float(np.mean([r["background_f1"] for r in runs])), 2)}
    RESULTS.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        print("records, expected pairs:", build())
    else:
        for v in sys.argv[2:] or list(VARIANTS):
            score(v)
