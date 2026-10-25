"""A synthetic personal address book, shaped like a real one, with the person behind each record.

Built from people, not rows: about 940 people, each with one or more email "identities", one
record per email address, as a mail client exports contacts. The shape copies aggregate counts
from a real 808-record address book (20 years of one person's mail; nothing from it but the
counts below), scaled to about 1,000 records:

  records per person   about 885 alone, 45 with 2, 7 with 3, 1-2 with 6-8 (about 95 true pairs)
  fields               first name always; surname 97-98%; phone 30-35%; no address or birth date
  domains              15-20% free mail; most at work domains, one per organisation and one
                       address template per organisation (first.last, flast, ...); a school
                       with about 70 records and two other employers of 40-50
How one person's records differ: an employer change (work domain A, then B - the commonest),
a personal address beside a work one, Apple's me.com / mac.com / icloud.com and Gmail dot
variants, a nickname (Kate / Katherine), a title (Dr), a maiden then married surname, a
double-barrelled surname losing a part, a phone on one record only or written +61, a first name
with no surname. Not typos: the real file has almost none.

The traps - true non-matches that look like matches - each tagged in truth.csv's `trap`:
  relay        5-10 service relay addresses (drive-shares-dm-noreply@google.com, ...) filed
               under a real person's name, sometimes with their phone: every false link of the
               best run on the real file
  switchboard  one organisation number on about 30 colleagues at the school
  household    15-20 shared home phones: a spouse or child with the same surname, another first name
  same_name    two people with one full name at different organisations, nothing else shared
  brand        shop or brand names that look like people ("Mind Games" / "Mindgames")

Names are drawn as research/scripts/au_customer_data.py draws them (Census surnames and SSA
given names at real frequencies, and its other name groups), so common surnames make realistic
false candidates. Work domains are invented under the reserved .example top-level domain; free
mail uses the real providers with invented user names. Mobiles are random 04 numbers, as in
au_customer_data.py: ACMA's fictional ranges (0491 570 ...) are junk on sight to custmatch, so a
file of them would carry no phone evidence at all.

    python research/scripts/address_book_data.py [--seed 1]

Writes data/address_book/s{seed}/: records.csv (no person id) and records_clean.csv (the same
people with the relay records removed and the switchboard number blanked - traps 1 and 2, the
input-hygiene ones), truth.csv (record_id, person_id, trap) and config.json, and prints the
generated profile against the targets.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import au_customer_data as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/address_book"
SIZES = [(1, 850), (2, 45), (3, 7)]   # people with that many records, plus 1-2 of 6-8 (build); the traps add
                                       # about 35 more people alone, for about 885
FREE = [("gmail.com", 50), ("icloud.com", 15), ("outlook.com", 10), ("hotmail.com", 10),
        ("bigpond.com", 10), ("yahoo.com.au", 5)]
TEMPLATES = ["first.last", "flast", "firstl", "first", "first_last", "lastf"]
BIG = [("westfield-college", 70), ("northside-health", 50), ("harbour-uni", 45)]   # records at each
ORG_WORDS = ("acme apex arbor atlas beacon bluegum cedar coastal crest delta ember fern harbour "
             "horizon iron jarrah kestrel lakeside lumen maple meridian nova oak orbit pacific "
             "peak pine quarry redgum ridge river sandstone silver solar summit tidal vista wattle "
             "willow zenith").split()
ORG_KINDS = "legal consulting health design build media group partners labs studio advisory".split()
RELAYS = ["drive-shares-dm-noreply@google.com", "calendar-notification@google.com",
          "comments-noreply@docs.google.com", "meet-recordings-noreply@google.com",
          "no-reply@dropboxmail.com", "notifications@slack-mail.example", "no_reply@sharefile.example",
          "noreply-calendar@zoom.example", "bounces+42@mailer.example", "notification@teams.example"]
BRANDS = [("mind games", "mindgames"), ("blue door", "bluedoor"), ("the green grocer", "green grocer"),
          ("kidz zone", "kids zone")]
TITLES = ["dr", "dr", "prof"]


class Book:
    def __init__(self, seed):
        self.g = A.Gen(seed)
        self.r = self.g.r
        self.rows = []
        self.used = set()
        self.names = set()
        small = [f"{a}-{b}" for a in ORG_WORDS for b in ORG_KINDS]
        self.orgs = list(self.r.permutation(small)[:160])
        self.template = {}

    def org_template(self, org):
        if org not in self.template:
            self.template[org] = self.g.pick(TEMPLATES)
        return self.template[org]

    def name(self):
        """A new person's name, redrawn while its full name is taken: the smaller name groups'
        lists are short, and chance repeats made 29 shared full names on a thousand records.
        Same full names come only from the same_name trap and one person's records."""
        groups = list(A.GROUPS)
        for _ in range(20):
            group = self.g.pick(groups, [A.GROUPS[k]["weight"] for k in groups])
            sex, age = self.g.pick("mf"), int(self.r.integers(22, 76))
            n = dict(given=A.given_for(self.g, group, sex, age), surname=A.surname_for(self.g, group), sex=sex)
            if (n["given"], n["surname"]) not in self.names:
                break
        self.names.add((n["given"], n["surname"]))
        return n

    def local(self, template, first, last, n=0):
        first, last = A.slug(first), A.slug(last) or A.slug(first)
        out = {"first.last": f"{first}.{last}", "flast": f"{first[:1]}{last}", "firstl": f"{first}{last[:1]}",
               "first": first, "first_last": f"{first}_{last}", "lastf": f"{last}{first[:1]}"}[template]
        return out + (str(n) if n else "")

    def address(self, template, first, last, domain):
        n = 0
        while True:
            a = f"{self.local(template, first, last, n)}@{domain}"
            if a not in self.used:
                self.used.add(a)
                return a
            n = n + 1 if n else int(self.r.integers(2, 99))

    def work(self, first, last, org=None):
        # small organisations a few to a dozen records each: the big three (BIG) are the hubs
        org = org or self.g.pick(self.orgs, [1 / (i + 1) ** 0.35 for i in range(len(self.orgs))])
        return self.address(self.org_template(org), first, last, f"{org}.example"), org

    def free(self, first, last, domain=None):
        domain = domain or self.g.pick([d for d, _ in FREE], [w for _, w in FREE])
        return self.address(self.g.pick(["first.last", "firstl", "flast", "first_last"]), first, last, domain)

    def phone(self, digits, plus=False):
        return f"+61 {digits[1:4]} {digits[4:7]} {digits[7:]}" if plus else f"{digits[:4]} {digits[4:7]} {digits[7:]}"

    def add(self, pid, first, last, email, phone="", trap=""):
        self.rows.append(dict(record_id=f"r{len(self.rows)}", first=first, last=last, email=email, phone=phone,
                              person_id=pid, trap=trap))


def build(seed):
    b = Book(seed)
    g, r = b.g, b.r
    # one person with 8 records, as in the real file, and sometimes another with 6
    sizes = [n for n, k in SIZES for _ in range(k)] + [8] + ([6] if g.p(0.3) else [])
    sizes = list(r.permutation(sizes))
    big_slots = [org for org, n in BIG for _ in range(n)]       # work records the big employers hold
    people = []
    for pid, size in enumerate(sizes):
        p = b.name() | {"pid": f"p{pid}", "size": int(size)}
        while size >= 6 and (p["sex"] != "f" or p["given"] not in A.NICKNAMES):
            # like the real file's person with eight records: three names - a nickname, the
            # formal name and a maiden name
            p = b.name() | {"pid": f"p{pid}", "size": int(size), "big": True}
        p["phone"] = g.mobile() if g.p(0.27 if size == 1 else 0.4) else ""
        people.append(p)
    r.shuffle(big_slots)
    singles = [p for p in people if p["size"] == 1]
    for p, org in zip(singles, big_slots):                       # the big employers' staff, alone in the book
        p["org"] = org
    for p in people:
        first, last = p["given"], p["surname"]
        no_last = g.p(0.025)
        if p["size"] == 1:
            if p.get("org"):
                email, _ = b.work(first, last, p["org"])
            elif g.p(0.17):
                email = b.free(first, last)
            else:
                email, p["org"] = b.work(first, last)
            b.add(p["pid"], first, "" if no_last else last, email, b.phone(p["phone"]) if p["phone"] else "")
            continue
        # one person, several identities
        nick = A.NICKNAMES.get(first)
        title = g.pick(TITLES) if g.p(0.1) else ""
        maiden = A.surname_for(g, "anglo") if p["sex"] == "f" and (p.get("big") or g.p(0.25)) else ""
        barrel = f"{last}-{A.surname_for(g, 'anglo')}" if g.p(0.05) else ""
        org = None
        icloud = None
        for k in range(p["size"]):
            f, s = first, (barrel or last)
            if nick and g.p(0.5 if p.get("big") else 0.3):
                f = g.pick(nick)
            if barrel and k and g.p(0.5):
                s = last                                        # the double-barrel loses a part
            if maiden and k < p["size"] // 2:
                s = maiden                                      # older identities: the maiden name
            kind = "work" if k == 0 else g.pick(["employer", "personal", "apple", "gmail_dots"], [0.45, 0.35, 0.12, 0.08])
            if kind in ("work", "employer"):
                email, org = b.work(f, s, None if kind == "employer" or org is None else org)
            elif kind == "apple" and icloud:
                user = icloud.split("@")[0]
                dom = g.pick([d for d in ("me.com", "mac.com", "icloud.com") if not icloud.endswith(d)])
                email = f"{user}@{dom}"
            elif kind == "gmail_dots" and any(x["email"].endswith("@gmail.com") for x in b.rows if x["person_id"] == p["pid"]):
                prev = next(x["email"] for x in b.rows if x["person_id"] == p["pid"] and x["email"].endswith("@gmail.com"))
                user = prev.split("@")[0]
                email = (user.replace(".", "") if "." in user else f"{user[:1]}.{user[1:]}") + "@gmail.com"
            else:
                email = b.free(f, s, "icloud.com" if kind == "apple" else None)
                icloud = email if email.endswith("@icloud.com") else icloud
            b.used.add(email)
            shown_first = f"{title} {f}" if title and g.p(0.5) else f
            ph = b.phone(p["phone"], plus=g.p(0.3)) if p["phone"] and g.p(0.6) else ""
            b.add(p["pid"], shown_first, "" if no_last and k == 0 else s, email, ph)
    n = len(people)

    def new_pid():
        nonlocal n
        n += 1
        return f"p{n - 1}"

    rows = b.rows
    # switchboard: one number on about 30 people at the school
    board = g.mobile()
    staff = [x for x in rows if x["email"].endswith("@westfield-college.example")]
    for x in list(r.permutation(len(staff)))[:int(r.integers(25, 36))]:
        staff[x]["phone"], staff[x]["trap"] = f"(02) {board[2:6]} {board[6:]}", "switchboard"
    # household phones: a spouse or child, same surname, another first name, one phone
    alone = [x for x in rows if x["trap"] == "" and x["last"] and sum(y["person_id"] == x["person_id"] for y in rows) == 1]
    for x in [alone[i] for i in r.permutation(len(alone))[:int(r.integers(15, 21))]]:
        other = b.name()
        if other["given"] == x["first"]:
            continue
        home = g.mobile()
        x["phone"], x["trap"] = b.phone(home), "household"
        b.add(new_pid(), other["given"], x["last"], b.free(other["given"], x["last"]), b.phone(home, plus=g.p(0.3)), "household")
    # same full name, different people, different organisations, nothing else shared
    for x in [alone[i] for i in r.permutation(len(alone))[:int(r.integers(4, 8))]]:
        if x["trap"]:
            continue
        x["trap"] = "same_name"
        b.add(new_pid(), x["first"], x["last"], b.work(x["first"], x["last"])[0], "", "same_name")
    # brands that look like people: two spellings, two different accounts
    for a, c in [BRANDS[i] for i in r.permutation(len(BRANDS))[:int(r.integers(2, 4))]]:
        for name, box in ((a, "orders"), (c, "hello")):
            first, _, last = name.partition(" ")
            b.add(new_pid(), first, last, f"{box}@{A.slug(name)}{int(r.integers(1, 9))}.example", "", "brand")
    # relay addresses filed under a real person's name, the people with most records first
    owners = sorted({x["person_id"] for x in rows if not x["trap"]},
                    key=lambda q: -sum(y["person_id"] == q for y in rows))
    k = int(r.integers(5, 11))
    chosen = owners[:max(2, k // 2)] + [owners[i] for i in r.choice(np.arange(len(owners) // 2, len(owners)), k - max(2, k // 2), replace=False)]
    for relay, q in zip(r.permutation(RELAYS), chosen):
        src = next(x for x in rows if x["person_id"] == q)
        b.add(new_pid(), src["first"], src["last"], relay, src["phone"] if g.p(0.4) else "", "relay")
    df = pd.DataFrame(b.rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def profile(df):
    sizes = df.groupby("person_id").size()
    counts = sizes.value_counts().sort_index()
    pairs = int((sizes * (sizes - 1) // 2).sum())
    free = df.email.str.split("@").str[1].isin([d for d, _ in FREE] + ["me.com", "mac.com"])
    dom = df.email.str.split("@").str[1].value_counts()
    ph = df.phone.str.replace(r"\D", "", regex=True).str[-9:]
    shared = ph[ph != ""].value_counts()
    full = (df["first"] + " " + df["last"]).where(df["last"] != "")
    same = full.dropna().value_counts()
    print(f"  records {len(df):,}; people {len(sizes):,}; records per person {dict(counts)}; true pairs {pairs}")
    print(f"  surname {100 * (df['last'] != '').mean():.1f}%; phone {100 * (df.phone != '').mean():.1f}%; "
          f"free mail {100 * free.mean():.1f}%; largest domains {dict(dom.head(3))}")
    print(f"  phones on 2+ records {int((shared > 1).sum())}, largest on {int(shared.max())}; "
          f"full names on 2+ records {int((same > 1).sum())}; relay {int((df.trap == 'relay').sum())}; "
          f"traps {dict(df.trap[df.trap != ''].value_counts())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    df = build(a.seed)
    out = OUT / f"s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["record_id", "first", "last", "email", "phone"]
    df[cols].to_csv(out / "records.csv", index=False)
    clean = df[df.trap != "relay"].copy()
    clean.loc[clean.trap == "switchboard", "phone"] = ""
    clean[cols].to_csv(out / "records_clean.csv", index=False)
    df[["record_id", "person_id", "trap"]].to_csv(out / "truth.csv", index=False)
    for name in ("records", "records_clean"):
        (out / f"{name}.json").write_text(json.dumps({"country": "AU", "sources": [
            {"name": "book", "path": f"{name}.csv", "columns": {"record_id": "record_id", "first_name": "first",
                                                                "last_name": "last", "email": "email", "phone": "phone"}}]}, indent=1))
    print(f"seed {a.seed} -> {out}")
    profile(df)


if __name__ == "__main__":
    main()
