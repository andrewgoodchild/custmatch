"""Build a sparse, Australian customer-matching dataset with known identities and households.

Real customer data is rarely published and none we know of has patchy names with emails and
mobiles and ground truth, so this is generated. The point is the SHAPE of Australian retail
data, not any real person:

  channels   web sign-ups   first name + email; surname, mobile, DOB, postcode sometimes
             loyalty app    first name + mobile (SMS-verified); surname, email, DOB sometimes
             store / CRM    full name + address; mobile, email, DOB sometimes
             No SSN or any national id anywhere.
  people     households sized like the 2021 Census (1 to 6+), couples, adult children at home,
             share houses of unrelated adults; names from Anglo-Celtic, Chinese, Indian,
             Italian, Vietnamese, Greek, Filipino and Lebanese / Arabic groups in roughly
             Census ancestry proportions (approximate - see GROUPS)
  errors     nicknames (Mick, Bec, Mel), anglicised names (Wei -> Kevin), family-name-first
             order, Vietnamese Van / Thi middle names, one-name people, spelling variants
             (Mohammed / Mohamed / Muhammad), married-name changes, typos; mobiles written
             0412 345 678 / +61 412 345 678 / 412345678; changed and recycled mobiles; shared
             family emails and mobiles; Gmail dots and +tags; mistyped domains; junk contacts
             (test@test.com, 0400 000 000, ACMA's fictional 0491 570 xxx); unit addresses as
             3/12, Unit 3, U3; street types abbreviated; moves; DD/MM vs MM/DD dates

Anglo-Celtic names follow real frequencies, so a full name identifies about as often as it
would in a real customer base: surnames from the US Census 2000 surname list (weighted by count
x share white), given names from US Social Security baby-name shares for the person's birth
year (both public domain; downloaded to data/names/ - see docs/reproducing.md), with 15% from a
list of names more common in Australia. The other groups use the lists below; Chinese given
names are mostly two syllables (Xiaoming, Jianhua), as they are in practice.

`--seed N` builds another population (data/au_customer_sN/) for repeating experiments over several
generated files, not one. `--no-shared` writes data/au_customer_noshared/: no family emails and no parent's contact on a
child's records - the world most favourable to exact-identifier matching (research/scripts/cdp_reassess.py).
`--scripts` writes the same file to data/au_customer_script/ with 40% of Chinese and Lebanese
customers' store records in Chinese characters or Arabic script (flagged in a `script` column).

Writes data/au_customer/source_{web,app,store}.parquet (with person_id and household_id as
truth columns the config does not map) and contact_truth.parquet (each email and mobile's
true class: personal, household, junk or recycled).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/au_customer"
SEED = 11
N_HOUSEHOLDS = 5200

# given names (male, female), surnames most common first, English names used by some,
# chance of family-name-first order in a record, chance of having only one name
GROUPS = {
    "anglo": dict(weight=0.62, order_swap=0.0, mononym=0.0,
                  male="james jack oliver william thomas michael david john peter robert matthew daniel "
                       "christopher andrew mark paul stephen anthony benjamin samuel joshua ryan luke "
                       "nicholas timothy scott jason craig brendan liam noah lachlan cooper harry".split(),
                  female="sarah jessica emma olivia charlotte emily rebecca melissa michelle jennifer "
                         "elizabeth katherine margaret patricia susan karen lisa kylie nicole amanda "
                         "rachel hannah chloe sophie grace ruby mia isla zoe georgia".split(),
                  surnames="smith jones williams brown wilson taylor johnson white martin anderson "
                           "thompson thomas walker harris ryan robinson kelly king campbell clarke "
                           "wright mitchell young lee murphy hughes baker cooper edwards evans hall "
                           "hill jackson morris ward scott stewart turner wood green davis collins "
                           "bennett mcdonald obrien sullivan fraser graham ross watson "
                           "gallagher byrne walsh burke dunn".split()),
    "chinese": dict(weight=0.07, order_swap=0.25, mononym=0.0,
                    male="wei jun hao lei ming jie yong qiang tao chao bo gang hui long".split(),
                    female="jing li min xin yan ying hui fang na mei xiu lan ling yu".split(),
                    surnames="wang li zhang liu chen yang huang zhao wu zhou xu sun ma zhu hu lin "
                             "guo he gao luo zheng liang xie tang".split(),
                    english_m="david kevin jason eric michael tony andy peter jack".split(),
                    english_f="jenny michelle grace amy vivian cindy helen lily annie".split()),
    "indian": dict(weight=0.06, order_swap=0.05, mononym=0.12,
                   male="amit rahul sandeep deepak ravi sunil vikram arjun harpreet gurpreet "
                        "manpreet rajesh suresh anil karthik".split(),
                   female="priya anjali pooja neha divya kavya sneha deepika harpreet gurpreet "
                          "simran lakshmi meena sunita".split(),
                   surnames="singh kumar patel sharma kaur shah gupta reddy rao nair iyer mehta "
                            "joshi verma malhotra chopra".split()),
    "italian": dict(weight=0.06, order_swap=0.0, mononym=0.0,
                    male="giuseppe giovanni salvatore antonio francesco domenico vincenzo luigi "
                         "carmelo rocco".split(),
                    female="maria giuseppina rosa anna carmela francesca angela teresa lucia "
                           "concetta".split(),
                    surnames="rossi russo ferrari esposito romano colombo ricci marino greco bruno "
                             "gallo conti de luca costa mancini lombardo".split(),
                    english_m={"giuseppe": "joe", "giovanni": "john", "salvatore": "sam",
                               "antonio": "tony", "francesco": "frank", "domenico": "dom",
                               "vincenzo": "vince", "luigi": "lou", "carmelo": "carl", "rocco": "rocky"},
                    english_f={"giuseppina": "josie", "rosa": "rose", "carmela": "carmel",
                               "francesca": "fran", "concetta": "connie", "teresa": "tess"}),
    "vietnamese": dict(weight=0.04, order_swap=0.3, mononym=0.0,
                       male="minh hung tuan duc long nam thanh quang hai khoa phuc dung".split(),
                       female="linh huong lan mai thu hoa ngoc trang phuong hanh thao anh".split(),
                       surnames="nguyen tran le pham hoang huynh phan vu vo dang bui do ho ngo duong ly".split(),
                       surname_weights=[38, 11, 9, 7, 5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 1],
                       english_m="tony kevin peter danny jimmy steven".split(),
                       english_f="jenny linda kim tina vivian cathy".split()),
    "greek": dict(weight=0.04, order_swap=0.0, mononym=0.0,
                  male="nikolaos georgios konstantinos dimitrios ioannis christos panagiotis "
                       "vasilios".split(),
                  female="maria eleni aikaterini vasiliki sofia georgia anastasia despina".split(),
                  surnames="papadopoulos georgiou nikolaidis konstantinidis papadakis dimitriou "
                           "ioannou vlahos karagiannis christodoulou".split(),
                  english_m={"nikolaos": "nick", "georgios": "george", "konstantinos": "con",
                             "dimitrios": "jim", "ioannis": "john", "christos": "chris",
                             "panagiotis": "peter", "vasilios": "bill"},
                  english_f={"eleni": "helen", "aikaterini": "kathy", "vasiliki": "vicky",
                             "anastasia": "tess", "despina": "debbie"}),
    "filipino": dict(weight=0.04, order_swap=0.0, mononym=0.0,
                     male="jose mark john paul christian ryan jerome carlo".split(),
                     female="maria joy cristina rose ana grace michelle jocelyn".split(),
                     surnames="santos reyes cruz bautista garcia mendoza ramos villanueva "
                              "castillo delos santos".split()),
    "arabic": dict(weight=0.03, order_swap=0.0, mononym=0.0,
                   male="mohammed ahmed ali hassan omar khaled youssef ibrahim".split(),
                   female="fatima zeinab mariam layla nour rania hiba sara".split(),
                   surnames="khoury haddad nasser saad hanna ibrahim el-hassan abdallah "
                            "chahine fakhoury".split()),
}
SPELLINGS = {"mohammed": ["mohamed", "muhammad", "mohammad"], "ahmed": ["ahmad"],
             "youssef": ["yousef", "joseph"], "zeinab": ["zainab"], "khaled": ["khalid"],
             "nguyen": ["nguyn", "ngyuen"], "huynh": ["huyn"], "khoury": ["khouri"],
             "katherine": ["catherine", "kathryn"], "jessica": ["jesica"], "rebecca": ["rebekah"],
             "sarah": ["sara"], "stephen": ["steven"], "matthew": ["mathew"]}
NICKNAMES = {"michael": ["mick", "mike"], "rebecca": ["bec", "becky"], "melissa": ["mel"],
             "robert": ["rob", "bob"], "william": ["will", "bill"], "elizabeth": ["liz", "beth"],
             "katherine": ["kate", "kath"], "christopher": ["chris"], "matthew": ["matt"],
             "daniel": ["dan", "danny"], "james": ["jim", "jimmy"], "jennifer": ["jen"],
             "stephen": ["steve"], "anthony": ["tony", "ant"], "thomas": ["tom"],
             "benjamin": ["ben"], "samuel": ["sam"], "joshua": ["josh"], "nicholas": ["nick"],
             "andrew": ["andy", "drew"], "timothy": ["tim"], "margaret": ["marg", "maggie"],
             "patricia": ["trish"], "jessica": ["jess"], "susan": ["sue"], "david": ["dave"],
             "peter": ["pete"], "jonathan": ["jon"], "charlotte": ["charlie"], "olivia": ["liv"],
             "emily": ["em"], "nicole": ["nic"], "amanda": ["mandy"], "lachlan": ["lachie"],
             "georgia": ["george"], "alexander": ["alex"], "hannah": ["han"], "kylie": ["ky"]}
SUBURBS = [("richmond", "vic", "3121"), ("st kilda", "vic", "3182"), ("footscray", "vic", "3011"),
           ("box hill", "vic", "3128"), ("springvale", "vic", "3171"), ("geelong", "vic", "3220"),
           ("parramatta", "nsw", "2150"), ("cabramatta", "nsw", "2166"), ("bondi", "nsw", "2026"),
           ("chatswood", "nsw", "2067"), ("liverpool", "nsw", "2170"), ("newcastle", "nsw", "2300"),
           ("bankstown", "nsw", "2200"), ("hurstville", "nsw", "2220"), ("penrith", "nsw", "2750"),
           ("sunnybank", "qld", "4109"), ("toowong", "qld", "4066"), ("southport", "qld", "4215"),
           ("cairns", "qld", "4870"), ("fremantle", "wa", "6160"), ("joondalup", "wa", "6027"),
           ("glenelg", "sa", "5045"), ("norwood", "sa", "5067"), ("hobart", "tas", "7000"),
           ("belconnen", "act", "2617"), ("darwin", "nt", "0800")]
STREETS = "high station church main park victoria george king queen william railway bridge " \
          "albert elizabeth hill river mill beach smith oxford princes pacific hume forest " \
          "lake acacia wattle banksia waratah jacaranda bottlebrush".split()
STREET_TYPES = {"street": "st", "road": "rd", "avenue": "ave", "drive": "dr", "court": "ct",
                "place": "pl", "crescent": "cres", "close": "cl", "parade": "pde",
                "terrace": "tce", "lane": "ln", "way": "wy", "grove": "gr", "circuit": "cct"}
DOMAINS = [("gmail.com", 42), ("hotmail.com", 10), ("outlook.com", 6), ("icloud.com", 10),
           ("bigpond.com", 8), ("yahoo.com.au", 5), ("optusnet.com.au", 3), ("live.com.au", 3),
           ("bigpond.net.au", 2), ("tpg.com.au", 2), ("iinet.net.au", 2)]
DOMAIN_TYPOS = {"gmail.com": ["gmial.com", "gmail.con", "gmai.com"], "hotmail.com": ["hotmial.com", "hotmail.con"],
                "bigpond.com": ["bigpond.con", "bigpong.com"], "icloud.com": ["iclod.com"]}
JUNK_EMAILS = ["test@test.com", "noemail@noemail.com", "none@none.com", "a@a.com",
               "no@email.com", "asdf@asdf.com", "customer@store.com.au", "na@na.com"]
JUNK_MOBILES = ["0400000000", "0412345678", "0400123456", "0411111111", "0000000000",
                "0491570156", "0491570157", "0491570006", "0491570110"]   # last four: ACMA fictional
NAMES = ROOT / "data/names"
PINYIN = ("wei jun hao lei ming jie yong qiang tao chao bo gang hui long jing li min xin yan ying "
          "fang na mei xiu lan ling yu xiao hong jian hua zhi guo wen ping hai dong feng yi chen "
          "yun qing rui xue jia zi han yang").split()
CHINESE_SURNAMES = ("wang li zhang liu chen yang huang zhao wu zhou xu sun ma zhu hu lin guo he gao "
                    "luo zheng liang xie tang song deng han feng cao peng zeng xiao tian dong pan yuan "
                    "cai jiang yu du ye cheng wei su lu ding ren shen yao").split()
HOUSEHOLD_SIZES = [(1, 0.25), (2, 0.34), (3, 0.16), (4, 0.15), (5, 0.07), (6, 0.03)]


class Gen:
    def __init__(self, seed):
        self.r = np.random.default_rng(seed)
        self.used_mobiles = set()
        s = pd.read_csv(NAMES / "census2000_surnames.csv", keep_default_na=False)
        s = s[s.pctwhite != "(S)"]
        w = s["count"].astype(float) * s.pctwhite.astype(float) / 100
        s = s.assign(w=w).nlargest(30000, "w")
        self.surnames, self.surname_p = s.name.str.lower().to_numpy(), (s.w / s.w.sum()).to_numpy()
        b = pd.read_csv(NAMES / "ssa_baby_names.csv")
        self.given = {(sex, y): (g.name.str.lower().to_numpy(), (g.percent / g.percent.sum()).to_numpy())
                      for (sex, y), g in b.groupby(["sex", "year"])}

    def anglo_surname(self):
        return self.surnames[self.r.choice(len(self.surnames), p=self.surname_p)]

    def anglo_given(self, sex, year):
        names, p = self.given[("boy" if sex == "m" else "girl", int(min(max(year, 1880), 2008)))]
        return names[self.r.choice(len(names), p=p)]

    def p(self, prob):
        return self.r.random() < prob

    def pick(self, seq, weights=None):
        if weights is None:
            return seq[self.r.integers(len(seq))]
        w = np.asarray(weights, float)
        return seq[self.r.choice(len(seq), p=w / w.sum())]

    def zipf(self, seq, weights=None):
        return self.pick(seq, weights if weights is not None else [1 / (i + 1) ** 0.8 for i in range(len(seq))])

    def mobile(self):
        while True:
            m = "04" + "".join(str(d) for d in self.r.integers(0, 10, 8))
            if m not in self.used_mobiles and m not in JUNK_MOBILES and not m.startswith("0491570"):
                self.used_mobiles.add(m)
                return m

    def dob(self, age):
        y = 2026 - age
        return pd.Timestamp(year=y, month=int(self.r.integers(1, 13)), day=int(self.r.integers(1, 29)))


def surname_for(g, group):
    if group == "anglo":
        return g.anglo_surname()
    if group == "chinese":
        return g.zipf(CHINESE_SURNAMES)
    G = GROUPS[group]
    return g.zipf(G["surnames"], G.get("surname_weights"))


def given_for(g, group, sex, age):
    G = GROUPS[group]
    if group == "anglo" and g.p(0.85):
        return g.anglo_given(sex, 2026 - age)
    if group == "chinese" and g.p(0.7):
        return g.pick(PINYIN) + g.pick(PINYIN)
    return g.pick(G["male"] if sex == "m" else G["female"])


def person(g, group, sex, age, surname):
    G = GROUPS[group]
    given = given_for(g, group, sex, age)
    english = None
    eng = G.get("english_m" if sex == "m" else "english_f")
    if isinstance(eng, dict) and given in eng and g.p(0.6):
        english = eng[given]
    elif isinstance(eng, list) and g.p(0.4):
        english = g.pick(eng)
    middle = ("van" if sex == "m" else "thi") if group == "vietnamese" else (
        given_for(g, group, sex, age) if g.p(0.4) else "")
    return dict(group=group, sex=sex, age=age, given=given, surname=surname, maiden=None,
                english=english, nick=NICKNAMES.get(given), middle=middle, dob=g.dob(age),
                mononym=G["mononym"] > 0 and g.p(G["mononym"]))


def build_people(g):
    groups = list(GROUPS)
    weights = [GROUPS[k]["weight"] for k in groups]
    people, households = [], []
    for hid in range(N_HOUSEHOLDS):
        size = g.pick([s for s, _ in HOUSEHOLD_SIZES], [w for _, w in HOUSEHOLD_SIZES])
        size = size + int(g.r.integers(0, 2)) if size == 6 else size
        suburb = g.pick(SUBURBS)
        stype = g.pick(list(STREET_TYPES))
        addr = dict(number=str(int(g.r.integers(1, 250))), street=g.pick(STREETS), stype=stype,
                    unit=str(int(g.r.integers(1, 30))) if g.p(0.3) else "", suburb=suburb)
        members = []
        share_house = size >= 3 and g.p(0.08)
        group = g.pick(groups, weights)
        family = surname_for(g, group)
        if share_house:
            for _ in range(size):
                gr = g.pick(groups, weights)
                members.append(person(g, gr, g.pick("mf"), int(g.r.integers(19, 40)), surname_for(g, gr)))
        else:
            head_age = int(g.r.integers(22, 88))
            head_sex = g.pick("mf")
            members.append(person(g, group, head_sex, head_age, family))
            if size >= 2 and (g.p(0.75) or size == 2):
                pg = group if g.p(0.88) else g.pick(groups, weights)
                pw = person(g, pg, "f" if head_sex == "m" else "m",
                            max(19, head_age + int(g.r.integers(-6, 7))), family)
                own = surname_for(g, pg)
                if pw["sex"] == "f" and group == "anglo" and g.p(0.7):
                    pw["maiden"] = own            # took the family name; old records use this
                else:
                    pw["surname"] = own
                members.append(pw)
            while len(members) < size:
                child_age = int(g.r.integers(0, min(35, max(1, head_age - 18))))
                members.append(person(g, group, g.pick("mf"), child_age, family))
        for m in members:
            m["household_id"] = hid
            m["address"] = addr
            if g.p(0.15):                          # moved: older records carry a previous address
                m["old_address"] = dict(number=str(int(g.r.integers(1, 250))), street=g.pick(STREETS),
                                        stype=g.pick(list(STREET_TYPES)), unit="", suburb=g.pick(SUBURBS))
            people.append(m)
        households.append(members)
    for pid, m in enumerate(people):
        m["person_id"] = pid
    return people, households


def slug(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def email_for(g, m, first=None):
    f, l = slug(first or m["given"]), slug(m["surname"] or m["given"])
    local = g.pick([f"{f}.{l}", f"{f}{l}", f"{f}_{l}", f"{f[0]}{l}", f"{f}{l[0]}",
                    f"{f}{m['dob'].year % 100:02d}", f"{f}.{l}{int(g.r.integers(1, 99))}",
                    f"{l}.{f}", f"{f}{int(g.r.integers(1, 999))}"],
                   [20, 12, 4, 10, 4, 10, 12, 4, 10])
    return f"{local}@{g.pick([d for d, _ in DOMAINS], [w for _, w in DOMAINS])}"


def assign_contacts(g, people, households):
    """Mobiles and emails, and each value's true class."""
    truth = {}
    for m in people:
        m["mobiles"], m["emails"] = [], []
        if m["age"] >= 14 and g.p(0.95):
            m["mobiles"].append(g.mobile())
            if g.p(0.12):                          # changed number; old records carry the first
                m["mobiles"].append(g.mobile())
        if m["age"] >= 14 and g.p(0.9):
            m["emails"].append(email_for(g, m, m["english"] or (g.pick(m["nick"]) if m["nick"] and g.p(0.3) else None)))
            if g.p(0.15):
                m["emails"].append(email_for(g, m))
        for v in m["mobiles"] + m["emails"]:
            truth[v] = "personal"
    for members in (households if SHARED else []):   # family email shared by a couple
        adults = [m for m in members if m["age"] >= 19]
        if len(adults) >= 2 and adults[0]["surname"] and g.p(0.2):
            fam = f"{slug(adults[0]['surname'])}family{int(g.r.integers(1, 99))}@{g.pick(['bigpond.com', 'gmail.com', 'optusnet.com.au'])}"
            for a in adults[:2]:
                a["emails"].insert(0, fam)
            truth[fam] = "household"
        for kid in [m for m in members if m["age"] < 18]:   # a parent's contact on a child's records
            parent = adults[0] if adults else None
            if parent and parent["mobiles"] and g.p(0.6):
                kid["shared_mobile"] = parent["mobiles"][-1]
                truth[parent["mobiles"][-1]] = "household"
            if parent and parent["emails"] and g.p(0.5):
                kid["shared_email"] = parent["emails"][0]
                truth[parent["emails"][0]] = "household"
    # recycled numbers: a number someone gave up is later reassigned to an unrelated person
    churned = [m for m in people if len(m["mobiles"]) == 2]
    for m in churned:
        if g.p(0.2):
            other = people[int(g.r.integers(len(people)))]
            if other["household_id"] != m["household_id"] and other["mobiles"]:
                other["mobiles"][-1] = m["mobiles"][0]
                truth[m["mobiles"][0]] = "recycled"
    for v in JUNK_EMAILS + JUNK_MOBILES:
        truth[v] = "junk"
    return truth


# ------------------------------------------------------------------ rendering one record
def typo(g, s):
    if len(s) < 3:
        return s
    i = int(g.r.integers(1, len(s) - 1))
    k = g.r.integers(3)
    if k == 0:
        return s[:i] + s[i + 1:]
    if k == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + g.pick(list("aeioursnt")) + s[i + 1:]


def render_first(g, m):
    if m["english"] and g.p(0.5):
        first = m["english"]
    elif m["nick"] and g.p(0.3):
        first = g.pick(m["nick"])
    else:
        first = m["given"]
    if first in SPELLINGS and g.p(0.35):
        first = g.pick(SPELLINGS[first])
    return first


def render_name(g, m, old):
    first = render_first(g, m)
    last = (m["maiden"] if old and m["maiden"] else m["surname"])
    if last in SPELLINGS and g.p(0.25):
        last = g.pick(SPELLINGS[last])
    if m["mononym"]:
        last = g.pick(["", "", first, ".", "lnu"])
    if m["group"] == "vietnamese" and g.p(0.3):
        first = f"{first} {m['middle']}" if g.p(0.5) else f"{m['middle']} {first}"
    if GROUPS[m["group"]]["order_swap"] and g.p(GROUPS[m["group"]]["order_swap"]) and last:
        first, last = last, first
    if g.p(0.04):
        first = typo(g, first)
    if last and g.p(0.04):
        last = typo(g, last)
    style = g.r.integers(3)
    f = (str.title, str.upper, str.lower)[style]
    return f(first), f(last)


def render_mobile(g, m, old):
    v = m.get("shared_mobile") if m.get("shared_mobile") and (not m["mobiles"] or g.p(0.5)) else None
    if v is None:
        if not m["mobiles"]:
            return ""
        v = m["mobiles"][0] if old else m["mobiles"][-1]
    d = v[1:]
    return g.pick([v, f"{v[:4]} {v[4:7]} {v[7:]}", f"+61 {d[:3]} {d[3:6]} {d[6:]}", f"+61{d}",
                   f"61{d}", d, f"({v[:2]}) {v[2:6]} {v[6:]}"], [30, 30, 12, 8, 6, 8, 6])


def render_email(g, m, old):
    if m.get("shared_email") and (not m["emails"] or g.p(0.5)):
        e = m["shared_email"]
    elif not m["emails"]:
        return ""
    else:
        e = m["emails"][-1] if old or len(m["emails"]) == 1 else m["emails"][int(g.r.integers(len(m["emails"])))]
    local, dom = e.split("@")
    if dom == "gmail.com" and g.p(0.06):
        local = g.pick([local.replace(".", ""), f"{local}+shop", ".".join(local)])
    if dom in DOMAIN_TYPOS and g.p(0.02):
        dom = g.pick(DOMAIN_TYPOS[dom])
    e = f"{local}@{dom}"
    return e.upper() if g.p(0.05) else e


def render_address(g, a):
    stype = a["stype"] if g.p(0.4) else STREET_TYPES[a["stype"]]
    street = f"{a['street']} {stype}"
    if g.p(0.03):
        street = typo(g, street)
    num = a["number"]
    if a["unit"]:
        num = g.pick([f"{a['unit']}/{num}", f"unit {a['unit']}, {num}", f"u{a['unit']} {num}",
                      f"{num}"], [50, 25, 15, 10])
    sub, state, pc = a["suburb"]
    tail = g.pick([f", {sub} {state} {pc}", f" {sub} {state.upper()} {pc}", f", {sub}", ""], [50, 20, 20, 10])
    return f"{num} {street}{tail}"


def render_dob(g, d, fmt):
    day, mon = d.day, d.month
    if g.p(0.02) and day <= 12:
        day, mon = mon, day                        # US-style entry
    y = d.year + (int(g.pick([-1, 1])) if g.p(0.01) else 0)
    return f"{day:02d}/{mon:02d}/{y}" if fmt == "au" else f"{y}-{mon:02d}-{day:02d}"


CHANNELS = {
    #          presence(adult, teen), dup, surname, mobile, email, dob, address/postcode, junk email, junk mobile
    "web": dict(presence=(0.6, 0.3), dup=0.12, surname=0.55, mobile=0.35, email=1.0, dob=0.25,
                place=0.5, junk_email=0.05, junk_mobile=0.08, dob_fmt="au"),
    "app": dict(presence=(0.45, 0.4), dup=0.06, surname=0.35, mobile=1.0, email=0.5, dob=0.6,
                place=0.0, junk_email=0.03, junk_mobile=0.01, dob_fmt="iso"),
    "store": dict(presence=(0.55, 0.2), dup=0.08, surname=0.97, mobile=0.6, email=0.35, dob=0.45,
                  place=0.85, junk_email=0.08, junk_mobile=0.05, dob_fmt="au"),
}


def records(g, people):
    rows = {c: [] for c in CHANNELS}
    for m in people:
        for ch, C in CHANNELS.items():
            p_in = C["presence"][0] if m["age"] >= 18 else C["presence"][1] if m["age"] >= 14 else \
                (0.05 if ch == "store" else 0.0)
            if not g.p(p_in):
                continue
            for _ in range(2 if g.p(C["dup"]) else 1):
                old = g.p(0.3)
                first, last = render_name(g, m, old)
                if not g.p(C["surname"]):
                    last = ""
                mob = render_mobile(g, m, old) if g.p(C["mobile"]) else ""
                if C["mobile"] == 1.0 and not mob:
                    continue                      # the app cannot be used without a mobile
                if g.p(C["junk_mobile"]):
                    mob = g.pick(JUNK_MOBILES)
                em = render_email(g, m, old) if g.p(C["email"]) else ""
                if (C["email"] == 1.0 and not em) or g.p(C["junk_email"]):
                    em = g.pick(JUNK_EMAILS)
                addr = m.get("old_address") if old and m.get("old_address") else m["address"]
                row = dict(first_name=first, last_name=last, email=em, mobile=mob,
                           date_of_birth=render_dob(g, m["dob"], C["dob_fmt"]) if g.p(C["dob"]) else "",
                           person_id=m["person_id"], household_id=m["household_id"])
                if ch == "store":
                    row["address"] = render_address(g, addr) if g.p(C["place"]) else ""
                elif ch == "web":
                    row["postcode"] = addr["suburb"][2] if g.p(C["place"]) else ""
                rows[ch].append(row)
    out = {}
    for ch, rs in rows.items():
        d = pd.DataFrame(rs).sample(frac=1, random_state=int(g.r.integers(1 << 30))).reset_index(drop=True)
        d.insert(0, "record_id", [f"{ch[0]}{i:06d}" for i in range(len(d))])
        out[ch] = d
    return out


HANZI_SURNAME = dict(zip(CHINESE_SURNAMES, "王 李 张 刘 陈 杨 黄 赵 吴 周 徐 孙 马 朱 胡 林 郭 何 高 罗 郑 梁 谢 唐 宋 "
                         "邓 韩 冯 曹 彭 曾 肖 田 董 潘 袁 蔡 蒋 余 杜 叶 程 魏 苏 卢 丁 任 沈 姚".split()))
HANZI_SYLLABLE = dict(zip(PINYIN, "伟 军 浩 磊 明 杰 勇 强 涛 超 波 刚 辉 龙 静 丽 敏 欣 燕 英 芳 娜 梅 秀 兰 玲 宇 晓 红 建 "
                          "华 志 国 文 平 海 东 峰 怡 晨 云 青 瑞 雪 佳 子 涵 洋".split()))
ARABIC = {"mohammed": "محمد", "ahmed": "أحمد", "ali": "علي", "hassan": "حسن", "omar": "عمر", "khaled": "خالد",
          "youssef": "يوسف", "ibrahim": "إبراهيم", "fatima": "فاطمة", "zeinab": "زينب", "mariam": "مريم",
          "layla": "ليلى", "nour": "نور", "rania": "رانيا", "hiba": "هبة", "sara": "سارة", "khoury": "خوري",
          "haddad": "حداد", "nasser": "ناصر", "saad": "سعد", "hanna": "حنا", "el-hassan": "الحسن",
          "abdallah": "عبدالله", "chahine": "شاهين", "fakhoury": "فاخوري"}


def hanzi_given(pinyin):
    """Greedy longest-syllable split of a pinyin given name into characters; None if it fails."""
    out, s = "", pinyin
    while s:
        syl = next((p for p in sorted(HANZI_SYLLABLE, key=len, reverse=True) if s.startswith(p)), None)
        if syl is None:
            return None
        out, s = out + HANZI_SYLLABLE[syl], s[len(syl):]
    return out


def native_script(srcs, people, rate=0.4, seed=5):
    """Rewrite some store records of Chinese and Lebanese customers in their own script, with a
    separate random stream so everything else matches the Latin-only file record for record."""
    r = np.random.default_rng(seed)
    by_id = {m["person_id"]: m for m in people}
    d = srcs["store"].copy()
    d["script"] = False
    for k, pid in enumerate(d.person_id):
        m = by_id[pid]
        if m["group"] not in ("chinese", "arabic") or r.random() >= rate:
            continue
        if m["group"] == "chinese":
            given, fam = hanzi_given(m["given"]), HANZI_SURNAME.get(m["surname"])
            if not (given and fam):
                continue
            first, last = (fam + given, "") if r.random() < 0.3 else (given, fam)
        else:
            given, fam = ARABIC.get(m["given"]), ARABIC.get(m["surname"])
            if not (given and fam):
                continue
            first, last = given, fam
        d.loc[k, ["first_name", "last_name", "script"]] = [first, last, True]
    srcs = dict(srcs)
    srcs["store"] = d
    return srcs


SHARED = True


def main():
    import sys
    global SHARED
    SHARED = "--no-shared" not in sys.argv
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else SEED
    g = Gen(seed)
    people, households = build_people(g)
    truth = assign_contacts(g, people, households)
    srcs = records(g, people)
    out_dir = OUT
    if not SHARED:
        out_dir = ROOT / "data/au_customer_noshared"
    if seed != SEED:
        out_dir = ROOT / f"data/au_customer_s{seed}"
    if "--scripts" in sys.argv:        # the same file with some names in Chinese and Arabic script
        srcs = native_script(srcs, people)
        out_dir = ROOT / "data/au_customer_script"
    OUT_ = out_dir
    OUT_.mkdir(parents=True, exist_ok=True)
    for ch, d in srcs.items():
        d.to_parquet(OUT_ / f"source_{ch}.parquet", index=False)
    pd.DataFrame({"value": list(truth), "contact_class": list(truth.values())}).to_parquet(
        OUT_ / "contact_truth.parquet", index=False)
    all_rec = pd.concat(srcs.values())
    stats = {"people": len(people), "households": len(households),
             "people_with_records": int(all_rec.person_id.nunique()),
             "records": {ch: len(d) for ch, d in srcs.items()},
             "records_without_surname": int((all_rec.last_name == "").sum()),
             "records_with_only_first_name_and_one_contact": int(
                 ((all_rec.last_name == "") & (all_rec.date_of_birth == "") &
                  ((all_rec.email == "") ^ (all_rec.mobile == ""))).sum()),
             "groups": pd.Series([m["group"] for m in people]).value_counts().to_dict(),
             **({"records_in_native_script": int(srcs["store"].script.sum())} if "script" in srcs["store"] else {})}
    (OUT_ / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
