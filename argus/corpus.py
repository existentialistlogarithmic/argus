"""Synthetic multi-source corpus generator.

The point of this module is not to make data -- it is to make *hard* data. Any
resolver looks perfect on clean records. What separates a real entity
resolution engine from a `GROUP BY` is what happens when the same person shows
up as "Robert J. Whitlock, b. 1974-03-08" in a national registry, "Bob
Whitlock" with only an email in an HR export, and a bare phone number in comms
metadata.

So every source here has a different schema, a different subset of the truth,
and its own characteristic corruptions. Ground truth is emitted separately, is
never present in the corpus files themselves, and is read only by the evaluator.

A network is planted inside the noise: a set of people who co-control shell
companies registered to one address, moving money in cycles in amounts sitting
just below a reporting threshold. None of it is visible in any single source.
It only appears after resolution stitches the sources together.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

FIRST_NAMES = [
    "James", "Robert", "William", "Katherine", "Elizabeth", "Margaret", "Michael",
    "Daniel", "Thomas", "Christopher", "Alexander", "Nicholas", "Benjamin",
    "Victoria", "Rebecca", "Natalie", "Susan", "Jennifer", "Abigail", "Sophia",
    "Isabella", "Anthony", "Gregory", "Lawrence", "Raymond", "Walter", "Gabriel",
    "Stephen", "Edward", "Samuel", "Matthew", "Andrew", "Peter", "Joseph",
    "Catherine", "Patricia", "Laura", "Helena", "Marcus", "Julian", "Adrian",
    "Yuriy", "Dmitri", "Anastasia", "Ekaterina", "Farida", "Omar", "Leila",
    "Hassan", "Amira", "Chen", "Wei", "Mei", "Hiroshi", "Yuki", "Priya", "Rajesh",
]

MIDDLE_NAMES = ["J", "A", "M", "R", "L", "P", "E", "T", "C", "K", "", "", "", ""]

LAST_NAMES = [
    "Whitlock", "Ashford", "Brennan", "Calloway", "Duarte", "Ellsworth", "Fairbanks",
    "Garrison", "Hollis", "Ingram", "Jarvis", "Kowalski", "Lindqvist", "Marchetti",
    "Novak", "Oyelaran", "Pemberton", "Quintero", "Rasmussen", "Sorenson",
    "Thornbury", "Ustinov", "Vasquez", "Whitfield", "Yarrow", "Zielinski",
    "Abernathy", "Bellweather", "Castellanos", "Drummond", "Eastwood", "Fontaine",
    "Grimaldi", "Havilland", "Ionescu", "Jankowski", "Kirkpatrick", "Lindstrom",
    "Moreau", "Nakamura", "Okonkwo", "Petrov", "Ramirez", "Sandoval", "Tanaka",
    "Underwood", "Volkov", "Wainwright", "Xiao", "Yamamoto",
]

STREETS = [
    "Aldgate", "Bishopsgate", "Cornhill", "Devonshire", "Eastcheap", "Fenchurch",
    "Gracechurch", "Houndsditch", "Ironmonger", "Jewry", "Kingsway", "Leadenhall",
    "Moorgate", "Newgate", "Old Broad", "Poultry", "Queen Victoria", "Rood",
    "Seething", "Threadneedle",
]
STREET_TYPES = ["Street", "Road", "Avenue", "Lane", "Court", "Square"]

CITIES = [
    ("London", "GB", "EC2M"), ("Manchester", "GB", "M1"), ("Dublin", "IE", "D02"),
    ("Nicosia", "CY", "1010"), ("Valletta", "MT", "VLT"), ("Riga", "LV", "LV-1050"),
    ("Zurich", "CH", "8001"), ("Luxembourg", "LU", "L-1150"), ("Panama City", "PA", "0801"),
    ("Dubai", "AE", "00000"), ("Singapore", "SG", "049315"), ("Valencia", "ES", "46001"),
]

SECTORS = [
    "logistics", "commodities trading", "consulting", "real estate", "shipping",
    "software", "pharmaceuticals", "construction", "media", "financial services",
]

ORG_HEADS = [
    "Meridian", "Ardent", "Cobalt", "Northgate", "Silverline", "Havenport",
    "Redstone", "Blackwater", "Everglade", "Kestrel", "Lodestar", "Marlin",
    "Orchid", "Pinnacle", "Quarry", "Sablewood", "Talon", "Verity", "Westmark",
    "Zephyr", "Halcyon", "Ironvale", "Juniper", "Larkspur",
]
ORG_TAILS = [
    "Holdings", "Trading", "Capital", "Ventures", "Partners", "Group",
    "Industries", "Logistics", "Consulting", "Enterprises",
]
ORG_FORMS = ["Ltd", "Limited", "LLC", "Inc", "GmbH", "S.A.", "PLC", "BV"]

BANKS = [
    "Northern Trust Bank", "Baltic Commercial Bank", "Levantine Credit",
    "Aegean Mercantile", "Helvetic Private Bank", "Gulf International Bank",
]

OCCUPATIONS = [
    "director", "consultant", "engineer", "accountant", "logistics manager",
    "sales manager", "analyst", "solicitor", "broker", "administrator",
]

CHANNELS = ["voice", "sms", "email", "messaging"]


@dataclass
class TruePerson:
    id: str
    first: str
    middle: str
    last: str
    dob: date
    national_id: str
    email: str
    phone: str
    address: str
    city: str
    country: str
    postcode: str
    nationality: str
    occupation: str
    is_ring: bool = False

    @property
    def full(self) -> str:
        return " ".join(p for p in (self.first, self.middle, self.last) if p)


@dataclass
class TrueOrg:
    id: str
    name: str
    form: str
    reg_number: str
    country: str
    address: str
    city: str
    postcode: str
    incorporated: date
    sector: str
    is_shell: bool = False

    @property
    def legal(self) -> str:
        return f"{self.name} {self.form}"


@dataclass
class TrueAccount:
    id: str
    number: str
    iban: str
    institution: str
    currency: str
    opened: date
    owner_person: str | None = None
    owner_org: str | None = None


@dataclass
class Corpus:
    people: list[TruePerson] = field(default_factory=list)
    orgs: list[TrueOrg] = field(default_factory=list)
    accounts: list[TrueAccount] = field(default_factory=list)
    employment: list[tuple] = field(default_factory=list)
    officers: list[tuple] = field(default_factory=list)
    stakes: list[tuple] = field(default_factory=list)
    transactions: list[dict] = field(default_factory=list)
    comms: list[dict] = field(default_factory=list)
    watchlist: list[str] = field(default_factory=list)
    ring_people: list[str] = field(default_factory=list)
    ring_orgs: list[str] = field(default_factory=list)
    truth: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# corruption helpers -- each models a specific real-world failure mode
# --------------------------------------------------------------------------

def _typo(rng: random.Random, text: str) -> str:
    """Introduce one realistic keyboard-level error."""
    if len(text) < 4:
        return text
    mode = rng.choice(("swap", "drop", "double", "sub"))
    index = rng.randrange(1, len(text) - 1)
    if mode == "swap":
        return text[:index] + text[index + 1] + text[index] + text[index + 2:]
    if mode == "drop":
        return text[:index] + text[index + 1:]
    if mode == "double":
        return text[:index] + text[index] + text[index:]
    neighbours = {"a": "s", "e": "r", "i": "o", "o": "i", "n": "m", "s": "a", "t": "r", "l": "k"}
    return text[:index] + neighbours.get(text[index].lower(), text[index]) + text[index + 1:]


_REVERSE_NICKNAMES = {
    "william": "Bill", "robert": "Bob", "richard": "Rick", "james": "Jim",
    "joseph": "Joe", "michael": "Mike", "thomas": "Tom", "stephen": "Steve",
    "christopher": "Chris", "anthony": "Tony", "edward": "Ted", "samuel": "Sam",
    "daniel": "Dan", "matthew": "Matt", "nicholas": "Nick", "alexander": "Alex",
    "katherine": "Kate", "elizabeth": "Liz", "margaret": "Maggie", "susan": "Sue",
    "jennifer": "Jenny", "rebecca": "Becky", "abigail": "Abby", "victoria": "Vicky",
    "andrew": "Andy", "peter": "Pete", "benjamin": "Ben", "gregory": "Greg",
    "lawrence": "Larry", "raymond": "Ray", "walter": "Walt", "gabriel": "Gabe",
    "natalie": "Nat", "isabella": "Bella", "sophia": "Sofia",
}


def _name_variant(rng: random.Random, person: TruePerson, noise: float) -> str:
    """Render a person's name the way some particular source system would."""
    first, last = person.first, person.last
    style = rng.random()
    if style < 0.18:
        nickname = _REVERSE_NICKNAMES.get(first.lower())
        if nickname:
            first = nickname
    elif style < 0.30:
        first = first[0] + "."
    text = f"{first} {person.middle} {last}".replace("  ", " ").strip() if rng.random() < 0.45 \
        else f"{first} {last}"
    if rng.random() < 0.06:
        text = f"{last}, {first}"
    if rng.random() < noise:
        text = _typo(rng, text)
    if rng.random() < 0.05:
        text = text.upper()
    if rng.random() < 0.04:
        text = f"Mr. {text}" if person.first not in ("Katherine", "Elizabeth") else f"Ms. {text}"
    return text


def _date_variant(rng: random.Random, value: date, noise: float) -> str | None:
    """Render a date, sometimes wrongly, in one of several source formats."""
    if rng.random() < noise * 0.5:
        mode = rng.random()
        if mode < 0.4 and value.day <= 12:
            value = date(value.year, value.day, value.month)  # day/month transposition
        elif mode < 0.7:
            value = value + timedelta(days=rng.choice((-1, 1)))
        else:
            return None
    fmt = rng.choice(("%Y-%m-%d", "%d/%m/%Y", "%d %b %Y"))
    return value.strftime(fmt)


def _phone_variant(rng: random.Random, phone: str) -> str:
    digits = phone
    style = rng.randrange(4)
    if style == 0:
        return f"+44 {digits[:3]} {digits[3:6]} {digits[6:]}"
    if style == 1:
        return f"0{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if style == 2:
        return f"({digits[:3]}) {digits[3:]}"
    return digits


def _email_variant(rng: random.Random, email: str) -> str:
    local, _, domain = email.partition("@")
    style = rng.randrange(4)
    if style == 0 and domain == "gmail.com" and len(local) > 4:
        cut = len(local) // 2
        return f"{local[:cut]}.{local[cut:]}@{domain}"
    if style == 1:
        return f"{local}+{rng.choice(('work', 'personal', 'biz'))}@{domain}"
    if style == 2:
        return email.upper()
    return email


def _address_variant(rng: random.Random, address: str) -> str:
    text = address
    replacements = {
        "Street": "St", "Road": "Rd", "Avenue": "Ave", "Lane": "Ln",
        "Court": "Ct", "Square": "Sq",
    }
    if rng.random() < 0.45:
        for full, abbrev in replacements.items():
            text = text.replace(full, abbrev)
    if rng.random() < 0.2:
        text = text.replace(",", "")
    return text


def _maybe(rng: random.Random, value, keep: float):
    """Drop a value with probability ``1 - keep``; sources are always partial."""
    return value if rng.random() < keep else ""


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

class CorpusBuilder:
    def __init__(self, seed: int = 20260813, n_people: int = 300, n_orgs: int = 80,
                 noise: float = 0.28):
        self.rng = random.Random(seed)
        self.n_people = n_people
        self.n_orgs = n_orgs
        self.noise = noise
        self.corpus = Corpus()

    # -- ground truth population ------------------------------------------

    def _make_people(self):
        rng = self.rng
        used_names = set()
        for index in range(self.n_people):
            while True:
                first = rng.choice(FIRST_NAMES)
                last = rng.choice(LAST_NAMES)
                # A handful of genuine name collisions are left in on purpose:
                # a resolver that merges on name alone must be punished for it.
                if (first, last) not in used_names or rng.random() < 0.04:
                    used_names.add((first, last))
                    break
            city, country, postcode = rng.choice(CITIES)
            person = TruePerson(
                id=f"T-P{index:04d}",
                first=first,
                middle=rng.choice(MIDDLE_NAMES),
                last=last,
                dob=date(rng.randint(1955, 1998), rng.randint(1, 12), rng.randint(1, 28)),
                national_id=f"{rng.choice('ABCDEFGHJK')}{rng.randrange(10**7, 10**8)}",
                email=f"{first.lower()}.{last.lower()}@{rng.choice(('gmail.com', 'outlook.com', 'protonmail.com', 'fastmail.com'))}",
                phone=str(rng.randrange(7000000000, 7999999999)),
                address=f"{rng.randint(1, 240)} {rng.choice(STREETS)} {rng.choice(STREET_TYPES)}",
                city=city,
                country=country,
                postcode=f"{postcode} {rng.randrange(1, 9)}{rng.choice('ABDEFGHJ')}{rng.choice('ABDEFGHJ')}",
                nationality=country,
                occupation=rng.choice(OCCUPATIONS),
            )
            self.corpus.people.append(person)

    def _make_orgs(self):
        rng = self.rng
        used_names = set()
        for index in range(self.n_orgs):
            # Company names must be distinct: two genuinely different firms
            # sharing a name would be unresolvable by any method, and would
            # show up as resolver error rather than as the data problem it is.
            while True:
                name = f"{rng.choice(ORG_HEADS)} {rng.choice(ORG_TAILS)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            city, country, postcode = rng.choice(CITIES)
            org = TrueOrg(
                id=f"T-O{index:04d}",
                name=name,
                form=rng.choice(ORG_FORMS),
                reg_number=f"{country}{rng.randrange(10**7, 10**8)}",
                country=country,
                address=f"{rng.randint(1, 200)} {rng.choice(STREETS)} {rng.choice(STREET_TYPES)}",
                city=city,
                postcode=f"{postcode} {rng.randrange(1, 9)}{rng.choice('ABDEFGHJ')}{rng.choice('ABDEFGHJ')}",
                incorporated=date(rng.randint(1998, 2023), rng.randint(1, 12), rng.randint(1, 28)),
                sector=rng.choice(SECTORS),
            )
            self.corpus.orgs.append(org)

    def _make_accounts(self):
        rng = self.rng
        counter = 0
        for person in self.corpus.people:
            for _ in range(rng.choices((1, 2, 3), weights=(70, 25, 5))[0]):
                counter += 1
                self.corpus.accounts.append(self._account(counter, owner_person=person.id))
        for org in self.corpus.orgs:
            for _ in range(rng.choices((1, 2), weights=(80, 20))[0]):
                counter += 1
                self.corpus.accounts.append(self._account(counter, owner_org=org.id))

    def _account(self, counter: int, owner_person=None, owner_org=None) -> TrueAccount:
        rng = self.rng
        number = f"{rng.randrange(10**9, 10**10)}"
        return TrueAccount(
            id=f"T-A{counter:05d}",
            number=number,
            iban=f"GB{rng.randrange(10, 99)}NWBK{rng.randrange(10**5, 10**6)}{number[:8]}",
            institution=rng.choice(BANKS),
            currency=rng.choices(("EUR", "USD", "GBP", "CHF"), weights=(40, 35, 20, 5))[0],
            opened=date(rng.randint(2005, 2024), rng.randint(1, 12), rng.randint(1, 28)),
            owner_person=owner_person,
            owner_org=owner_org,
        )

    def _make_relationships(self):
        rng = self.rng
        people = self.corpus.people
        orgs = self.corpus.orgs

        for person in people:
            if rng.random() < 0.82:
                employer = rng.choice(orgs)
                self.corpus.employment.append((person.id, employer.id, person.occupation,
                                               date(rng.randint(2010, 2024), rng.randint(1, 12), 1)))
        for org in orgs:
            for _ in range(rng.choices((1, 2, 3), weights=(50, 35, 15))[0]):
                officer = rng.choice(people)
                self.corpus.officers.append((officer.id, org.id,
                                             rng.choice(("director", "secretary", "shareholder")),
                                             date(rng.randint(2005, 2024), rng.randint(1, 12), 1)))
        for org in orgs:
            if rng.random() < 0.25:
                parent = rng.choice(orgs)
                if parent.id != org.id:
                    self.corpus.stakes.append((parent.id, org.id, round(rng.uniform(15, 100), 1)))

    def _make_background_activity(self):
        """Ordinary transactions and communications -- the haystack."""
        rng = self.rng
        accounts = self.corpus.accounts
        start = date(2024, 1, 1)
        # Scaled to the population so a small corpus stays small: a fixed
        # volume over few people makes every selector block enormous and turns
        # a quick test run into a quadratic one.
        n_transactions = max(200, len(accounts) * 8)
        n_communications = max(200, self.n_people * 10)
        for index in range(n_transactions):
            src, dst = rng.sample(accounts, 2)
            self.corpus.transactions.append({
                "id": f"TX{index:06d}",
                "from": src.number,
                "to": dst.number,
                "amount": round(rng.lognormvariate(6.2, 1.4), 2),
                "currency": src.currency,
                "date": start + timedelta(days=rng.randrange(400)),
                "reference": rng.choice(("invoice", "services", "consulting fee",
                                         "goods", "retainer", "settlement")),
            })
        for index in range(n_communications):
            a, b = rng.sample(self.corpus.people, 2)
            self.corpus.comms.append({
                "id": f"CM{index:06d}",
                "from": a.id,
                "to": b.id,
                "channel": rng.choice(CHANNELS),
                "date": start + timedelta(days=rng.randrange(400)),
            })

    def _plant_network(self):
        """Plant the signal.

        Six principals, five shell companies sharing one registered address,
        cyclic transfers structured just below a 10,000 reporting threshold,
        and dense off-hours communication. Every strand of it is split across
        sources so that no single dataset reveals the shape.
        """
        rng = self.rng
        principals = rng.sample(self.corpus.people, 6)
        shells = rng.sample([o for o in self.corpus.orgs], 5)

        shared_address = "17 Seething Lane"
        shared_city, shared_country, shared_postcode = "Nicosia", "CY", "1010"
        base_incorporation = date(2022, 6, 1)

        for offset, shell in enumerate(shells):
            shell.is_shell = True
            shell.address = shared_address
            shell.city = shared_city
            shell.country = shared_country
            shell.postcode = f"{shared_postcode} 4AB"
            shell.incorporated = base_incorporation + timedelta(days=offset * 11)
            shell.sector = "commodities trading"
            self.corpus.ring_orgs.append(shell.id)

        for person in principals:
            person.is_ring = True
            self.corpus.ring_people.append(person.id)

        # Each principal is an officer of at least two shells: the overlap is
        # what makes the cluster hold together under community detection.
        for index, person in enumerate(principals):
            for shell in (shells[index % len(shells)], shells[(index + 2) % len(shells)]):
                self.corpus.officers.append((person.id, shell.id, "director", base_incorporation))

        shell_accounts = []
        counter = 90000
        for shell in shells:
            counter += 1
            account = self._account(counter, owner_org=shell.id)
            account.institution = "Baltic Commercial Bank"
            account.currency = "EUR"
            account.opened = base_incorporation + timedelta(days=20)
            self.corpus.accounts.append(account)
            shell_accounts.append(account)

        # Round-tripping: value leaves the first account and returns to it,
        # in tranches that never breach the reporting threshold.
        start = date(2024, 3, 4)
        transaction_id = 900000
        for cycle in range(14):
            day = start + timedelta(days=cycle * 9 + rng.randrange(3))
            for step in range(len(shell_accounts)):
                src = shell_accounts[step]
                dst = shell_accounts[(step + 1) % len(shell_accounts)]
                transaction_id += 1
                self.corpus.transactions.append({
                    "id": f"TX{transaction_id:06d}",
                    "from": src.number,
                    "to": dst.number,
                    "amount": round(rng.uniform(8600, 9850), 2),
                    "currency": "EUR",
                    "date": day + timedelta(days=step),
                    "reference": rng.choice(("consulting fee", "logistics", "commodity settlement")),
                })

        # Principals also move personal money into the first shell account.
        person_accounts = {a.owner_person: a for a in self.corpus.accounts if a.owner_person}
        for person in principals:
            account = person_accounts.get(person.id)
            if not account:
                continue
            for _ in range(rng.randint(3, 6)):
                transaction_id += 1
                self.corpus.transactions.append({
                    "id": f"TX{transaction_id:06d}",
                    "from": account.number,
                    "to": shell_accounts[0].number,
                    "amount": round(rng.uniform(9000, 9900), 2),
                    "currency": "EUR",
                    "date": start + timedelta(days=rng.randrange(120)),
                    "reference": "loan repayment",
                })

        comm_id = 900000
        for i, a in enumerate(principals):
            for b in principals[i + 1:]:
                for _ in range(rng.randint(6, 14)):
                    comm_id += 1
                    self.corpus.comms.append({
                        "id": f"CM{comm_id:06d}",
                        "from": a.id,
                        "to": b.id,
                        "channel": rng.choice(("voice", "messaging")),
                        "date": start + timedelta(days=rng.randrange(150)),
                    })

        # One principal is known; everything else has to be inferred from them.
        self.corpus.watchlist.append(principals[0].id)
        for person in rng.sample(self.corpus.people, 8):
            if person.id not in self.corpus.watchlist:
                self.corpus.watchlist.append(person.id)

    def build(self) -> Corpus:
        self._make_people()
        self._make_orgs()
        self._make_accounts()
        self._make_relationships()
        self._make_background_activity()
        self._plant_network()
        return self.corpus


# --------------------------------------------------------------------------
# source emission -- one writer per source system
# --------------------------------------------------------------------------

class CorpusWriter:
    """Projects ground truth into deliberately inconsistent source files."""

    def __init__(self, corpus: Corpus, out_dir: Path, seed: int = 7, noise: float = 0.28):
        self.corpus = corpus
        self.out = Path(out_dir)
        self.rng = random.Random(seed)
        self.noise = noise
        self.truth: dict[str, str] = {}

    def _record_truth(self, key: str, truth_id: str):
        self.truth[key] = truth_id

    def write(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        stats = {
            "natreg": self._write_natreg(),
            "hr": self._write_hr(),
            "bank": self._write_bank(),
            "corpreg": self._write_corpreg(),
            "txn": self._write_transactions(),
            "comms": self._write_comms(),
            "watch": self._write_watchlist(),
        }
        (self.out / "truth.json").write_text(
            json.dumps({"records": self.truth}, indent=1), encoding="utf-8"
        )
        return stats

    def _write_natreg(self) -> int:
        """National registry: authoritative, complete, formal names."""
        rng = self.rng
        path = self.out / "natreg_persons.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["record_id", "surname", "given_names", "date_of_birth",
                             "national_id", "residential_address", "city", "country", "nationality"])
            count = 0
            for person in self.corpus.people:
                if rng.random() > 0.93:
                    continue  # a few people are simply absent from the registry
                local_id = f"NR{count:05d}"
                given = f"{person.first} {person.middle}".strip()
                writer.writerow([
                    local_id,
                    person.last.upper(),
                    given,
                    person.dob.strftime("%Y-%m-%d"),
                    person.national_id,
                    _address_variant(rng, person.address),
                    person.city,
                    person.country,
                    person.nationality,
                ])
                self._record_truth(f"natreg:{local_id}", person.id)
                count += 1
        return count

    def _write_hr(self) -> int:
        """Corporate HR: informal names, contact details, no dates of birth."""
        rng = self.rng
        employers = {org.id: org for org in self.corpus.orgs}
        people = {p.id: p for p in self.corpus.people}
        path = self.out / "hr_directory.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["staff_id", "display_name", "date_of_birth", "work_email", "mobile",
                             "job_title", "employer_id", "employer_name", "employer_country", "start_date"])
            count = 0
            for person_id, org_id, role, since in self.corpus.employment:
                person = people[person_id]
                org = employers[org_id]
                staff_id = f"HR{count:05d}"
                employer_local = f"HRORG{org_id[-4:]}"
                writer.writerow([
                    staff_id,
                    _name_variant(rng, person, self.noise),
                    _maybe(rng, _date_variant(rng, person.dob, self.noise) or "", 0.7),
                    _maybe(rng, _email_variant(rng, person.email), 0.88),
                    _maybe(rng, _phone_variant(rng, person.phone), 0.72),
                    role,
                    employer_local,
                    org.legal if rng.random() > self.noise * 0.4 else _typo(rng, org.legal),
                    org.country,
                    since.strftime("%d/%m/%Y"),
                ])
                self._record_truth(f"hr:{staff_id}", person.id)
                self._record_truth(f"hr:{employer_local}", org.id)
                count += 1
        return count

    def _write_bank(self) -> int:
        """Bank KYC: names plus dates of birth, and the account details."""
        rng = self.rng
        people = {p.id: p for p in self.corpus.people}
        orgs = {o.id: o for o in self.corpus.orgs}
        path = self.out / "bank_kyc.jsonl"
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for account in self.corpus.accounts:
                if rng.random() > 0.95:
                    continue
                local_id = f"BK{count:05d}"
                row = {
                    "kyc_id": local_id,
                    "account_no": account.number,
                    "iban": account.iban if rng.random() < 0.8 else None,
                    "bank": account.institution,
                    "ccy": account.currency,
                    "opened_on": account.opened.strftime("%d/%m/%Y"),
                }
                if account.owner_person:
                    person = people[account.owner_person]
                    row["holder_type"] = "individual"
                    row["holder_name"] = _name_variant(rng, person, self.noise)
                    row["holder_dob"] = _date_variant(rng, person.dob, self.noise)
                    row["holder_address"] = _maybe(rng, _address_variant(rng, person.address), 0.8)
                    row["holder_phone"] = _maybe(rng, _phone_variant(rng, person.phone), 0.55)
                    row["holder_email"] = _maybe(rng, _email_variant(rng, person.email), 0.6)
                    self._record_truth(f"bank:{local_id}H", account.owner_person)
                else:
                    org = orgs[account.owner_org]
                    row["holder_type"] = "corporate"
                    row["holder_name"] = org.legal if rng.random() > self.noise * 0.5 else _typo(rng, org.legal)
                    row["holder_reg"] = org.reg_number if rng.random() < 0.55 else None
                    row["holder_address"] = _address_variant(rng, org.address)
                    row["holder_country"] = org.country
                    self._record_truth(f"bank:{local_id}H", account.owner_org)
                self._record_truth(f"bank:{local_id}A", account.id)
                handle.write(json.dumps(row) + "\n")
                count += 1
        return count

    def _write_corpreg(self) -> int:
        """Company registry: legal names, registration numbers, officers."""
        rng = self.rng
        people = {p.id: p for p in self.corpus.people}
        officers_by_org: dict[str, list] = {}
        for person_id, org_id, role, since in self.corpus.officers:
            officers_by_org.setdefault(org_id, []).append((person_id, role, since))

        path = self.out / "company_registry.jsonl"
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for org in self.corpus.orgs:
                local_id = f"CR{count:05d}"
                officers = []
                for index, (person_id, role, since) in enumerate(officers_by_org.get(org.id, [])):
                    person = people[person_id]
                    officer_local = f"{local_id}O{index}"
                    officers.append({
                        "officer_id": officer_local,
                        "name": _name_variant(rng, person, self.noise * 0.8),
                        "dob": _date_variant(rng, person.dob, self.noise * 0.6),
                        "role": role,
                        "appointed": since.strftime("%d %b %Y"),
                        "nationality": person.nationality,
                    })
                    self._record_truth(f"corpreg:{officer_local}", person.id)
                handle.write(json.dumps({
                    "entry_id": local_id,
                    "company_name": org.legal,
                    "registration_number": org.reg_number,
                    "jurisdiction": org.country,
                    "incorporated_on": org.incorporated.strftime("%Y-%m-%d"),
                    "sic_description": org.sector,
                    "registered_office": {
                        "street": _address_variant(rng, org.address),
                        "city": org.city,
                        "postcode": org.postcode,
                        "country": org.country,
                    },
                    "officers": officers,
                }) + "\n")
                self._record_truth(f"corpreg:{local_id}", org.id)
                count += 1

            for parent_id, child_id, percent in self.corpus.stakes:
                handle.write(json.dumps({
                    "entry_kind": "shareholding",
                    "parent_reg": next(o.reg_number for o in self.corpus.orgs if o.id == parent_id),
                    "child_reg": next(o.reg_number for o in self.corpus.orgs if o.id == child_id),
                    "percent": percent,
                }) + "\n")
        return count

    def _write_transactions(self) -> int:
        """Payment rail data: account numbers only, no names at all."""
        path = self.out / "transactions.jsonl"
        # The thin account records implied by the payment rail are labelled too,
        # so that bank-to-rail account merges get scored rather than assumed.
        for account in self.corpus.accounts:
            self._record_truth(f"txn:ACC{account.number}", account.id)
        with path.open("w", encoding="utf-8") as handle:
            for txn in self.corpus.transactions:
                handle.write(json.dumps({
                    "txn_id": txn["id"],
                    "debit_account": txn["from"],
                    "credit_account": txn["to"],
                    "amount": txn["amount"],
                    "ccy": txn["currency"],
                    "value_date": txn["date"].strftime("%Y-%m-%d"),
                    "narrative": txn["reference"],
                }) + "\n")
        return len(self.corpus.transactions)

    def _write_comms(self) -> int:
        """Comms metadata: selectors only. Identity must be inferred."""
        rng = self.rng
        people = {p.id: p for p in self.corpus.people}
        path = self.out / "comms_metadata.jsonl"
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for event in self.corpus.comms:
                a, b = people[event["from"]], people[event["to"]]
                use_email = event["channel"] == "email"
                a_local, b_local = f"{event['id']}A", f"{event['id']}B"
                handle.write(json.dumps({
                    "event_id": event["id"],
                    "channel": event["channel"],
                    "originator": {
                        "selector_id": a_local,
                        "msisdn": None if use_email else _phone_variant(rng, a.phone),
                        "address": a.email if use_email else None,
                    },
                    "recipient": {
                        "selector_id": b_local,
                        "msisdn": None if use_email else _phone_variant(rng, b.phone),
                        "address": b.email if use_email else None,
                    },
                    "timestamp": event["date"].strftime("%Y-%m-%d"),
                }) + "\n")
                self._record_truth(f"comms:{a_local}", a.id)
                self._record_truth(f"comms:{b_local}", b.id)
                count += 1
        return count

    def _write_watchlist(self) -> int:
        """Watchlist: sparse, name-and-dob only, deliberately degraded."""
        rng = self.rng
        people = {p.id: p for p in self.corpus.people}
        path = self.out / "watchlist.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["listing_id", "subject_name", "date_of_birth",
                             "nationality", "category", "listed_on"])
            for index, person_id in enumerate(self.corpus.watchlist):
                person = people[person_id]
                local_id = f"WL{index:04d}"
                writer.writerow([
                    local_id,
                    _name_variant(rng, person, self.noise),
                    _date_variant(rng, person.dob, self.noise * 0.4) or "",
                    person.nationality,
                    rng.choice(("financial crime", "sanctions nexus", "adverse media")),
                    "2024-01-15",
                ])
                self._record_truth(f"watch:{local_id}", person.id)
        return len(self.corpus.watchlist)


def generate(out_dir: str | Path, seed: int = 20260813, n_people: int = 300,
             n_orgs: int = 80, noise: float = 0.28) -> dict:
    """Build ground truth and emit all source files. Fully deterministic."""
    corpus = CorpusBuilder(seed=seed, n_people=n_people, n_orgs=n_orgs, noise=noise).build()
    writer = CorpusWriter(corpus, Path(out_dir), seed=seed % 9973, noise=noise)
    stats = writer.write()

    meta = {
        "seed": seed,
        "people": len(corpus.people),
        "organizations": len(corpus.orgs),
        "accounts": len(corpus.accounts),
        "transactions": len(corpus.transactions),
        "communications": len(corpus.comms),
        "ring_people": corpus.ring_people,
        "ring_orgs": corpus.ring_orgs,
        "watchlist": corpus.watchlist,
        "source_records": stats,
    }
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta
