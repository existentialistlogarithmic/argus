"""Source adapters.

One adapter per source system. Each knows that system's peculiar schema and
nothing else: how its columns map onto ontology properties, and which
relationships it asserts. Adapters emit ``Record`` and ``LinkRecord`` objects
and make no identity decisions whatsoever -- that is the resolver's job, and
keeping the two apart is what lets either be changed without the other.

Links are emitted against *source-local* keys. At ingest time there is no such
thing as an entity yet.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .model import LinkRecord, Record
from .ontology import Ontology


def read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class Ingestor:
    """Runs every adapter over a corpus directory."""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self.records: dict[str, Record] = {}
        self.links: list[LinkRecord] = []

    # -- emission helpers --------------------------------------------------

    def add(self, source: str, local_id: str, entity_type: str, raw: dict) -> str:
        """Emit a record, merging into any existing record with the same key.

        Sources repeat themselves -- an employer appears on every staff row --
        so a key collision means "same assertion", not "conflicting object".
        """
        key = f"{source}:{local_id}"
        clean = {k: v for k, v in raw.items() if v not in (None, "", [])}
        etype = self.ontology.entity(entity_type)
        existing = self.records.get(key)
        if existing is not None:
            existing.raw.update(clean)
            existing.props = etype.normalize_props(existing.raw)
            return key
        self.records[key] = Record(
            source=source,
            local_id=local_id,
            entity_type=entity_type,
            raw=clean,
            props=etype.normalize_props(clean),
        )
        return key

    def link(self, source: str, local_id: str, link_type: str,
             src_key: str, dst_key: str, props: dict | None = None):
        if src_key == dst_key:
            return  # self-links carry no information and break path search
        self.links.append(LinkRecord(
            source=source,
            local_id=local_id,
            link_type=link_type,
            source_key=src_key,
            target_key=dst_key,
            props={k: v for k, v in (props or {}).items() if v not in (None, "")},
        ))

    # -- adapters ----------------------------------------------------------

    def ingest_natreg(self, path: Path):
        """National population registry (CSV)."""
        for row in read_csv(path):
            local = row["record_id"]
            person_key = self.add("natreg", local, "person", {
                "full_name": f"{row['given_names']} {row['surname']}".strip(),
                "dob": row["date_of_birth"],
                "national_id": row["national_id"],
                "address": row["residential_address"],
                "nationality": row["country"],
            })
            if row.get("residential_address"):
                location_key = self.add("natreg", f"LOC{local}", "location", {
                    "address": row["residential_address"],
                    "city": row.get("city"),
                    "country": row.get("country"),
                })
                self.link("natreg", f"RES{local}", "resides_at", person_key, location_key)

    def ingest_hr(self, path: Path):
        """Corporate HR directory (CSV): informal names, contact details."""
        for row in read_csv(path):
            staff = row["staff_id"]
            person_key = self.add("hr", staff, "person", {
                "full_name": row["display_name"],
                "dob": row.get("date_of_birth"),
                "email": row.get("work_email"),
                "phone": row.get("mobile"),
                "occupation": row.get("job_title"),
            })
            org_key = self.add("hr", row["employer_id"], "organization", {
                "legal_name": row["employer_name"],
                "country": row.get("employer_country"),
            })
            self.link("hr", f"EMP{staff}", "employed_by", person_key, org_key, {
                "role": row.get("job_title"),
                "since": row.get("start_date"),
            })

    def ingest_bank(self, path: Path):
        """Bank KYC file (JSONL): account holders and their accounts."""
        for row in read_jsonl(path):
            kyc = row["kyc_id"]
            account_key = self.add("bank", f"{kyc}A", "account", {
                "account_number": row.get("account_no"),
                "iban": row.get("iban"),
                "institution": row.get("bank"),
                "currency": row.get("ccy"),
                "opened": row.get("opened_on"),
            })
            if row.get("holder_type") == "individual":
                holder_key = self.add("bank", f"{kyc}H", "person", {
                    "full_name": row.get("holder_name"),
                    "dob": row.get("holder_dob"),
                    "address": row.get("holder_address"),
                    "phone": row.get("holder_phone"),
                    "email": row.get("holder_email"),
                })
                self.link("bank", f"OWN{kyc}", "owns_account", holder_key, account_key)
            else:
                holder_key = self.add("bank", f"{kyc}H", "organization", {
                    "legal_name": row.get("holder_name"),
                    "reg_number": row.get("holder_reg"),
                    "address": row.get("holder_address"),
                    "country": row.get("holder_country"),
                })
                self.link("bank", f"CTL{kyc}", "controls_account", holder_key, account_key)

    def ingest_corpreg(self, path: Path):
        """Company registry (JSONL): companies, officers, shareholdings."""
        by_registration: dict[str, str] = {}
        shareholdings = []

        for row in read_jsonl(path):
            if row.get("entry_kind") == "shareholding":
                shareholdings.append(row)
                continue

            entry = row["entry_id"]
            office = row.get("registered_office") or {}
            org_key = self.add("corpreg", entry, "organization", {
                "legal_name": row["company_name"],
                "reg_number": row.get("registration_number"),
                "country": row.get("jurisdiction"),
                "incorporated": row.get("incorporated_on"),
                "sector": row.get("sic_description"),
                "address": office.get("street"),
            })
            if row.get("registration_number"):
                by_registration[row["registration_number"]] = org_key

            if office.get("street"):
                location_key = self.add("corpreg", f"LOC{entry}", "location", {
                    "address": office.get("street"),
                    "city": office.get("city"),
                    "country": office.get("country"),
                    "postcode": office.get("postcode"),
                })
                self.link("corpreg", f"REG{entry}", "registered_at", org_key, location_key)

            for officer in row.get("officers", []):
                officer_key = self.add("corpreg", officer["officer_id"], "person", {
                    "full_name": officer.get("name"),
                    "dob": officer.get("dob"),
                    "nationality": officer.get("nationality"),
                    # The registry's "role" is a corporate office, not an
                    # occupation. It belongs on the officer_of link, and
                    # mapping it here would manufacture false disagreement
                    # against an HR job title for the same person.
                })
                self.link("corpreg", f"OFF{officer['officer_id']}", "officer_of",
                          officer_key, org_key,
                          {"role": officer.get("role"), "since": officer.get("appointed")})

        for index, row in enumerate(shareholdings):
            parent = by_registration.get(row["parent_reg"])
            child = by_registration.get(row["child_reg"])
            if parent and child:
                self.link("corpreg", f"SH{index:05d}", "owns_stake", parent, child,
                          {"percent": row.get("percent")})

    def ingest_transactions(self, path: Path):
        """Payment rail extract (JSONL): account numbers, no identities.

        Every account number seen becomes a record with exactly one property.
        Those thin records are the seam: resolution merges them into the bank's
        fully-described accounts, and only then does a transaction connect to a
        human being.
        """
        for row in read_jsonl(path):
            debit = str(row["debit_account"])
            credit = str(row["credit_account"])
            debit_key = self.add("txn", f"ACC{debit}", "account", {"account_number": debit})
            credit_key = self.add("txn", f"ACC{credit}", "account", {"account_number": credit})
            self.link("txn", row["txn_id"], "transacted", debit_key, credit_key, {
                "amount": row.get("amount"),
                "currency": row.get("ccy"),
                "date": row.get("value_date"),
                "reference": row.get("narrative"),
            })

    def ingest_comms(self, path: Path):
        """Communications metadata (JSONL): selectors only, never names."""
        for row in read_jsonl(path):
            endpoints = []
            for side in ("originator", "recipient"):
                party = row.get(side) or {}
                props = {}
                if party.get("msisdn"):
                    props["phone"] = party["msisdn"]
                if party.get("address"):
                    props["email"] = party["address"]
                if not props:
                    endpoints.append(None)
                    continue
                endpoints.append(self.add("comms", party["selector_id"], "person", props))
            if all(endpoints):
                self.link("comms", row["event_id"], "communicated", endpoints[0], endpoints[1], {
                    "channel": row.get("channel"),
                    "date": row.get("timestamp"),
                })

    def ingest_watchlist(self, path: Path):
        """Watchlist (CSV): name and date of birth, nothing else."""
        for row in read_csv(path):
            self.add("watch", row["listing_id"], "person", {
                "full_name": row["subject_name"],
                "dob": row.get("date_of_birth"),
                "nationality": row.get("nationality"),
            })

    # -- driver ------------------------------------------------------------

    SOURCES = (
        ("natreg_persons.csv", "ingest_natreg"),
        ("hr_directory.csv", "ingest_hr"),
        ("bank_kyc.jsonl", "ingest_bank"),
        ("company_registry.jsonl", "ingest_corpreg"),
        ("transactions.jsonl", "ingest_transactions"),
        ("comms_metadata.jsonl", "ingest_comms"),
        ("watchlist.csv", "ingest_watchlist"),
    )

    def ingest_dir(self, directory: str | Path) -> dict:
        directory = Path(directory)
        report = {}
        for filename, method in self.SOURCES:
            path = directory / filename
            if not path.exists():
                continue
            before_records, before_links = len(self.records), len(self.links)
            getattr(self, method)(path)
            report[filename] = {
                "records": len(self.records) - before_records,
                "links": len(self.links) - before_links,
            }
        return report

    def by_type(self) -> dict[str, list[Record]]:
        grouped: dict[str, list[Record]] = {}
        for record in self.records.values():
            grouped.setdefault(record.entity_type, []).append(record)
        return grouped

    def attach_truth(self, truth_path: str | Path) -> int:
        """Load ground truth for evaluation. Never consulted by the resolver."""
        path = Path(truth_path)
        if not path.exists():
            return 0
        truth = json.loads(path.read_text(encoding="utf-8")).get("records", {})
        attached = 0
        for key, truth_id in truth.items():
            record = self.records.get(key)
            if record is not None:
                record.truth_id = truth_id
                attached += 1
        return attached
