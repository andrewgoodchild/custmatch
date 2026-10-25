"""Contact profiling: junk vs household vs personal.

These guard the detector that reached 100% precision/recall on junk values, and the
first-name-diversity rule that separates a shared household contact from one person's
records duplicated across two systems.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/scripts"))
from custmatch.core import profile_contacts, syntactic_junk  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["first_name", "last_name", "addr_number",
                                       "addr_street", "email", "phone"])


@pytest.mark.parametrize("value", [
    "test@test.com", "asdf@asdf.com", "none@none.com", "noreply@example.com",
    "a@a.com", "0000000", "1234567890", "5551234567",
    # service relays, filed by mail clients under the name of the person they wrote for
    "drive-shares-noreply@google.com", "calendar-notification@google.com", "notifications@github.com",
    "no-reply@accounts.example.com", "do_not_reply@bank.example", "mailer-daemon@example.com",
])
def test_syntactic_junk_catches_common_placeholders(value):
    assert syntactic_junk(value)


@pytest.mark.parametrize("value", [
    "jane.smith42@mail.com", "thesmithfamily7@mail.com", "2065551234x@mail.com",
    "gertie.noreply@bigcorp.net", "nora.reply@mail.com", "knoreply@mail.com",   # BPID's people have the first
])
def test_syntactic_junk_does_not_fire_on_real_looking_values(value):
    assert not syntactic_junk(value)


def test_junk_value_shared_by_unrelated_people_is_flagged():
    rows = [(f"first{i}", f"surname{i}", str(i), f"{i} street", "shared@mail.com", "")
            for i in range(12)]
    prof = profile_contacts(frame(rows), "email")
    assert prof["shared@mail.com"][0] == "junk"
    assert prof["shared@mail.com"][1] == 0.0        # zero evidential weight


def test_large_family_at_one_address_is_household_not_junk():
    """The clause that fixed junk precision: a big household is diverse in surname but NOT
    in address. Without the address test this was classified junk."""
    rows = [(f"first{i}", f"surname{i}", "10", "elm st", "family@mail.com", "")
            for i in range(12)]
    prof = profile_contacts(frame(rows), "email")
    assert prof["family@mail.com"][0] == "household"


def test_two_people_sharing_a_contact_is_household():
    rows = [("john", "smith", "10", "elm st", "smiths@mail.com", ""),
            ("jane", "smith", "10", "elm st", "smiths@mail.com", "")]
    prof = profile_contacts(frame(rows), "email")
    assert prof["smiths@mail.com"][0] == "household"
    assert prof["smiths@mail.com"][1] < 1.0          # down-weighted, not trusted as identity


def test_one_persons_duplicate_records_stay_personal():
    """Same person in two source systems shares a first name - that is a duplicate, not a
    household, and must keep full evidential weight."""
    rows = [("john", "smith", "10", "elm st", "john.smith@mail.com", ""),
            ("john", "smith", "10", "elm st", "john.smith@mail.com", "")]
    prof = profile_contacts(frame(rows), "email")
    assert prof["john.smith@mail.com"][0] == "personal"


def test_unique_contact_keeps_full_weight():
    rows = [("john", "smith", "10", "elm st", "unique@mail.com", ""),
            ("jane", "jones", "11", "oak st", "other@mail.com", "")]
    prof = profile_contacts(frame(rows), "email")
    assert prof["unique@mail.com"] == ("personal", 1.0)


def test_aware_false_disables_all_classification():
    """The naive ablation must genuinely treat every value as a trusted personal key."""
    rows = [(f"first{i}", f"surname{i}", str(i), f"{i} st", "test@test.com", "")
            for i in range(12)]
    prof = profile_contacts(frame(rows), "email", aware=False)
    assert prof["test@test.com"] == ("personal", 1.0)
