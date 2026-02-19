import json
import os
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PROFILE_PATH = PROJECT_ROOT / "scripts" / "utils" / "schema" / "schema_identity.json"
LEGACY_SCHEMA_PROFILE_PATH = PROJECT_ROOT / "outputs" / "00_las_info" / "schema_identity.json"


def is_iso_date(text):
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    try:
        date.fromisoformat(t)
        return True
    except ValueError:
        return False


def load_schema_profile():
    if not SCHEMA_PROFILE_PATH.exists() and LEGACY_SCHEMA_PROFILE_PATH.exists():
        try:
            SCHEMA_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SCHEMA_PROFILE_PATH.write_text(
                LEGACY_SCHEMA_PROFILE_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except Exception:
            pass

    if not SCHEMA_PROFILE_PATH.exists():
        return None
    try:
        data = json.loads(SCHEMA_PROFILE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def save_schema_profile(profile):
    SCHEMA_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def prompt_and_save_schema_profile(existing=None):
    current = existing or {}
    current_name = current.get("contactName", "Unknown")
    current_email = current.get("emailAddress", "unknown@example.com")

    print("\n=== SCHEMA PROFILE SETUP ===")
    print(
        "Schema validation may require metadata fields "
        "(pointOfContact, referenceDate)."
    )
    print("This profile fills missing metadata consistently across runs.")
    print("referenceDate is auto-set to today's date each run.")

    name = input(f"Contact name [{current_name}]: ").strip() or current_name
    email = input(f"Contact email [{current_email}]: ").strip() or current_email

    profile = {
        "contactName": name,
        "emailAddress": email,
    }
    save_schema_profile(profile)
    print(f"[INFO] Saved schema identity profile: {SCHEMA_PROFILE_PATH}")
    return profile


def ensure_profile_for_validation_interactive():
    profile = load_schema_profile()
    if profile:
        print("\n=== SCHEMA PROFILE ===")
        print(
            "Schema validation uses this profile to auto-fill missing "
            "required metadata fields."
        )
        print(f"Contact name:  {profile.get('contactName', 'Unknown')}")
        print(f"Contact email: {profile.get('emailAddress', 'unknown@example.com')}")
        print(f"Reference date:{date.today().isoformat()} (auto-set each run)")
        use_saved = input("Use this saved profile? [Y/n]: ").strip().lower()
        if use_saved in {"", "y", "yes"}:
            return profile
        return prompt_and_save_schema_profile(existing=profile)

    create_now = input(
        "\nNo schema profile found. Create one now for option [3]? [Y/n]: "
    ).strip().lower()
    if create_now in {"n", "no"}:
        fallback = {
            "contactName": "Unknown",
            "emailAddress": "unknown@example.com",
            "referenceDate": date.today().isoformat(),
        }
        save_schema_profile(fallback)
        print("[INFO] Saved fallback schema profile.")
        print(f"[INFO] You can edit this later: {SCHEMA_PROFILE_PATH}")
        return fallback
    return prompt_and_save_schema_profile(existing=None)


def resolve_schema_defaults():
    profile = load_schema_profile() or {}
    default_name = os.environ.get("OMNI_CONTACT_NAME") or profile.get("contactName") or "Unknown"
    default_email = os.environ.get("OMNI_CONTACT_EMAIL") or profile.get("emailAddress") or "unknown@example.com"
    default_date = (os.environ.get("OMNI_REFERENCE_DATE") or "").strip()
    if not is_iso_date(default_date):
        default_date = date.today().isoformat()
    return {
        "contactName": default_name,
        "emailAddress": default_email,
        "referenceDate": default_date,
    }
