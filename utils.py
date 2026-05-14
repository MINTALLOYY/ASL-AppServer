from typing import Optional


def validate_user_identifier(value: str) -> bool:
    """
    Return True when value is a non-empty opaque account-scoped identifier.

    Firebase UID values are expected to work here, so this intentionally does
    not enforce UUID formatting.
    """
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    return len(candidate) <= 256


def normalize_user_identifier(value: str) -> Optional[str]:
    """
    Return a normalized account-scoped identifier if one is present.
    """
    if not validate_user_identifier(value):
        return None
    return value.strip()


def _get_accessor_user_id(request) -> Optional[str]:
    """
    Extract the account-scoped user identifier from headers, query params,
    JSON bodies, or form data.
    """
    if request is None:
        return None

    for header_name in ("X-User-Id", "X-Firebase-Uid", "X-Conversation-UUID", "X-Accessor-UUID"):
        value = request.headers.get(header_name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("user_id", "firebase_uid", "uid", "conversation_uuid", "accessor_uuid"):
        value = request.args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    try:
        json_data = request.get_json(silent=True) or {}
    except Exception:
        json_data = {}
    if isinstance(json_data, dict):
        for key in ("user_id", "firebase_uid", "uid", "conversation_uuid", "accessor_uuid"):
            value = json_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    form = getattr(request, "form", None)
    if form is not None:
        for key in ("user_id", "firebase_uid", "uid", "conversation_uuid", "accessor_uuid"):
            value = form.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


# Backwards-compatible aliases for the previous UUID-centric naming.
validate_uuid = validate_user_identifier
normalize_uuid = normalize_user_identifier
_get_accessor_uuid = _get_accessor_user_id
