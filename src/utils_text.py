import hashlib
from typing import Optional


def stable_string_to_bucket_id(value: Optional[str], num_buckets: int) -> int:
    """
    Map a string to a stable integer ID in [0, num_buckets).

    Empty/None values map to 0.
    """
    if not value:
        return 0
    # Normalize to uppercase and strip spaces to reduce variants
    normalized = value.strip().upper()
    if normalized == "":
        return 0
    # Use sha256 for stable hashing across processes and runs
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    # Convert first 8 bytes to int for speed, then mod buckets
    int_val = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int_val % max(1, num_buckets)


