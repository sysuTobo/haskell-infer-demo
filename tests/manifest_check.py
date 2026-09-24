"""Canonical-encoding verification for the execution manifest (stdlib only).

The C emitter is not trusted to report its own digests correctly: this module
re-derives every identity from the *parsed* document, using the one canonical
rule the contract fixes (keys sorted by byte value, no whitespace, integers
bare, non-integer constants as strings, printable ASCII strings, so no escaping
convention can differ between writers).

Used by tests/test_manifest_hashes.py (the CPU ctest gate) and by
tests/capture_logits.py, which refuses to record a capture whose embedded
manifest does not verify.
"""

import hashlib
import json

# (identity block, key holding that block's digest)
IDENTITY_BLOCKS = (
    ("semantic", "semantic_id"),
    ("numerical_policy", "numerical_policy_id"),
    ("deployment", "deployment_id"),
)

# The plan's Stage 0 gate also requires that comparisons refuse a document whose
# execution provenance is not established. These spellings mean exactly that and
# must never be treated as compatible values.
UNESTABLISHED = ("unknown", "unavailable", "unsupported", "unspecified")


def canonical(value) -> str:
    """The canonical text of a parsed JSON value."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def verify(text: str) -> list:
    """Return the problems found; an empty list means the manifest verifies."""
    problems = []
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        return [f"the manifest is not valid JSON: {exc}"]
    if not isinstance(parsed, dict):
        return ["the manifest is not a JSON object"]
    if canonical(parsed) != text:
        problems.append("the document is not canonical (sorted keys, no whitespace, "
                        "integers bare, other constants as strings)")
    # The version gate comes first: a document without it is a legacy capture, and
    # interpreting its blocks as identity-bearing would be inventing an identity.
    if parsed.get("manifest_version") != 1:
        problems.append(f"manifest_version is {parsed.get('manifest_version')}, expected 1")

    for block_name, id_key in IDENTITY_BLOCKS:
        block = parsed.get(block_name)
        if not isinstance(block, dict) or not isinstance(block.get("fields"), dict):
            problems.append(f"{block_name}: missing fields")
            continue
        want = block.get(id_key)
        got = digest(block["fields"])
        if want != got:
            problems.append(f"{block_name}: {id_key} is {want} but its fields hash to {got}")

    regions = parsed.get("regions")
    numerical = parsed.get("numerical_policy", {})
    numerical_fields = numerical.get("fields", {}) if isinstance(numerical, dict) else {}
    if not isinstance(regions, list):
        problems.append("regions: missing or not an array")
    elif numerical_fields.get("regions_sha256") != digest(regions):
        problems.append("numerical_policy.fields.regions_sha256 does not match the regions array")

    # Types the contract fixes. A consumer parses these as a boolean or an integer,
    # so emitting a number that looks like a boolean (or a string that looks like a
    # number) is a contract violation even when the value reads correctly.
    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool)

    for index, entry in enumerate(regions or []):
        if not isinstance(entry.get("rng_dependency"), bool):
            problems.append(f"regions[{index}].rng_dependency is not a JSON boolean")
        for key in ("region", "cases", "implementation", "determinism", "mechanism"):
            if not isinstance(entry.get(key), str):
                problems.append(f"regions[{index}].{key} is not a string")
    descriptor = parsed.get("descriptor", {})
    if not integer(descriptor.get("bytes")):
        problems.append("descriptor.bytes is not an integer")
    if not integer(descriptor.get("desc_version")):
        problems.append("descriptor.desc_version is not an integer")
    weights = parsed.get("weights", {})
    if not integer(weights.get("tensor_count")):
        problems.append("weights.tensor_count is not an integer")
    sampling = parsed.get("sampling", {})
    if not integer(sampling.get("transform_version")):
        problems.append("sampling.transform_version is not an integer")

    return problems


def identities(text: str) -> dict:
    """The identity values a comparison keys on."""
    parsed = json.loads(text)
    out = {id_key: parsed[block][id_key] for block, id_key in IDENTITY_BLOCKS}
    weights = parsed.get("weights") or {}
    out["parameter_manifest_sha256"] = weights.get("parameter_manifest_sha256")
    return out


def unestablished_paths(text: str) -> list:
    """Paths whose value is one of the UNESTABLISHED spellings.

    A strict comparison must refuse these: an unknown numerical setting is not a
    default, it is a fact the comparison cannot establish.
    """
    parsed = json.loads(text)
    paths = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + [key])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + [str(index)])
        elif isinstance(node, str) and node in UNESTABLISHED:
            paths.append(".".join(path))

    walk(parsed.get("provenance", {}), ["provenance"])
    # The generation policy is a numerical-policy field too: a manifest that does
    # not say which sampler arithmetic ran cannot support an exact claim.
    walk(parsed.get("sampling", {}), ["sampling"])
    return paths


def flatten(value, prefix=""):
    """Flatten a JSON value into dotted leaf paths, so a difference reads as a
    field name rather than as two whole documents."""
    if isinstance(value, dict):
        if not value:
            return {prefix or "<root>": "{}"}
        out = {}
        for key, child in value.items():
            out.update(flatten(child, prefix + "." + key if prefix else key))
        return out
    if isinstance(value, list):
        if not value:
            return {prefix: "[]"}
        out = {}
        for index, child in enumerate(value):
            out.update(flatten(child, prefix + "." + str(index)))
        return out
    if value is None:
        return {prefix: "null"}
    if isinstance(value, bool):
        return {prefix: "true" if value else "false"}
    return {prefix: str(value)}


def provenance_differences(left_text: str, right_text: str) -> list:
    """The provenance paths that differ, as (path, left, right)."""
    left = flatten(json.loads(left_text).get("provenance", {}))
    right = flatten(json.loads(right_text).get("provenance", {}))
    return [(key, left.get(key), right.get(key))
            for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key)]
