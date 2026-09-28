"""Frozen judge instructions used by the factorized evaluator.

API clients are intentionally not bundled. They should pass these prompts to
the judge versions specified in the paper and enforce strict schema parsing.
"""

REFERENCE_FIDELITY_PROMPT = """Given two images:
- Image 1 is a source image containing a specific element.
- Image 2 is an AI-generated composite image.

Does the element from Image 1 appear in Image 2?
Answer only \"yes\" or \"no\"."""

ATOMIC_VERIFIER_PROMPT = """You are a strict visual verifier.
Decide whether ONE atomic visual requirement is satisfied by the candidate image.
Answer YES only when the requirement is clearly satisfied. Answer NO if required
content is missing, the wrong reference is used, a role binding, action, or
spatial relation is wrong, or evidence is insufficient. Do not judge aesthetics,
general image quality, or unrelated requirements. Return exactly one JSON object:
{\"answer\":\"YES\"} or {\"answer\":\"NO\"}."""

PROMPT_FOLLOWING_SCHEMA = {
    "entities_counts": "number or null in [0,10]",
    "attributes_roles": "number or null in [0,10]",
    "actions_interactions": "number or null in [0,10]",
    "spatial_relations": "number or null in [0,10]",
    "scene_composition": "number or null in [0,10]",
}

PERCEPTUAL_QUALITY_PROMPT = """Evaluate only the candidate image's perceptual
quality without considering an instruction or reference image. Score naturalness
and artifact freedom from 0 to 10. Return exactly one JSON object with numeric
keys \"naturalness\" and \"artifacts\"."""

