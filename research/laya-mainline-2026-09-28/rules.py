"""Executable English rule families for a controlled, synthetic diagnostic."""
from __future__ import annotations

import json
import random

TRAIN_FAMILIES = ("threshold", "membership", "conjunction", "bands")
DEV_FAMILIES = ("disjunction", "veto")
TEST_FAMILIES = ("exclusive_or", "weighted_bands")
CATEGORIES = ("amber", "birch", "cedar", "dahlia", "elm", "fir")


def sample_state(rng):
    return {"amount": rng.randint(0, 99), "age": rng.randint(0, 99),
            "category": rng.choice(CATEGORIES), "flag": bool(rng.randrange(2)),
            "reference": f"case-{rng.getrandbits(48):012x}"}


def sample_policy(family, rng):
    low, high = sorted(rng.sample(range(10, 90), 2))
    return {"cutoff": rng.randint(10, 89), "age_limit": rng.randint(10, 89),
            "allowed": rng.sample(list(CATEGORIES), 3), "low": low, "high": high,
            "weight": rng.choice((1, 2, 3))}


def oracle(family, state, policy):
    amount, age = state["amount"], state["age"]
    a, b = amount >= policy["cutoff"], age <= policy["age_limit"]
    if family == "threshold":
        return int(a)
    if family == "membership":
        return int(state["category"] in policy["allowed"])
    if family == "conjunction":
        return int(a and b)
    if family == "disjunction":
        return int(a or b)
    if family == "veto":
        return int(a and not state["flag"])
    if family == "exclusive_or":
        return int(a != b)
    if family in ("bands", "weighted_bands"):
        value = amount if family == "bands" else policy["weight"] * amount + age
        return int(value >= policy["low"]) + int(value >= policy["high"])
    raise ValueError(f"unknown rule family: {family}")


def predicate(family, policy):
    a = f"amount is at least {policy['cutoff']}"
    b = f"age is at most {policy['age_limit']}"
    if family == "threshold":
        return a
    if family == "membership":
        return "category is one of " + ", ".join(policy["allowed"])
    if family == "conjunction":
        return f"both conditions hold: {a}; {b}"
    if family == "disjunction":
        return f"at least one condition holds: {a}; {b}"
    if family == "veto":
        return f"{a} and flag is false; flag=true always forbids acceptance"
    if family == "exclusive_or":
        return f"exactly one condition holds, not both: {a}; {b}"
    raise ValueError(f"no Boolean predicate for {family}")


def question(family, policy, use_choice, reverse_labels):
    if family in ("bands", "weighted_bands"):
        expr = "amount" if family == "bands" else f"({policy['weight']} times amount plus age)"
        low, high = policy["low"], policy["high"]
        return {"type": "score", "instructions": "Assign the level defined by the supplied criteria.",
                "criteria": [f"{expr} is less than {low}",
                             f"{expr} is at least {low} and less than {high}",
                             f"{expr} is at least {high}"]}, [0, 1, 2]
    condition = predicate(family, policy)
    descriptions = [f"The following acceptance condition does NOT hold: {condition}.",
                    f"The following acceptance condition holds: {condition}."]
    if use_choice:
        mapping = [1, 0] if reverse_labels else [0, 1]
        criteria = {key: descriptions[value] for key, value in zip(("A", "B"), mapping)}
        return {"type": "choice", "instructions": "Select the candidate matching this case.",
                "criteria": criteria}, mapping
    return {"type": "noul", "instructions": "Does this case satisfy the acceptance condition?",
            "criteria": dict(zip(("false", "true"), descriptions))}, [0, 1]


def make_group(family, split, index, rng):
    k = 3 if family in ("bands", "weighted_bands") else 2
    desired = index % k
    for _ in range(20000):
        state, policy = sample_state(rng), sample_policy(family, rng)
        state.update(amount=rng.randint(20, 79), age=rng.randint(20, 79))
        if family == "veto":
            state["flag"] = False  # both acceptance and rejection must be reachable by policy.
        if family == "weighted_bands":
            state.update(amount=rng.randint(5, 35), age=rng.randint(5, 35))
        if oracle(family, state, policy) == desired:
            break
    else:
        raise RuntimeError(f"cannot sample base: {family}")
    for _ in range(20000):
        changed_policy = sample_policy(family, rng)
        if oracle(family, state, changed_policy) != desired:
            break
    else:
        raise RuntimeError(f"cannot sample policy contrast: {family}")
    for _ in range(20000):
        changed_state = sample_state(rng)
        changed_state["reference"] = state["reference"]
        if oracle(family, changed_state, policy) != desired:
            break
    else:
        raise RuntimeError(f"cannot sample fact contrast: {family}")
    # Fact interventions alter relevant values only; metadata/category distractors stay fixed.
    keys = {"threshold": ("amount",), "membership": ("category",),
            "conjunction": ("amount", "age"), "disjunction": ("amount", "age"),
            "veto": ("amount", "flag"), "exclusive_or": ("amount", "age"),
            "bands": ("amount",), "weighted_bands": ("amount", "age")}[family]
    changed_state = {**state, **{key: changed_state[key] for key in keys}}
    irrelevant = {**state, "reference": f"case-{rng.getrandbits(48):012x}"}
    group_id = f"{split}/{family}/{index:04d}"
    use_choice, reverse = bool(rng.randrange(2)), bool(rng.randrange(2))
    variants = (("base", state, policy), ("policy", state, changed_policy),
                ("fact", changed_state, policy), ("irrelevant", irrelevant, policy))
    records = []
    for variant, facts, rule in variants:
        q, mapping = question(family, rule, use_choice, reverse)
        gold = oracle(family, facts, rule)
        label = mapping.index(gold)
        records.append({"id": f"{group_id}/{variant}", "group_id": group_id,
                        "family": family, "split": split, "variant": variant,
                        "state": facts, "question": q, "rule": rule,
                        "semantic_values": mapping, "label_index": label,
                        "target": [float(i == label) for i in range(k)]})
    values = [r["semantic_values"][r["label_index"]] for r in records]
    assert values[0] != values[1] and values[0] != values[2] and values[0] == values[3]
    return records


def semantic_state(record):
    return json.dumps({k: v for k, v in record["state"].items() if k != "reference"}, sort_keys=True)
