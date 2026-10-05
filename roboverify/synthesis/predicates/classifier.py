from synthesis.predicates.enumerate import enumerate_separator


def learn_classifier(positive, negative, scope, **kwargs):
    if not positive or not negative:
        from synthesis.predicates.enumerate import SearchResult

        return SearchResult("insufficient_examples")
    return enumerate_separator(
        [(s, True) for s in positive] + [(s, False) for s in negative],
        tuple(sorted(scope)),
        mode="classifier",
        **kwargs
    )


LearnClassifier = learn_classifier


def learn_ground_classifier(positive, negative, scope, *, language=None):
    """Learn a quantifier-free separator over the shared concrete ID universe.

    Existing aliases (notably b0 and tbl) can abbreviate a fixed ID, but no
    existential names are introduced. Differing universes are unsupported.
    """
    from dataclasses import replace

    from synthesis.predicates.enumerate import SearchResult
    from synthesis.predicates.language import Language
    from synthesis.predicates.term import block_id, ref

    if not positive or not negative:
        return SearchResult("insufficient_examples")
    scenes = [*positive, *negative]
    ids = {i for i in scenes[0].positions if type(i) is int}
    if any({i for i in s.positions if type(i) is int} != ids for s in scenes):
        raise ValueError(
            "ID-first refinement requires the same block IDs in every scene"
        )
    aliases = {}
    for name in sorted(scope):
        values = {s.bindings.get(name) for s in scenes}
        if len(values) == 1 and None not in values:
            aliases.setdefault(values.pop(), ref(name))
    constants = [aliases.get(i, block_id(i)) for i in sorted(ids)]
    if "tbl" in aliases:
        constants.append(aliases["tbl"])
    return enumerate_separator(
        [(s, True) for s in positive] + [(s, False) for s in negative],
        constants,
        mode="classifier",
        language=replace(language or Language(), max_variables=0),
    )
