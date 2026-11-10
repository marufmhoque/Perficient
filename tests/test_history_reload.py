"""Regression: Streamlit Cloud reloads changed modules WITHOUT restarting the process, while the
report library lives on in st.cache_resource. A library holding instances of the old classes then
rejected inputs built from the reloaded classes (pydantic ValidationError in add_search) and handed
old-class objects to new code. The store must survive a reload of the schema / history modules."""

from __future__ import annotations

import importlib


def _briefing(schemas, bid="r1"):
    return schemas.Briefing(
        id=bid, input=schemas.ConditionInput(condition_term="Atopic Dermatitis"),
        condition=schemas.ResolvedCondition(display_name="Atopic Dermatitis", trial_query_term="Atopic Dermatitis"),
    )


def test_history_survives_module_reload():
    import biointel.history as history_mod
    import biointel.schemas as schemas

    state = history_mod.new_state()  # what st.cache_resource keeps across reruns
    old = history_mod.InMemoryHistory(state=state)
    old.save_report(_briefing(schemas, "before-reload"))
    old.add_search(schemas.ConditionInput(condition_term="Plaque Psoriasis"))

    # --- code update on Streamlit Cloud: modules re-executed, classes re-created ---
    schemas = importlib.reload(schemas)
    history_mod = importlib.reload(history_mod)
    new = history_mod.InMemoryHistory(state=state)

    # the original crash: a search built from the reloaded ConditionInput
    new.add_search(schemas.ConditionInput(condition_term="Atopic Dermatitis"), report_id="r2")
    new.save_report(_briefing(schemas, "after-reload"))

    assert [s.label for s in new.recent_searches()] == ["Atopic Dermatitis", "Plaque Psoriasis"]
    assert {r.id for r in new.list_reports()} == {"before-reload", "after-reload"}
    got = new.get_report("before-reload")
    assert isinstance(got, schemas.Briefing)  # current class, so isinstance-based code keeps working
    assert isinstance(new.recent_searches()[0].input, schemas.ConditionInput)


def test_store_holds_only_plain_data():
    import biointel.history as history_mod
    import biointel.schemas as schemas
    from pydantic import BaseModel

    state = history_mod.new_state()
    h = history_mod.InMemoryHistory(state=state)
    h.save_report(_briefing(schemas))
    h.add_search(schemas.ConditionInput(condition_term="x"))

    def walk(o):
        assert not isinstance(o, BaseModel), f"model instance cached across reruns: {type(o).__name__}"
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk({k: v for k, v in state.items() if k != "lock"})
