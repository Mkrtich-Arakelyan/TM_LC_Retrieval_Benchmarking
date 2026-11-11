"""Local routing double for standalone safety checks."""

import json
from uuid import uuid4
from workflow.graph import Integrations
from fixtures.offline import offline_integrations


class TestRouter:
    """Explicit test double. Not used by the live router or fixture production path."""
    def __init__(self, invalid=False):
        self.invalid = invalid
        self.calls = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        context = json.loads(messages[1][1])
        self.calls.append(context)
        if self.invalid:
            return {"action_id": "acquire", "reason": "TEST DOUBLE invalid direct execution"}
        actions = context["available_actions"]
        comparisons = [a for a in actions if a["kind"] == "run_operation"
                       and a["request"]["operation"] == "tm.compare_reference_free"]
        selected = comparisons[0] if comparisons else None
        if selected is None:
            if not context["datasets"] and not context["acquisition_capability"]["available"]:
                order = ["request_information"]
            else:
                order = ["conclude", "review_results", "interpret", "preflight", "prepare_procedure", "advise", "propose", "request_information"]
            selected = next(a for kind in order for a in actions if a["kind"] == kind)
        return {"action_id": selected["id"], "record_ids": [],
                "reason": "TEST DOUBLE: " + ("Missing compatible saved data and an available acquisition backend" if selected["kind"] == "request_information" else selected["description"])}


def double_integrations(config, *, invalid=False):
    services = offline_integrations(config, uuid4(), include_analyst=True)
    return Integrations(router_model=TestRouter(invalid), **services)


