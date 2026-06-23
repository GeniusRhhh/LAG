"""Legacy intent API routed to the project-wide bvr_intent_new model."""

import logging

from intent_runtime_bridge import infer_project_intent, project_intent_to_legacy_text


LOG = logging.getLogger(__name__)


class ProjectIntentRecognizer:
    """Project-wide intent recognizer backed by bvr_intent_new."""

    _logged_backend = False

    def __init__(self):
        if not self.__class__._logged_backend:
            LOG.info("ProjectIntentRecognizer: using bvr_intent_new")
            self.__class__._logged_backend = True

    def recognize(self, my_state: dict, enemy_state: dict) -> str:
        intent, _meta = infer_project_intent(my_state, enemy_state)
        return project_intent_to_legacy_text(intent)
