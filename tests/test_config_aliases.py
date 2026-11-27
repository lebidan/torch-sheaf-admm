"""Explicit config names for the agent state and message sizes."""

from __future__ import annotations

import pytest

from sheaf_admm.models import model_config_from_dict


def test_explicit_names_map_to_internal_fields():
    cfg = model_config_from_dict({"agent_state_dim": 200, "message_dim": 8})
    assert (cfg.d_v, cfg.d_e) == (200, 8)


def test_internal_names_still_work():
    cfg = model_config_from_dict({"d_v": 160, "d_e": 16})
    assert (cfg.d_v, cfg.d_e) == (160, 16)


def test_alias_and_internal_name_together_are_refused():
    with pytest.raises(KeyError):
        model_config_from_dict({"d_v": 160, "agent_state_dim": 200})
