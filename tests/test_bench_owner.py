"""USB setup must not race modeld or outlive the parked condition."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests import load_script

bench = load_script(Path(__file__).resolve().parents[1] / 'scripts/comma/jetlink_bench.py')


def test_waits_for_gadget_before_modeld(monkeypatch):
  monkeypatch.setattr(bench.time, 'sleep', Mock())
  configured = Mock(side_effect=[False, True])
  bench.wait_for_owner(Mock(poll=Mock(return_value=None)), Mock(get_bool=Mock(return_value=True)), lambda: False, configured)
  assert configured.call_count == 2


@pytest.mark.parametrize('offroad,stopped,exited', [(False, False, None), (True, True, None), (True, False, 1)])
def test_ready_gadget_does_not_override_stop(offroad, stopped, exited):
  with pytest.raises(RuntimeError):
    bench.wait_for_owner(Mock(poll=Mock(return_value=exited)), Mock(get_bool=Mock(return_value=offroad)),
                         lambda: stopped, lambda: True)


def test_setup_has_a_deadline():
  with pytest.raises(TimeoutError):
    bench.wait_for_owner(Mock(poll=Mock(return_value=None)), Mock(get_bool=Mock(return_value=True)),
                         lambda: False, lambda: False, timeout=0)
