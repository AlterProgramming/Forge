"""Compatibility surface for the v11 texture generator.

The pre-repository v6 snapshot exposed shared GLB/scalp helpers from this module.
v11 moved those reusable helpers to `tools.groom_core`; re-export them here so
recovered code and old commands do not silently break.
"""

from .groom_core import *  # noqa: F401,F403
