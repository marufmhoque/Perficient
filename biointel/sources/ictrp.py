"""WHO ICTRP adapter - placeholder.

The public ICTRP web service is access-restricted, so v1 ships this stub. It shows where a new
registry adapter plugs in: implement ``search`` to return TrialRecords and register them in
the SourceRegistry; no downstream agent or aggregator code changes are needed.
"""

from __future__ import annotations

from ..schemas import TrialRecord


class ICTRPAdapter:
    name = "WHO ICTRP"
    available = False

    async def search(self, *args, **kwargs) -> tuple[list[TrialRecord], int]:
        return [], 0
