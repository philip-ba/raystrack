"""Grasshopper's out-of-process, asynchronous Raystrack v2 integration.

The numerical core has no Rhino dependency. The compiled Grasshopper plug-in
communicates with ``python -m raystrack.integrations.grasshopper.worker`` using
the versioned JSON-lines protocol documented in ``worker``.
"""

