"""Queries select output channels; scene geometry always remains intact."""
from __future__ import annotations
from dataclasses import dataclass
from reprlib import repr as short_repr


def _ids(value, role):
    """Normalize optional selections into unique, stable surface ID tuples."""
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError(f"{role} must be a sequence of surface IDs")
    value = tuple(value)
    if any(not isinstance(item, str) or not item for item in value) or len(set(value)) != len(value):
        raise ValueError(f"{role} must contain unique nonempty surface IDs")
    return value


@dataclass(frozen=True)
class Query:
    """Select sender rows and scene/sky outputs while retaining every occluder.

    ``None`` sender or receiver selections mean all scene surfaces. IDs are
    stable Scene identifiers; display labels never select geometry.
    """
    senders: tuple[str, ...] | None = None
    receivers: tuple[str, ...] | None = None
    scene: bool = True
    sky_mode: str | None = None
    receiver_sides: tuple[str, ...] = ("front", "back")

    def __post_init__(self):
        """Validate selections, receiver sides and the requested output kinds."""
        object.__setattr__(self, "senders", _ids(self.senders, "senders"))
        object.__setattr__(self, "receivers", _ids(self.receivers, "receivers"))
        sides = tuple(self.receiver_sides)
        if not sides or len(set(sides)) != len(sides) or any(s not in ("front", "back") for s in sides):
            raise ValueError("receiver_sides must select front and/or back")
        object.__setattr__(self, "receiver_sides", sides)
        if not isinstance(self.scene, bool):
            raise TypeError("scene must be a bool")
        if self.sky_mode not in (None, "merged", "tregenza145"):
            raise ValueError("sky must be merged or tregenza145")
        if not self.scene and self.sky_mode is None:
            raise ValueError("A query must request scene or sky outputs")

    def __repr__(self):
        """Describe selected outputs using bounded sender and receiver lists."""
        senders = "all" if self.senders is None else short_repr(self.senders)
        receivers = "all" if self.receivers is None else short_repr(self.receivers)
        return (f"Query(senders={senders}, receivers={receivers}, scene={self.scene}, "
                f"sky={self.sky_mode!r}, sides={self.receiver_sides!r})")

    @classmethod
    def matrix(cls, senders=None, receivers=None, *, sky=None, receiver_sides=("front", "back")):
        """Request selected scene rows, optionally sharing their sky tracing."""
        return cls(senders, receivers, True, sky, receiver_sides)

    @classmethod
    def row(cls, sender, receivers=None, *, sky=None, receiver_sides=("front", "back")):
        """Request one sender row without removing unselected scene occluders."""
        return cls.matrix((sender,), receivers, sky=sky, receiver_sides=receiver_sides)

    @classmethod
    def pair(cls, sender, receiver, *, receiver_sides=("front", "back")):
        """Request one scene pair, including a same-ID self pair, retaining occluders."""
        return cls.matrix((sender,), (receiver,), receiver_sides=receiver_sides)

    @classmethod
    def sky(cls, senders=None, *, discrete=False):
        """Request merged sky or 145 Tregenza patches for selected senders."""
        return cls(senders, None, False, "tregenza145" if discrete else "merged")

    def resolve(self, surface_ids):
        """Resolve all-surface selections and reject IDs absent from the scene."""
        ids = tuple(surface_ids)
        senders = ids if self.senders is None else self.senders
        receivers = ids if self.receivers is None else self.receivers
        unknown = (set(senders) | (set(receivers) if self.scene else set())) - set(ids)
        if unknown:
            raise ValueError(f"Unknown surface IDs: {sorted(unknown)}")
        return senders, receivers
