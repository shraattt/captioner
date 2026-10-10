"""A tiny registry for swappable backends.

Two kinds are registered in this package: ``asr`` (speech to timed words)
and ``mt`` (subtitle translation). A backend is a class or factory with no
required constructor arguments; the registry only stores how to build it and
how to describe it, so adding a backend never touches the call sites.

Selection order for every lookup is: explicit name, then the environment
variable for that kind (``CAPTIONER_ASR_ENGINE`` / ``CAPTIONER_MT_ENGINE``),
then the kind's default. Unknown names raise ``UnknownEngine`` rather than
silently falling back: running a different model than the user asked for is
worse than saying the name is wrong.
"""
import os
from collections.abc import Callable
from typing import Any

# The kinds this package knows about.
ASR = "asr"
MT = "mt"
KINDS = (ASR, MT)

ENV_VARS = {
    ASR: "CAPTIONER_ASR_ENGINE",
    MT: "CAPTIONER_MT_ENGINE",
}


class EngineError(RuntimeError):
    """A backend could not be built, or a request to it failed."""


class UnknownEngine(EngineError):
    """No backend registered under that name."""

    def __init__(self, kind: str, name: str, known):
        self.kind = kind
        self.name = name
        self.known = sorted(known)
        options = ", ".join(self.known) if self.known else "none registered"
        super().__init__(f"unknown {kind} engine {name!r}; available: {options}")


class _Entry:
    __slots__ = ("factory", "describe", "default")

    def __init__(self, factory: Callable[[], Any], describe: str, default: bool):
        self.factory = factory
        self.describe = describe
        self.default = default


class Registry:
    """Name -> factory map for one kind of backend."""

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, _Entry] = {}
        self._default: str = ""

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        describe: str = "",
        default: bool = False,
    ) -> None:
        """Add (or replace) a backend. Later registrations win."""
        self._entries[name] = _Entry(factory, describe or name, default)
        if default or not self._default:
            self._default = name

    def names(self) -> list[str]:
        """Registered names, default first, then alphabetical."""
        rest = sorted(name for name in self._entries if name != self._default)
        return ([self._default] if self._default else []) + rest

    def describe(self) -> dict[str, str]:
        """``{name: one-line description}``, in ``names()`` order."""
        return {name: self._entries[name].describe for name in self.names()}

    def default_name(self) -> str:
        return self._default

    def resolve_name(self, name: str = None) -> str:
        """Explicit name, else the kind's env var, else the default."""
        chosen = name or os.environ.get(ENV_VARS[self.kind]) or self._default
        if chosen not in self._entries:
            raise UnknownEngine(self.kind, chosen, self._entries)
        return chosen

    def create(self, name: str = None) -> Any:
        """Build the backend for ``name`` (resolved as in ``resolve_name``)."""
        return self._entries[self.resolve_name(name)].factory()

    def probe(self) -> dict[str, tuple[bool, str]]:
        """``{name: (usable, reason)}`` for every backend, without raising.

        Backends answer ``available()`` themselves; one that blows up while
        being probed is reported as unusable with the error instead of taking
        the whole listing down.
        """
        report: dict[str, tuple[bool, str]] = {}
        for name in self.names():
            try:
                backend = self.create(name)
                ok, reason = backend.available()
            except Exception as exc:  # noqa: BLE001 - a probe must not raise
                ok, reason = False, f"{type(exc).__name__}: {exc}"
            report[name] = (bool(ok), reason)
        return report


_REGISTRIES: dict[str, Registry] = {kind: Registry(kind) for kind in KINDS}


def register(
    kind: str,
    name: str,
    factory: Callable[[], Any],
    describe: str = "",
    default: bool = False,
) -> None:
    """Register a backend for ``kind``."""
    if kind not in _REGISTRIES:
        raise KeyError(f"unknown engine kind {kind!r}")
    _REGISTRIES[kind].register(name, factory, describe=describe, default=default)


def registry(kind: str) -> Registry:
    if kind not in _REGISTRIES:
        raise KeyError(f"unknown engine kind {kind!r}")
    return _REGISTRIES[kind]


def names(kind: str) -> list[str]:
    return registry(kind).names()


def describe_all(kind: str) -> dict[str, str]:
    return registry(kind).describe()


def resolve_name(kind: str, name: str = None) -> str:
    return registry(kind).resolve_name(name)


def create(kind: str, name: str = None) -> Any:
    """The one entry point every caller uses: ``create("asr", name)``."""
    return registry(kind).create(name)


def probe(kind: str) -> dict[str, tuple[bool, str]]:
    """Usability of every backend of ``kind`` (see ``Registry.probe``)."""
    return registry(kind).probe()
