"""DumpScope / IDumper / ContextVar coverage."""
from __future__ import annotations

import asyncio
import threading

from tilefoundry.dump import (
    DumpFlags,
    DumpScope,
    FileDumper,
    MemoryDumper,
    NullDumper,
    current_scope,
    dump,
)


def test_dump_scope_replace_flag_gating_and_null() -> None:
    """Replace-form scope routes to its dumper; flag-gated bits drop writes;
    NullDumper short-circuits without surfacing anywhere."""
    dumper = MemoryDumper()
    with DumpScope(dumper=dumper, flags=DumpFlags.PASS_IR):
        dump("a.txt", "hello", DumpFlags.PASS_IR)
        dump("ignored.cu", "src", DumpFlags.CODEGEN_SOURCE)  # masked off
    assert dumper.entries == {"a.txt": "hello"}

    # NullDumper scope: writes are no-ops, scope IS still installed.
    with DumpScope(dumper=NullDumper, flags=DumpFlags.NONE):
        dump("any.txt", "x", DumpFlags.PASS_IR)
        assert current_scope() is not None


def test_dump_scope_subdir_intersects_parent_flags_and_path() -> None:
    """Subdir scope nests path under parent and intersects flags."""
    dumper = MemoryDumper()
    with DumpScope(dumper=dumper, flags=DumpFlags.PASS_IR):
        with DumpScope("inner", DumpFlags.CODEGEN_SOURCE):
            dump("x.cu", "src", DumpFlags.CODEGEN_SOURCE)  # parent masked → drop
        with DumpScope("inner", DumpFlags.ALL):
            dump("y.txt", "ir", DumpFlags.PASS_IR)
    assert dumper.entries == {"inner/y.txt": "ir"}


def test_file_dumper_writes_files(tmp_path) -> None:
    fd = FileDumper(tmp_path / "scope")
    with DumpScope(dumper=fd, flags=DumpFlags.ALL):
        dump("nested/module.cu", "kernel", DumpFlags.CODEGEN_SOURCE)
    assert (tmp_path / "scope" / "nested" / "module.cu").read_text() == "kernel"


def test_dump_scope_isolation_across_threads_and_asyncio_tasks() -> None:
    """Child thread / asyncio.Task with its own ``DumpScope`` writes only
    into that scope's dumper; parent's ContextVar is untouched."""
    parent_dumper = MemoryDumper()
    thread_results: list[dict[str, str | bytes]] = []

    def worker():
        local = MemoryDumper()
        with DumpScope(dumper=local, flags=DumpFlags.ALL):
            dump("t.txt", "child", DumpFlags.PASS_IR)
        thread_results.append(dict(local.entries))

    with DumpScope(dumper=parent_dumper, flags=DumpFlags.ALL):
        t = threading.Thread(target=worker)
        t.start(); t.join()  # noqa: E702
        dump("p.txt", "self", DumpFlags.PASS_IR)
    assert thread_results == [{"t.txt": "child"}]
    assert parent_dumper.entries == {"p.txt": "self"}

    async def task_local():
        local = MemoryDumper()
        with DumpScope(dumper=local, flags=DumpFlags.ALL):
            dump("a.txt", "child", DumpFlags.PASS_IR)
        return local.entries

    parent2 = MemoryDumper()

    async def driver():
        with DumpScope(dumper=parent2, flags=DumpFlags.ALL):
            entries = await asyncio.create_task(task_local())
            dump("p2.txt", "self", DumpFlags.PASS_IR)
        return entries

    entries = asyncio.run(driver())
    assert entries == {"a.txt": "child"}
    assert parent2.entries == {"p2.txt": "self"}
