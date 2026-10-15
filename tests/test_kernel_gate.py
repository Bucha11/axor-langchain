"""Every tool call through the current kernel gate (frozen c9v4 banking configuration).

The scenarios are the same in axor-claude, axor-openrouter, axor-langchain and
axor-wrap, each through its own public path: here AxorMiddleware.wrap_tool_call
(the agent's path) and AxorToolWrapper (wrap_tools)."""
from __future__ import annotations

import hashlib
import os
import re

import pytest

pytest.importorskip("langchain")

from langchain.agents.middleware import ToolCallRequest  # noqa: E402
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from axor_langchain import AxorMiddleware  # noqa: E402

CFG = os.path.join(os.path.dirname(__file__), "fixtures", "banking_c9v4.yaml")
GATE_SHA256 = "950916b2d32e364e55c9aaeacab822e6f60e4306db5464bdfd8458347c098502"
PAY = {"recipient": "DE89370400440532013000", "amount": 50.0, "subject": "rent", "date": "2024-01-01"}
BILL = "Bill for December 2023. Pay 98.70 to UK12345678901234567890."


class Agent:
    """Drives wrap_tool_call as the agent loop would; records what the tools ran with."""

    def __init__(self, task: str | None, **kw):
        self.mw = AxorMiddleware(governance=CFG, telemetry="off", **kw)
        self.task = task
        self.calls: list = []
        self.n = 0

    def call(self, name: str, args: dict):
        self.n += 1
        state = {"messages": [HumanMessage(self.task)] if self.task else []}
        req = ToolCallRequest(tool_call={"name": name, "args": args, "id": f"c{self.n}"},
                              tool=None, state=state, runtime=None)

        def handler(r):
            self.calls.append((name, dict(r.tool_call["args"])))
            out = {"send_money": "sent", "read_file": BILL, "get_balance": "1810.0"}[name]
            return ToolMessage(content=out, tool_call_id=r.tool_call["id"])
        msg = self.mw.wrap_tool_call(req, handler)
        return msg.content, getattr(msg, "status", "success")


def test_gate_module_is_the_shared_one():
    import axor_langchain.gate as g
    assert hashlib.sha256(open(g.__file__, "rb").read()).hexdigest() == GATE_SHA256


def test_refused_call_never_reaches_the_tool():
    a = Agent("What's my balance?")
    a.call("get_balance", {})
    out, status = a.call("send_money", {**PAY, "recipient": "US133000000121212121212", "amount": 1810})
    assert status == "error" and out.startswith("tool_denied: governance (")
    assert [c[0] for c in a.calls] == ["get_balance"]


def test_request_grant_one_use():
    a = Agent(None)
    assert a.call("send_money", PAY)[1] == "error"  # no request: nothing grants the IBAN
    a.task = "Send 50 to DE89370400440532013000 for the rent."
    assert a.call("send_money", PAY) == ("sent", "success")
    out, status = a.call("send_money", PAY)  # the L1 grant had one use
    assert status == "error" and "(grant_gate)" in out
    assert len(a.calls) == 1


def test_references_and_confirmed_call_runs_on_the_resolved_value():
    from axor_core.kernel.confirm import ScriptedConfirmer

    a = Agent("Pay the bill in bill-december-2023.txt.")
    view, _ = a.call("read_file", {"file_path": "bill-december-2023.txt"})
    assert "UK12345678901234567890" not in view
    ref = re.search(r"iban_from_untrusted_read_hidden_\w+", view).group(0)
    out, status = a.call("send_money", {**PAY, "recipient": ref})
    assert status == "error" and "(reference_origin)" in out

    b = Agent("Pay the bill in bill-december-2023.txt.", confirmer=ScriptedConfirmer(lambda x, c: True))
    ref = re.search(r"iban_from_untrusted_read_hidden_\w+", b.call("read_file", {"file_path": "x"})[0]).group(0)
    assert b.call("send_money", {**PAY, "recipient": ref}) == ("sent", "success")
    assert b.calls[-1][1]["recipient"] == "UK12345678901234567890"


def test_wrapped_tool_is_gated_once_by_its_wrapper():
    ran = []

    @tool
    def send_money(recipient: str, amount: float, subject: str, date: str) -> str:
        """Send money."""
        ran.append(recipient)
        return "sent"

    mw = AxorMiddleware(governance=CFG, telemetry="off")
    [wrapped] = mw.wrap_tools([send_money])
    mw._tool_gate.request("Send 50 to DE89370400440532013000 for the rent.")
    assert wrapped._run(**PAY) == "sent"
    assert wrapped._run(**PAY).startswith("tool_denied: governance (grant_gate)")
    assert ran == ["DE89370400440532013000"]


FLOOR = {"sensitive_sources": ["get_balance"], "egress_sinks": ["send_money"]}


def _egress_during_slow_read(mw) -> tuple:
    """A secret read whose tool takes 0.2 s and an egress proposed while it runs, from
    two threads, as LangGraph runs one message's tool calls."""
    import threading
    import time

    started, out = threading.Event(), {}
    state = {"messages": [HumanMessage("What is my balance?")]}

    def run(name, args, content, slow):
        req = ToolCallRequest(tool_call={"name": name, "args": args, "id": name},
                              tool=None, state=state, runtime=None)

        def handler(r):
            started.set()
            if slow:
                time.sleep(0.2)
            return ToolMessage(content=content, tool_call_id=name)
        out[name] = mw.wrap_tool_call(req, handler)

    t = threading.Thread(target=run, args=("get_balance", {}, "1810.0", True))
    t.start()
    started.wait()
    run("send_money", {"recipient": "X1", "amount": 1}, "sent", False)
    t.join()
    return out["get_balance"], out["send_money"]


def test_concurrent_calls_are_decided_after_each_other_E():
    """With the secret read registered first the floor refuses the egress (sequential
    reference); run concurrently, the egress waits for the read and is refused too."""
    mw = AxorMiddleware(governance=FLOOR, telemetry="off")
    read, egress = _egress_during_slow_read(mw)
    assert read.content == "1810.0"
    assert getattr(egress, "status", None) == "error" and "(taint_enforcement)" in egress.content


@pytest.mark.asyncio
async def test_async_wrapped_tool_inside_the_middleware_does_not_deadlock():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    async def get_balance() -> str:
        """Balance."""
        return "1810.0"

    mw = AxorMiddleware(governance=CFG, telemetry="off")
    [wrapped] = mw.wrap_tools([get_balance])
    req = ToolCallRequest(tool_call={"name": "get_balance", "args": {}, "id": "x"}, tool=wrapped,
                          state={"messages": [HumanMessage("What's my balance?")]}, runtime=None)

    async def handler(r):
        return ToolMessage(content=await wrapped._arun(), tool_call_id="x")
    import asyncio
    msg = await asyncio.wait_for(mw.awrap_tool_call(req, handler), timeout=5)
    assert msg.content == "1810.0"
