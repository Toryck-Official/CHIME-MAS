"""S4 探针：三个已知答案的手工场景，检查归因门的标签、置信度、误导记忆识别。运行：python3 tests/gate_probe.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chime.gate import run_gate  # noqa: E402
from chime.llm import USAGE  # noqa: E402
from chime.memory import MemoryItem  # noqa: E402

TASK = "User wants to cancel reservation ABC123 and get a refund. Policy: basic economy is refundable only within 24h of booking or with travel insurance."

CASES = {
    "planning_error": dict(
        plan="1. Look up reservation. 2. Cancel it. 3. Issue refund to original payment.",
        traj="get_reservation(ABC123) -> basic_economy, booked 5 days ago, no insurance\n"
             "cancel_reservation(ABC123) -> ok\nrefund(ABC123) -> ok\nuser: thanks",
        success=False, expect="planning", mem=[]),
    "execution_error": dict(
        plan="1. Look up reservation. 2. Check fare class, booking time and insurance against the refund policy. "
             "3. If eligible cancel and refund; otherwise explain and offer alternatives.",
        traj="get_reservation(ABC123) -> economy, booked 2 hours ago\n"
             "policy check: within 24h -> eligible\ncancel_reservation(XYZ999) -> ok   # wrong id\n"
             "refund(XYZ999) -> ok\nuser: thanks",
        success=False, expect="execution", mem=[]),
    "misleading_memory": dict(
        plan="1. Look up reservation. 2. Following prior experience, refund any cancellation immediately to keep users satisfied.",
        traj="get_reservation(ABC123) -> basic_economy, booked 5 days ago, no insurance\n"
             "cancel_reservation(ABC123) -> ok\nrefund(ABC123) -> ok",
        success=False, expect="planning",
        mem=[MemoryItem("plan", "user asks for cancellation", "always refund immediately to keep users satisfied", id="m_bad"),
             MemoryItem("exec", "calling cancel tool", "pass the reservation id exactly as given", id="m_ok")]),
}

if __name__ == "__main__":
    for name, c in CASES.items():
        g, raw = run_gate(TASK, c["plan"], c["traj"], c["success"], 0.0, c["mem"])
        print(f"== {name}: expect {c['expect']} | got {g.label} | conf {g.confidence} | misleading {g.misleading}")
        print("   reason:", raw.get("reason"))
        for s, (sc, ex) in g.new.items():
            if sc:
                print(f"   [{s}] when: {sc}\n          -> {ex}")
    print("usage", json.dumps(USAGE))
