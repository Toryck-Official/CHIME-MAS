"""导出 PDDL 60 题的目标描述（task_main），供离线归因评测使用；只用环境，不调用 LLM。"""
import json, sys
sys.path.insert(0, "/home/claude/chime")
from compare_attr import task_goals
json.dump(task_goals(), open("/home/claude/chime/mas_runs/e1/task_goals.json", "w"), ensure_ascii=False, indent=1)
print("GOALS_DONE")
