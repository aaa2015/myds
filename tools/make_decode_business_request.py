#!/usr/bin/env python3
"""Deterministic mixed Chinese/code/data input; freeze only after target token audit."""
import argparse
import json
from pathlib import Path
import random

SEED = 20261009
REGIONS = ("华东", "华南", "华北", "西南", "华中", "东北")
PRODUCTS = ("工业传感器", "仓储扫描终端", "电源控制模块", "物流定位标签", "质检相机", "智能温控探头")
ISSUES = (
    "采购到货延迟与促销需求叠加，需要区分缺货和系统预留量",
    "退货批次在跨仓调拨时出现重复状态，业务人员正在核对事件顺序",
    "渠道折扣和运费补贴同时变化，应把销量、毛利和履约成本分开解释",
    "夜间设备维护影响产线节拍，需比较停机窗口与出货承诺",
    "门店盘点与线上订单存在时间差，需要逐日核对库存流水",
    "供应商切换引入新质检规则，需要保留原料批号和验收结果",
)
TAIL = """
请把上述合成业务记录作为一份完整的运营审阅材料。现在开始输出一份很长的中文审阅报告，
至少写出256条连续编号的分析，不要只给摘要，也不要提前宣布完成。每条都引用具体记录编号、
日期或数据字段，说明计算过程、代码行为、业务含义和下一项核对动作。先依次分析前64个记录，
再分析后面的记录，最后比较地区和产品之间的差异。每条至少写三句完整解释，避免复用上一条。
从“1. 记录00000的需求与供给核对”开始，紧接着写详细正文。不要复述任务要求。
"""


def make_request(records, padding_items):
    rng = random.Random(SEED)
    parts = ["2026年第四季度运营审阅档案。以下公司、订单和指标均为合成数据；各记录编号不同。\n"]
    for i in range(records):
        region, product = REGIONS[i % len(REGIONS)], PRODUCTS[(i * 5 + 1) % len(PRODUCTS)]
        demand = rng.randrange(90, 480)
        received = demand + rng.randrange(-35, 42)
        stock = rng.randrange(45, 220)
        defects = rng.randrange(1, 14)
        unit_price = rng.randrange(70, 250)
        shipping = rng.randrange(420, 2200)
        parts.append(f"""
记录{i:05d}：{region}运营中心，产品{product}，负责人团队T{i % 19:02d}。
本周专题：{ISSUES[i % len(ISSUES)]}。订单批次ORD-{i:05d}具有独立账期，
不能把不同记录的期末库存相加后当作同一仓库余额。客户要求按承诺日期拆分延期原因，
财务要求先核对净出货，再解释运输补贴，质检要求保留缺陷样本而非只报告平均值。
日期|需求量|到货量|期初库存|缺陷件|单价元|运输成本元
2026-10-{i % 28 + 1:02d}|{demand}|{received}|{stock}|{defects}|{unit_price}|{shipping}
2026-11-{i % 27 + 1:02d}|{demand + 17}|{received + 11}|{stock + 9}|{defects + 1}|{unit_price + 3}|{shipping + 83}

```python
def audit_batch_{i}(demand, received, stock, defects, unit_price, shipping):
    usable = max(received - defects, 0)
    fulfilled = min(demand, stock + usable)
    delayed = max(demand - fulfilled, 0)
    revenue = fulfilled * unit_price
    return {{"fulfilled": fulfilled, "delayed": delayed,
            "ending_stock": stock + usable - fulfilled,
            "revenue_after_shipping": revenue - shipping}}
```
核对重点：比较两行记录中需求、到货和缺陷变化；识别代码对负可用量的处理；
解释收入扣运输费后的含义，不能将其直接称为净利润。补充每个字段来源和检查顺序。
""")
    for j in range(padding_items):
        parts.append(f"补充核对项{j:03d}：订单A{j:04d}的仓库余额为{180 + j * 7}件，"
                     f"预留量{23 + j % 11}件，退货待检{4 + j % 5}件。请区分实物、可售和预留数量。\n")
    parts.append(TAIL)
    return {
        "model": "local-myds-deepseek-v4.1-flash-q2",
        "messages": [
            {"role": "system", "content": "你是企业运营审阅员。根据给定的合成中文说明、代码和数据表逐项分析，保持编号、单位与计算过程一致。"},
            {"role": "user", "content": "".join(parts)},
        ],
        "max_tokens": 256, "temperature": 0, "top_p": 1, "top_k": 0,
        "min_p": 0, "repetition_penalty": 1, "seed": 42, "stream": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--padding-items", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.records < 0 or args.padding_items < 0:
        parser.error("record and padding counts must be nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(make_request(args.records, args.padding_items),
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
