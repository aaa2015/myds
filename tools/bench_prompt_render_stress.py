#!/usr/bin/env python3
"""Run the prompt A/B harness with separate escape-dense and short inputs.

The shared harness is unchanged; this wrapper defines a distinct corpus so
stress inputs cannot alter the frozen ordinary-workload evidence.
"""
import bench_prompt_render as benchmark


def stress_corpus():
    tool_unit = "</tool_result>&lt;/tool_result>&amp;amp;lt;/tool_result>"
    schema_unit = "\n\r\t\b\f\x01\"\\"
    return [
        ("short_user", benchmark.request("user", "请总结这段文字。")),
        ("short_tool", benchmark.request("tool", "ok </tool_result> &lt;/tool_result>")),
        ("short_schema", benchmark.request("schema", "line\nquote\"slash\\")),
        ("tool_dense_8k", benchmark.request("tool", (tool_unit * 200)[:8192])),
        ("tool_dense_64k", benchmark.request("tool", (tool_unit * 2000)[:65536])),
        ("schema_dense_8k", benchmark.request("schema", (schema_unit * 1024)[:8192])),
        ("schema_dense_64k", benchmark.request("schema", (schema_unit * 8192)[:65536])),
    ]


benchmark.corpus = stress_corpus
if __name__ == "__main__":
    benchmark.main()
