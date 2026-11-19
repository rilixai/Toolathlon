# monkeypatch
from __future__ import annotations
import os
from typing import Any
from agents import RunContextWrapper
from agents.logger import logger
from openai.types.responses import ResponseFunctionToolCall
import shortuuid


MAX_SINGLE_TURN_RETURN_CHARS = int(os.getenv("BENCH_MAX_SINGLE_TURN_RETURN_CHARS", 100000))


def _resolve_registered_tool_name(returned_name: str, registered_names) -> str | None:
    """Resolve a provider-normalized name to one unambiguous registered name."""
    names = list(registered_names)
    if returned_name in names:
        return returned_name

    normalized_name = returned_name.replace("-", "_")
    matches = [name for name in names if name.replace("-", "_") == normalized_name]
    return matches[0] if len(matches) == 1 else None


def _canonicalize_tool_call(
    tool_call: ResponseFunctionToolCall, registered_names
) -> ResponseFunctionToolCall:
    """Copy a tool call with its canonical registered name when resolvable."""
    canonical_name = _resolve_registered_tool_name(tool_call.name, registered_names)
    if canonical_name is None or canonical_name == tool_call.name:
        return tool_call
    return tool_call.model_copy(update={"name": canonical_name})


def _truncate_overlong_tool_output(
    tool_output: Any, context_wrapper: RunContextWrapper[Any]
) -> Any:
    tool_output_text = tool_output if isinstance(tool_output, str) else str(tool_output)
    if len(tool_output_text) <= MAX_SINGLE_TURN_RETURN_CHARS:
        return tool_output

    original_length = len(tool_output_text)
    logger.warning("Tool output is too long, return truncated one.")
    tool_short_uuid = shortuuid.uuid()

    agent_workspace = context_wrapper.context.get("_agent_workspace", ".")
    agent_workspace = os.path.abspath(agent_workspace)
    overlong_toolcall_save_dir = os.path.join(
        agent_workspace, ".overlong_tool_outputs"
    )
    os.makedirs(overlong_toolcall_save_dir, exist_ok=True)
    output_path = os.path.join(overlong_toolcall_save_dir, f"{tool_short_uuid}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tool_output_text)
    logger.warning(f"Tool output saved to {output_path}")

    return (
        tool_output_text[:MAX_SINGLE_TURN_RETURN_CHARS]
        + f" ...\n\n(The output of the tool call (shortuuid identifier: {tool_short_uuid}) is too long! "
        f"Only the first {MAX_SINGLE_TURN_RETURN_CHARS} characters are shown here. "
        f"The original output length is {original_length} characters. "
        f"The full output has been saved to the file {output_path}. "
        "Please check this file carefully, as it may be very long!)"
    )
