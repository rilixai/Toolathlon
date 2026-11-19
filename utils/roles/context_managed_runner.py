# context_managed_runner.py (English version)
import json
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from pathlib import Path

from agents import Runner, RunConfig, RunHooks, Agent, RunResult
from agents.run_context import RunContextWrapper
from agents.items import (
    RunItem, TResponseInputItem, MessageOutputItem, 
    ItemHelpers,
    ModelResponse
)
from agents.run_config import ModelInputData
from types import SimpleNamespace
import inspect


from dataclasses import dataclass, field, replace

from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from utils.api_model.model_provider import ContextTooLongError



# This is cirectly copied from agents/run.py in new version of openai-agents-sdk library
@dataclass
class _ServerConversationTracker:
    """Tracks server-side conversation state for either conversation_id or
    previous_response_id modes.

    Note: When auto_previous_response_id=True is used, response chaining is enabled
    automatically for the first turn, even when there's no actual previous response ID yet.
    """

    previous_response_id: str | None = None
    auto_previous_response_id: bool = False
    sent_items: set[int] = field(default_factory=set)
    server_items: set[int] = field(default_factory=set)

    def track_server_items(self, model_response: ModelResponse) -> None:
        for output_item in model_response.output:
            self.server_items.add(id(output_item))

        # Update previous_response_id when using previous_response_id mode or auto mode
        if (
            (self.previous_response_id is not None or self.auto_previous_response_id)
            and model_response.response_id is not None
        ):
            self.previous_response_id = model_response.response_id

    def reset(self,):
        # we use this when we encounter a context too long error
        # in this case, the state of this tracker should be completely reset
        self.previous_response_id = None
        self.sent_items = set()
        self.server_items = set()

    def prepare_input(
        self,
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
    ) -> list[TResponseInputItem]:
        input_items: list[TResponseInputItem] = []

        # On first call (when there are no generated items yet), include the original input
        if not generated_items:
            input_items.extend(ItemHelpers.input_to_new_input_list(original_input))

        # Process generated_items, skip items already sent or from server
        for item in generated_items:
            raw_item_id = id(item.raw_item)

            if raw_item_id in self.sent_items or raw_item_id in self.server_items:
                continue
            input_items.append(item.to_input_item())
            self.sent_items.add(raw_item_id)

        return input_items


class ContextManagedRunner(Runner):
    """A Runner that supports context management and history recording."""
    
    # Default directory for storing conversation history files
    DEFAULT_HISTORY_DIR = Path("conversation_histories")
    
    @classmethod
    async def run(
        cls,
        starting_agent: Agent,
        input: str | list[TResponseInputItem],
        *,
        context: Any = None,
        max_turns: int = 10,
        hooks: RunHooks | None = None,
        run_config: RunConfig | None = None,
        previous_response_id: str | None = None,
        history_dir: Union[str, Path, None] = None,  # newly added parameter
        session_id: Optional[str] = None,  # allow specifying session_id,
    ) -> RunResult:
        """Override the run method to add context management functionality.
        
        Args:
            history_dir: Directory to store conversation history files. Use default if None.
            session_id: Specify the session ID. If None, one is generated automatically.
            ... other parameters as in the parent class.
        """
        
        # Handle history directory
        if history_dir is None:
            history_dir = cls.DEFAULT_HISTORY_DIR
        else:
            history_dir = Path(history_dir)
        
        # Make sure the directory exists
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate or use provided session ID
        if session_id is None:
            session_id = cls._generate_session_id()
        
        # Create wrapped context
        if context is None:
            context = {}
        
        # Initialize context metadata, including history dir info
        wrapped_context = cls._init_context_metadata(context, session_id, history_dir)
        
        # Record initial input to history (already managed by user side, skip here)
        # cls._save_initial_input_to_history(session_id, input, history_dir)

        # SDK 0.20 exposes model-input filters and lifecycle hooks instead of
        # Runner._run_single_turn. Keep this runner's original metadata,
        # history format and truncation policies on those public boundaries.
        tracker = wrapped_context.get("_server_conversation_tracker")
        state = _ContextRun(cls, wrapped_context, input, hooks or RunHooks())
        config = run_config or RunConfig()
        upstream_filter = config.call_model_input_filter

        async def filter_input(data):
            filtered = state.filter_input(data.model_data)
            if upstream_filter is not None:
                filtered = upstream_filter(replace(data, model_data=filtered))
                if inspect.isawaitable(filtered):
                    filtered = await filtered
            return filtered

        try:
            result = await super().run(
                starting_agent=starting_agent,
                input=input,
                context=wrapped_context,
                max_turns=max_turns,
                hooks=state,
                run_config=replace(config, call_model_input_filter=filter_input),
                previous_response_id=(tracker.previous_response_id if tracker else previous_response_id),
                auto_previous_response_id=bool(tracker and tracker.auto_previous_response_id),
            )
        except ContextTooLongError as error:
            meta = wrapped_context["_context_meta"]
            wrapped_context["_force_reset_context"] = {
                "reason": str(error),
                "token_count": error.token_count,
                "max_tokens": error.max_tokens,
                "timestamp": datetime.now().isoformat(),
                "executed_mini_turns": meta["mini_turns_in_current_sequence"],
                "executed_turns": meta["turns_in_current_sequence"],
            }
            raise
        except Exception as error:
            if getattr(error, "run_data", None) is not None:
                state.finish(error.run_data)
            raise
        state.finish(result)
        return result

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent,
        input: str | list[TResponseInputItem],
        *,
        context: Any = None,
        max_turns: int = 10,
        hooks: RunHooks | None = None,
        run_config: RunConfig | None = None,
        previous_response_id: str | None = None,
        history_dir: Union[str, Path, None] = None,  # new parameter
        session_id: Optional[str] = None,  # allow specifying session_id
    ) -> RunResult:
        """Synchronous version of run, supporting history recording."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            cls.run(
                starting_agent=starting_agent,
                input=input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                previous_response_id=previous_response_id,
                history_dir=history_dir,
                session_id=session_id,
            )
        )

    @classmethod
    def _init_context_metadata(cls, context: Any, session_id: str, history_dir: Path) -> Any:
        """Initialize or update context metadata."""
        # If context is None, create a new one
        if context is None:
            context = {}
        
        # Check if already initialized
        if "_context_meta" in context:
            # Already initialized, don't overwrite
            return context
        
        # First-time initialization
        metadata = {
            "session_id": session_id,
            "history_dir": str(history_dir),
            "started_at": datetime.now().isoformat(),
            "current_turn": 0,
            "total_turns_ever": 0,
            "turns_in_current_sequence": 0,
            "mini_turns_in_current_sequence": 0,
            "boundary_in_current_sequence": [],
            "truncated_turns": 0,
            "truncation_history": []
        }
        
        context["_session_id"] = session_id
        context["_history_dir"] = str(history_dir)
        context["_context_meta"] = metadata
        context["_context_limit"] = context.get("_context_limit", 128000)
        
        return context

    @classmethod
    def _handle_truncation(
        cls,
        original_input: List[TResponseInputItem],
        pre_step_items: List[RunItem],
        new_step_items: List[RunItem],
        truncate_params: Dict[str, Any],
        context_wrapper: RunContextWrapper
    ):
        """Handle context truncation request."""
        method = truncate_params.get("method")
        value = truncate_params.get("value")
        preserve_system = truncate_params.get("preserve_system", True)
        preserve_first_user_input = truncate_params.get("preserve_first_user_input", True)
        
        ctx = context_wrapper.context if context_wrapper else {}
        meta = ctx.get("_context_meta", {})
        
        # Get all turn boundaries
        turn_boundaries = ctx["_context_meta"]["boundary_in_current_sequence"]
        total_turns = len(turn_boundaries)
        
        # Assert turn count consistency
        current_turns_in_sequence = meta.get("turns_in_current_sequence", 0)
        assert total_turns == current_turns_in_sequence, (
            f"Turn boundary count ({total_turns}) does not equal turns_in_current_sequence ({current_turns_in_sequence})"
        )
        
        if total_turns == 0:
            return  # Nothing to truncate

        if method == "smart_ranges":
            retained_turn_indices = []
            if preserve_first_user_input:
                retained_turn_indices.append(0)

            for start, end in truncate_params.get("ranges", []):
                for turn_idx in range(start, end + 1):
                    retained_turn_indices.append(turn_idx)

            retained_turn_indices = sorted(set(retained_turn_indices))
            delete_turns = total_turns - len(retained_turn_indices)
            if delete_turns <= 0:
                return

            retained_boundaries = [turn_boundaries[idx] for idx in retained_turn_indices]
            deleted_items_count, new_boundaries = cls._retain_sequential_turns(
                original_input,
                pre_step_items,
                new_step_items,
                retained_boundaries,
            )
            keep_turns = len(new_boundaries)
        else:
            protected_boundaries = turn_boundaries[:1] if preserve_first_user_input else []
            eligible_boundaries = turn_boundaries[len(protected_boundaries):]
            eligible_turns = len(eligible_boundaries)
            keep_turns = eligible_turns

            if method == "keep_recent_turns":
                keep_turns = min(int(value), eligible_turns)
            elif method == "keep_recent_percent":
                keep_turns = max(1, int(eligible_turns * value / 100)) if eligible_turns > 0 else 0
            elif method == "delete_first_turns":
                keep_turns = max(0, eligible_turns - int(value))
            elif method == "delete_first_percent":
                delete_turns = int(eligible_turns * value / 100)
                keep_turns = max(0, eligible_turns - delete_turns)
            else:
                return

            if keep_turns >= eligible_turns:
                return

            print("keep_turns < total_turns, truncating history")
            delete_turns = eligible_turns - keep_turns
            deleted_items_count, new_boundaries = cls._truncate_sequential_lists(
                original_input,
                pre_step_items,
                new_step_items,
                eligible_boundaries,
                delete_turns,
                preserve_system,
                protected_boundaries,
            )
            keep_turns = len(new_boundaries)

        if deleted_items_count > 0:
            meta["turns_in_current_sequence"] = keep_turns
            meta["truncated_turns"] = meta.get("truncated_turns", 0) + delete_turns
            meta["truncation_history"].append({
                "at_turn": meta["current_turn"],
                "method": method,
                "value": value,
                "deleted_items": deleted_items_count,
                "deleted_turns": delete_turns,
                "timestamp": datetime.now().isoformat(),
                "preserve_first_user_input": preserve_first_user_input,
            })
            ctx["_context_truncated"] = True

            meta["mini_turns_in_current_sequence"] = len(original_input) + len(pre_step_items) + len(new_step_items)
            meta["boundary_in_current_sequence"] = new_boundaries
    
    @classmethod
    def _find_turn_boundaries(cls, items: List[TResponseInputItem]) -> List[tuple[int, int]]:
        """Find dialogue turn boundaries [(start_idx, end_idx), ...]
        
        Turn is defined as: a user or assistant message starts a new turn, tools are attached to their preceding assistant.
        """
        boundaries = []
        # TODO: implement this
        
        return boundaries
    
    @classmethod
    def _truncate_sequential_lists(
        cls,
        original_input: List[TResponseInputItem],
        pre_step_items: List[RunItem],
        new_step_items: List[RunItem],
        boundaries: List[tuple[int, int]],
        delete_turns: int,
        preserve_system: bool,
        protected_boundaries: Optional[List[tuple[int, int]]] = None,
    ) -> tuple[int, List[tuple[int, int]]]:
        """Truncate three lists in order, optionally preserving selected leading boundaries."""
        if delete_turns <= 0:
            current_boundaries = list(protected_boundaries or []) + boundaries
            return 0, current_boundaries

        retained_boundaries = list(protected_boundaries or [])
        retained_boundaries.extend(boundaries[delete_turns:])
        return cls._retain_sequential_turns(
            original_input,
            pre_step_items,
            new_step_items,
            retained_boundaries,
        )

    @classmethod
    def _retain_sequential_turns(
        cls,
        original_input: List[TResponseInputItem],
        pre_step_items: List[RunItem],
        new_step_items: List[RunItem],
        retained_boundaries: List[tuple[int, int]],
    ) -> tuple[int, List[tuple[int, int]]]:
        """Retain only the specified turn boundaries and rebase them onto the compacted sequence."""
        total_items = len(original_input) + len(pre_step_items) + len(new_step_items)
        if not retained_boundaries:
            original_input.clear()
            pre_step_items.clear()
            new_step_items.clear()
            return total_items, []

        combined_items = [("original", item) for item in original_input]
        combined_items.extend([("pre", item) for item in pre_step_items])
        combined_items.extend([("new", item) for item in new_step_items])

        kept_items = []
        boundary_idx = 0
        for item_idx, tagged_item in enumerate(combined_items):
            while boundary_idx < len(retained_boundaries) and item_idx >= retained_boundaries[boundary_idx][1]:
                boundary_idx += 1
            if boundary_idx >= len(retained_boundaries):
                break

            start_idx, end_idx = retained_boundaries[boundary_idx]
            if item_idx < start_idx:
                continue

            kept_items.append(tagged_item)

        original_input[:] = [item for source, item in kept_items if source == "original"]
        pre_step_items[:] = [item for source, item in kept_items if source == "pre"]
        new_step_items[:] = [item for source, item in kept_items if source == "new"]

        new_boundaries = []
        cursor = 0
        for start_idx, end_idx in retained_boundaries:
            boundary_length = end_idx - start_idx
            new_boundaries.append((cursor, cursor + boundary_length))
            cursor += boundary_length

        deleted_items_count = total_items - len(kept_items)
        return deleted_items_count, new_boundaries
    
    @classmethod
    def _create_truncation_notice(cls, method: str, value: Any, deleted_items: int, deleted_turns: int) -> MessageOutputItem:
        """Create a system truncation notice message."""
        content = (f"[Context Management] Due to token limit, used strategy {method}({value}) for truncation. "
                   f"{deleted_items} message(s) (about {deleted_turns} turns) were deleted.")
        
        # Create a system message
        raw_message = ResponseOutputMessage(
            id="system_truncation",
            content=[ResponseOutputText(
                text=content,
                type="output_text",
                annotations=[]
            )],
            role="system",
            type="message",
            status="completed"
        )
        
        # Use a placeholder agent for this system message.
        from agents import Agent
        placeholder_agent = Agent(name="system", model="gpt-4")
        
        return MessageOutputItem(
            agent=placeholder_agent,
            raw_item=raw_message
        )
    
    @classmethod
    def _save_items_to_history(
        cls,
        session_id: str,
        turn_number: int,
        items: List[RunItem],
        agent_name: str,
        history_dir: Path
    ):
        """Save items to the history file."""
        history_path = history_dir / f"{session_id}_history.jsonl"
        # Ensure directory exists
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # print("Entering _save_items_to_history")
        with open(history_path, 'a', encoding='utf-8') as f:
            for step_idx, item in enumerate(items):
                # print("Saving item")
                record = {
                    "in_turn_steps": step_idx,  # Step index within the turn
                    "turn": turn_number,
                    "timestamp": datetime.now().isoformat(),
                    "agent": agent_name,
                    "item_type": item.type,
                    "raw_content": item.raw_item.model_dump() if hasattr(item.raw_item, 'model_dump') else item.raw_item
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    @classmethod
    def _save_initial_input_to_history(cls, 
                                       session_id: str, 
                                       input: Union[str, List[TResponseInputItem]], 
                                       history_dir: Path,
                                       turn_number: int = 0):
        """Save the initial input to history."""
        history_path = history_dir / f"{session_id}_history.jsonl"
        # Ensure directory exists
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing initial input entry
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get("type") == "initial_input":
                            return  # Already exists
                    except json.JSONDecodeError:
                        continue

        with open(history_path, 'a', encoding='utf-8') as f:
            record = {
                "in_turn_steps": 0,  # Initial input is always the first step
                "turn": turn_number,
                "timestamp": datetime.now().isoformat(),
                "type": "initial_input",
                "content": input if isinstance(input, str) else [item.model_dump() for item in input]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


    @classmethod
    def _save_user_input_to_history(cls, session_id: str, user_input: Union[str, TResponseInputItem], history_dir: Path, turn_number: int):
        """Save user input to history."""
        history_path = Path(history_dir) / f"{session_id}_history.jsonl"
        # Ensure directory exists
        history_path.parent.mkdir(parents=True, exist_ok=True)

        with open(history_path, 'a', encoding='utf-8') as f:
            record = {
                "in_turn_steps": 0,  # User input is the first step in its turn
                "turn": turn_number,
                "timestamp": datetime.now().isoformat(),
                "type": "user_input",
                "content": user_input if isinstance(user_input, str) else [item.model_dump() for item in user_input]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    @classmethod
    def _generate_session_id(cls) -> str:
        """Generate a unique session ID."""
        from uuid import uuid4
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


    @classmethod
    def get_formatted_history(cls, history_dir: Union[str, Path], session_id: str) -> List[Dict[str, Any]]:
        """Get formatted conversation history suitable for logging.
        
        Returns:
            The formatted message list, suitable for logging.
        """
        history_file = Path(history_dir) / f"{session_id}_history.jsonl"
        
        if not history_file.exists():
            return []
        
        formatted_messages = []
        
        # Read all records, sorted by turn and step
        records = []
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        records.sort(key=lambda x: (x.get("turn", 0), x.get("in_turn_steps", 0)))
        
        # Process each turn
        current_turn = -1
        current_turn_records = []
        
        for record in records:
            # Skip initial input record
            if record.get("type") == "initial_input":
                continue
                
            turn = record.get("turn", 0)
            
            # If turn changes, process previous turn's records
            if turn != current_turn and current_turn_records:
                formatted_messages.extend(cls._process_turn_records(current_turn_records))
                current_turn_records = []
            
            current_turn = turn
            current_turn_records.append(record)
        
        # Process last turn's records
        if current_turn_records:
            formatted_messages.extend(cls._process_turn_records(current_turn_records))
        
        return formatted_messages
    
    @classmethod
    def _process_turn_records(cls, records: List[Dict]) -> List[Dict]:
        """Process the records of a single turn, returns the formatted message list."""
        formatted_messages = []
        item_index = 0
        
        while item_index < len(records):
            current_record = records[item_index]
            
            if current_record.get("type") == "user_input":
                # User input
                content = current_record.get("content", "")
                if isinstance(content, list):
                    # If content is a list, extract text parts
                    content_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content_parts.append(item.get("text", ""))
                    content = " ".join(content_parts)
                
                formatted_messages.append({
                    "role": "user",
                    "content": content
                })
                item_index += 1
                
            elif current_record.get("item_type") == "message_output_item":
                raw_content = current_record.get("raw_content", {})
                role = "unknown"
                content = ""
                pure_thinking_str = None

                if isinstance(raw_content, dict):
                    role = raw_content.get("role", "unknown")
                    # Extract text content and reasoning content
                    content_parts = []
                    for content_item in raw_content.get("content", []):
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "output_text":
                                content_parts.append(content_item.get("text", ""))
                            elif content_item.get("type") == "reasoning_content":
                                pure_thinking_str = content_item.get("pure_thinking_str")
                    content = " ".join(content_parts)
                
                if role == "system" and "Context Management" in content:
                    # Skip context management system messages
                    item_index += 1
                    continue
                
                # Check if this is the last message (no following tool call)
                if item_index == len(records) - 1:
                    # Last message, final assistant reply
                    message = {
                        "role": role,
                        "content": content
                    }
                    if pure_thinking_str:
                        message["thinking"] = pure_thinking_str
                    formatted_messages.append(message)
                    item_index += 1
                else:
                    # Not last; check if tools follow
                    tool_calls = []
                    next_index = item_index + 1
                    
                    # Collect subsequent tool calls
                    while next_index < len(records) and records[next_index].get("item_type") == "tool_call_item":
                        tool_record = records[next_index]
                        raw_content = tool_record.get("raw_content", {})
                        tool_call = {
                            "id": raw_content.get("call_id", "unknown") if isinstance(raw_content, dict) else "unknown",
                            "type": "function",
                            "function": {
                                "name": raw_content.get("name", "unknown") if isinstance(raw_content, dict) else "unknown",
                                "arguments": raw_content.get("arguments", "{}") if isinstance(raw_content, dict) else "{}"
                            }
                        }
                        tool_calls.append(tool_call)
                        next_index += 1
                    
                    if tool_calls:
                        # Assistant message with tool calls
                        message = {
                            "role": role,
                            "content": content,
                            "tool_calls": tool_calls
                        }
                        if pure_thinking_str:
                            message["thinking"] = pure_thinking_str
                        formatted_messages.append(message)
                        item_index = next_index
                    else:
                        # No tool call
                        message = {
                            "role": role,
                            "content": content
                        }
                        if pure_thinking_str:
                            message["thinking"] = pure_thinking_str
                        formatted_messages.append(message)
                        item_index += 1
                        
            elif current_record.get("item_type") == "tool_call_item":
                # Tool call with no content
                tool_calls = []
                next_index = item_index
                
                # Collect consecutive tool calls
                while next_index < len(records) and records[next_index].get("item_type") == "tool_call_item":
                    tool_record = records[next_index]
                    raw_content = tool_record.get("raw_content", {})
                    tool_call = {
                        "id": raw_content.get("call_id", "unknown") if isinstance(raw_content, dict) else "unknown",
                        "type": "function",
                        "function": {
                            "name": raw_content.get("name", "unknown") if isinstance(raw_content, dict) else "unknown",
                            "arguments": raw_content.get("arguments", "{}") if isinstance(raw_content, dict) else "{}"
                        }
                    }
                    tool_calls.append(tool_call)
                    next_index += 1
                
                # Create an assistant message without content
                formatted_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })
                item_index = next_index
                
            elif current_record.get("item_type") == "tool_call_output_item":
                # Tool execution result
                raw_content = current_record.get("raw_content", {})
                formatted_messages.append({
                    "role": "tool",
                    "content": raw_content.get("output", "") if isinstance(raw_content, dict) else "",
                    "tool_call_id": raw_content.get("call_id", "unknown") if isinstance(raw_content, dict) else "unknown"
                })
                item_index += 1
                
            else:
                # Skip other types of records
                item_index += 1
        
        return formatted_messages

    @classmethod
    def get_recent_turns_summary(cls, history_dir: Union[str, Path], session_id: str, num_turns: int = 5) -> str:
        """Get simplified summary of last N rounds for memory on context reset.
        
        Args:
            history_dir: History file directory
            session_id: Session ID
            num_turns: Number of rounds to summarize
            
        Returns:
            Formatted summary as string
        """
        history_file = Path(history_dir) / f"{session_id}_history.jsonl"
        
        if not history_file.exists():
            return "No history"
        
        # Read all records, group by turn
        records = []
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        # Sort by turn and step
        records.sort(key=lambda x: (x.get("turn", 0), x.get("in_turn_steps", 0)))
        
        # Group by turn number
        turns_data = {}
        for record in records:
            turn_num = record.get("turn", 0)
            if turn_num not in turns_data:
                turns_data[turn_num] = []
            turns_data[turn_num].append(record)
        
        # Select recent turn numbers
        recent_turn_nums = sorted(turns_data.keys())[-num_turns:] if len(turns_data) > num_turns else sorted(turns_data.keys())
        
        if not recent_turn_nums:
            return "No recent turns"
        
        summary_lines = []
        summary_lines.append(f"=== Overview of recent {len(recent_turn_nums)} turns of interaction history ===")
        
        for turn_num in recent_turn_nums:
            turn_records = turns_data[turn_num]
            summary_lines.append(f"\nTurn#{turn_num}:")
            
            for record in turn_records:
                if record.get("type") == "user_input":
                    # User input
                    content = record.get("content", "")
                    if isinstance(content, list):
                        # Handle list content
                        content_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                content_parts.append(item.get("text", ""))
                        content = " ".join(content_parts)
                    
                    formatted_content = cls._format_multiline_content(content, max_length=500)
                    summary_lines.append(f"  User:")
                    summary_lines.append(f"    {formatted_content}")
                    
                elif record.get("item_type") == "message_output_item":
                    # Agent response
                    raw_content = record.get("raw_content", {})
                    role = raw_content.get("role", "unknown")
                    
                    if role == "assistant":
                        # Extract text content
                        content_parts = []
                        for content_item in raw_content.get("content", []):
                            if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                                content_parts.append(content_item.get("text", ""))
                        content = " ".join(content_parts)
                        
                        if content.strip():  # Only display non-empty content
                            formatted_content = cls._format_multiline_content(content, max_length=500)
                            summary_lines.append(f"  Assistant:")
                            summary_lines.append(f"    {formatted_content}")
                
                elif record.get("item_type") == "tool_call_item":
                    # Tool call
                    raw_content = record.get("raw_content", {})
                    if isinstance(raw_content, dict):
                        tool_name = raw_content.get("name", "unknown")
                        call_id = raw_content.get("call_id", "unknown")
                        arguments = raw_content.get("arguments", "{}")
                        
                        # Format multi-line arg content
                        formatted_args = cls._format_multiline_content(arguments, max_length=300)
                        summary_lines.append(f"  Tool Call: {tool_name}")
                        summary_lines.append(f"    ID: {call_id}")
                        summary_lines.append(f"    Args: {formatted_args}")
                    else:
                        summary_lines.append(f"  Tool Call: unknown")
                    
                elif record.get("item_type") == "tool_call_output_item":
                    # Tool execution result
                    raw_content = record.get("raw_content", {})
                    if isinstance(raw_content, dict):
                        call_id = raw_content.get("call_id", "unknown")
                        output = raw_content.get("output", "")
                        if output.strip():
                            formatted_output = cls._format_multiline_content(output, max_length=400)
                            summary_lines.append(f"  Tool Result (ID: {call_id}):")
                            summary_lines.append(f"    {formatted_output}")
                    else:
                        summary_lines.append(f"  Tool Result: unknown")
        
        summary_lines.append("\nNote: This is a simplified overview. Please use the history record search tool to view the complete content and search information in it.")
        return "\n".join(summary_lines)
    
    @classmethod
    def _format_multiline_content(cls, content: str, max_length: int = 500) -> str:
        """Format multi-line content; handle line breaks and length limits.
        
        Args:
            content: Raw content.
            max_length: Max allowed length.
            
        Returns:
            Formatted content string.
        """
        if not content:
            return "[No content]"
        
        content = content.strip()
        
        # If content isn't long, return as-is (keep newlines)
        if len(content) <= max_length:
            lines = content.split('\n')
            if len(lines) <= 1:
                return content
            else:
                # Multiline: indent each line but the first
                formatted_lines = [lines[0]]
                for line in lines[1:]:
                    formatted_lines.append(f"    {line}")
                return '\n'.join(formatted_lines)
        
        # Content too long, try to truncate by lines first
        lines = content.split('\n')
        if len(lines) > 1:
            accumulated = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 <= max_length - 20:  # reserve room for ellipsis and notice
                    accumulated.append(line)
                    current_length += len(line) + 1
                else:
                    break
            
            if accumulated:
                result_lines = [accumulated[0]]
                for line in accumulated[1:]:
                    result_lines.append(f"    {line}")
                
                if len(accumulated) < len(lines):
                    result_lines.append(f"    ... (truncated, total {len(lines)} lines, {len(content)} chars)")
                
                return '\n'.join(result_lines)
        
        # Single-line or line truncation didn't help, do head/tail cutoff
        half_length = (max_length - 20) // 2
        truncated = content[:half_length] + " ... " + content[-half_length:]
        return f"{truncated}\n    (truncated from {len(content)} chars)"

    @classmethod
    def _format_content_with_truncation(cls, content: str, max_length: int = 500) -> str:
        """Format content, truncate if over limit.
        
        Args:
            content: Raw content
            max_length: Max total length
            
        Returns:
            Formatted content string
        """
        if not content:
            return "[No content]"
        
        content = content.strip()
        if len(content) <= max_length:
            return content
        
        # Truncation: 250 chars head + ... + 250 chars tail
        half_length = (max_length - 5) // 2  # minus " ... "
        truncated = content[:half_length] + " ... " + content[-half_length:]
        return f"(actual length: {len(content)} chars, truncated to {max_length} chars) {truncated} "

    @classmethod
    def get_session_stats(cls, history_dir: Union[str, Path], session_id: str) -> Dict[str, Any]:
        """Get session statistics."""
        history_file = Path(history_dir) / f"{session_id}_history.jsonl"
        
        if not history_file.exists():
            return {}
        
        stats = {
            "total_turns": 0,
            "total_messages": 0,
            "tool_calls": 0,
            "truncations": 0,
            "user_input_turns": 0,
            "assistant_turns": 0, # Assistant turns are assistant outputs + all their tool executions count as one
        }
        
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    stats["total_messages"] += 1
                    
                    if record.get("type") == "user_input":
                        stats["user_input_turns"] += 1
                    if record.get("item_type") == "message_output_item":
                        pass
                    elif record.get("item_type") == "tool_call_item":
                        stats["tool_calls"] += 1
                        
                except json.JSONDecodeError:
                    continue
            # stats["total_turns"] is "turn" from last line + 1 (since user input is included), and watch for 0-based index
            stats["total_turns"] = record.get("turn", 0)+1
            stats["assistant_turns"] = record.get("turn", 0)+1 - stats["user_input_turns"]
            
        
        return stats


class _ContextRun(RunHooks):
    """One invocation of the original context runner on SDK lifecycle hooks.

    The SDK owns the unabridged run items. Retained indices persist Toolathlon's
    deletions across requests and are also applied to the returned RunResult.
    """

    def __init__(self, runner, context, input, hooks):
        self.runner, self.context, self.hooks = runner, context, hooks
        self.initial_size = len(ItemHelpers.input_to_new_input_list(input))
        self.seen = self.initial_size
        self.retained = list(range(self.initial_size))
        self.wrapper = RunContextWrapper(context)
        self.agent = None
        self.server_mode = context.get("_server_conversation_tracker") is not None
        self.server_pending = []
        self.server_seen = 0

    def filter_input(self, model_data):
        items = model_data.input
        if not self.server_mode and len(items) < self.seen:
            raise RuntimeError("Unexpected SDK conversation history size")
        if self.server_mode:
            # Server-side response chaining sends only unsent items. Archive
            # the previous response plus its actual SDK-generated tool results,
            # without interpreting the delta as the full local conversation.
            if self.server_pending:
                turn_items = self.server_pending + [item for item in items if item.get("type") == "function_call_output"]
                self.record_turn(turn_items)
                self.server_seen += len(turn_items)
                self.server_pending = []
            return model_data
        new_items = items[self.seen:]
        if new_items:
            self.record_turn(new_items)
            self.retained.extend(range(self.seen, len(items)))
        self.seen = len(items)
        selected = [items[index] for index in self.retained]
        pending = self.context.get("_pending_truncate")
        if pending:
            # _handle_truncation mutates the selected list and rebases original
            # turn boundaries; keep the corresponding SDK indices for later.
            positions = {id(item): index for item, index in zip(selected, self.retained)}
            self.runner._handle_truncation(selected, [], [], pending, self.wrapper)
            self.retained = [positions[id(item)] for item in selected]
            self.context["_pending_truncate"] = None
        return ModelInputData(input=selected, instructions=model_data.instructions)

    def record_turn(self, items):
        if not items:
            return
        meta = self.context["_context_meta"]
        start = meta["mini_turns_in_current_sequence"]
        meta["boundary_in_current_sequence"].append((start, start + len(items)))
        meta["mini_turns_in_current_sequence"] += len(items)
        kinds = {"message": "message_output_item", "function_call": "tool_call_item",
                 "function_call_output": "tool_call_output_item", "reasoning": "reasoning_item"}
        self.runner._save_items_to_history(
            self.context["_session_id"], meta["current_turn"],
            [SimpleNamespace(type=kinds.get(item.get("type"), item.get("type", "message")),
                             raw_item=item) for item in items],
            self.agent.name, Path(self.context["_history_dir"]),
        )

    def finish(self, result):
        if self.server_mode:
            self.record_turn([item.to_input_item() for item in result.new_items[self.server_seen:]])
            return
        original = ItemHelpers.input_to_new_input_list(result.input)
        full = original + [item.to_input_item() for item in result.new_items]
        self.filter_input(ModelInputData(input=full, instructions=None))
        kept = set(self.retained)
        result.input = [item for index, item in enumerate(original) if index in kept]
        result.new_items = [item for index, item in enumerate(result.new_items, len(original)) if index in kept]

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self.wrapper, self.agent = context, agent
        meta = self.context["_context_meta"]
        for key in ("current_turn", "total_turns_ever", "turns_in_current_sequence"):
            meta[key] = meta.get(key, 0) + 1
        await self.hooks.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(self, context, agent, response):
        fields = ("total_tokens", "input_tokens", "output_tokens", "requests")
        self.context["_last_response_usage"] = {key: getattr(response.usage, key) for key in fields}
        self.context["_cumulative_usage"] = {key: getattr(context.usage, key) for key in fields}
        tracker = self.context.get("_server_conversation_tracker")
        if tracker is not None:
            tracker.track_server_items(response)
            self.server_pending = [item.model_dump(exclude_unset=True) for item in response.output]
        await self.hooks.on_llm_end(context, agent, response)

    async def on_agent_start(self, context, agent):
        await self.hooks.on_agent_start(context, agent)

    async def on_agent_end(self, context, agent, output):
        await self.hooks.on_agent_end(context, agent, output)

    async def on_handoff(self, context, from_agent, to_agent):
        await self.hooks.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context, agent, tool):
        await self.hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result):
        await self.hooks.on_tool_end(context, agent, tool, result)
