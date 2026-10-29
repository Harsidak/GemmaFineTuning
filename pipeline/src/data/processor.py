from typing import List, Dict, Any, Optional

def format_trajectory_for_sft(
    messages: List[Dict[str, Any]], 
    system_prompt: Optional[str] = None
) -> str:
    """Formats a multi-turn agent interaction trajectory into Gemma 4 token format.

    Gemma 4 Special Tokens:
    - User Turn: <start_of_turn>user\n{content}<end_of_turn>\n
    - Model Turn: <start_of_turn>model\n{content}<end_of_turn>\n

    Tool calls within model turns are formatted cleanly with function call structures.
    """
    formatted_text = ""

    # Prepend system prompt if provided
    current_user_prefix = ""
    if system_prompt:
        current_user_prefix = f"[System Instructions]\n{system_prompt.strip()}\n\n"

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            current_user_prefix += f"{content.strip()}\n\n"
            continue

        if role == "user":
            user_content = f"{current_user_prefix}{content}".strip()
            current_user_prefix = "" # reset once prepended
            formatted_text += f"<start_of_turn>user\n{user_content}<end_of_turn>\n"

        elif role == "model" or role == "assistant":
            # If there are structured tool calls
            tool_calls = msg.get("tool_calls", [])
            model_text = content or ""
            if tool_calls:
                tool_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", "")
                    tool_lines.append(f"call:{fn_name}{{{fn_args}}}")
                if model_text:
                    model_text += "\n"
                model_text += "\n".join(tool_lines)

            formatted_text += f"<start_of_turn>model\n{model_text.strip()}<end_of_turn>\n"

        elif role == "tool":
            tool_name = msg.get("name", "tool")
            tool_output = content or ""
            # Format tool output into a user turn
            formatted_text += (
                f"<start_of_turn>user\n"
                f"[Tool Result: {tool_name}]\n{tool_output.strip()}<end_of_turn>\n"
            )

    return formatted_text
