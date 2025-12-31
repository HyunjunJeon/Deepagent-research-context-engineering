"""Middleware to patch dangling tool calls in the messages history."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


class PatchToolCallsMiddleware(AgentMiddleware):
    """메시지 기록에서 댕글링(dangling) 도구 호출을 패치하는 미들웨어."""

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """에이전트가 실행되기 전에 AIMessage의 댕글링 도구 호출을 처리합니다."""
        messages = state["messages"]
        if not messages or len(messages) == 0:
            return None

        patched_messages = []
        # Iterate over the messages and add any dangling tool calls
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    corresponding_tool_msg = next(
                        (msg for msg in messages[i:] if msg.type == "tool" and msg.tool_call_id == tool_call["id"]),
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # We have a dangling tool call which needs a ToolMessage
                        tool_msg = (
                            f"도구 호출 {tool_call['name']} (ID: {tool_call['id']})이 취소되었습니다 - "
                            "완료되기 전에 다른 메시지가 도착했습니다."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        return {"messages": Overwrite(patched_messages)}
