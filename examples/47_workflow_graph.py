# Demo 47 — Workflow Graph with a Review Loop
# Execute: uv run python examples/47_workflow_graph.py
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment or .env.

import asyncio
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from anycode import END, START, Agent, Workflow
from anycode.workflow import add

load_dotenv()

MAX_ROUNDS = 3


class ReviewState(BaseModel):
    """State threaded through every node. Frozen — nodes return patches, never mutate."""

    model_config = ConfigDict(frozen=True)

    topic: str = ""
    draft: str = ""
    critique: str = ""
    rounds: int = 0
    tokens: Annotated[int, add] = 0
    log: Annotated[list[str], add] = []


writer = Agent(
    name="writer",
    instructions="You write tight technical blurbs. Two sentences, no preamble, no markdown headings.",
    tools=[],
    max_turns=2,
)

critic = Agent(
    name="critic",
    instructions=(
        "You are a demanding technical editor. If the blurb is accurate and tight, reply with exactly APPROVED. "
        "Otherwise reply with one sentence naming the single biggest problem."
    ),
    tools=[],
    max_turns=2,
)

workflow = Workflow(ReviewState, name="write-review")


@workflow.node
async def write(state: ReviewState) -> dict:
    revision = f"\n\nPrevious critique to address: {state.critique}" if state.critique else ""
    result = await writer.run(f"Write a two-sentence blurb about: {state.topic}{revision}")
    return {
        "draft": result.output.strip(),
        "rounds": state.rounds + 1,
        "tokens": result.token_usage.input_tokens + result.token_usage.output_tokens,
        "log": [f"round {state.rounds + 1}: drafted {len(result.output)} chars"],
    }


@workflow.node
async def review(state: ReviewState) -> dict:
    result = await critic.run(f"Topic: {state.topic}\n\nBlurb:\n{state.draft}")
    verdict = result.output.strip()
    return {
        "critique": verdict,
        "tokens": result.token_usage.input_tokens + result.token_usage.output_tokens,
        "log": [f"round {state.rounds}: critic said {verdict[:40]}"],
    }


def gate(state: ReviewState) -> str:
    """Stop when the critic approves or the round budget runs out."""
    if "APPROVED" in state.critique.upper() or state.rounds >= MAX_ROUNDS:
        return END
    return "write"


workflow.add_edge(START, "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edge("review", gate)


async def main() -> None:
    app = workflow.compile()

    print("Graph\n" + "=" * 55)
    print(app.to_mermaid())

    print("\n\nStreaming the run\n" + "=" * 55)
    final = None
    async for event in app.stream(ReviewState(topic="why vector databases need hybrid search"), max_steps=8):
        if event.type == "node_start":
            print(f"  -> {event.node}")
        elif event.type == "route":
            print(f"     {event.node} routes to {', '.join(event.targets)}")
        elif event.type == "done":
            final = event.result

    assert final is not None
    print("\nFinal draft\n" + "=" * 55)
    print(final.state.draft)
    print("=" * 55)
    print(f"\nSucceeded:   {final.success}")
    print(f"Rounds:      {final.state.rounds}")
    print(f"Steps:       {final.steps}")
    print(f"Path:        {' -> '.join(final.path)}")
    print(f"Stop reason: {final.stop_reason.code if final.stop_reason else 'reached END'}")
    print(f"Tokens:      {final.state.tokens} (summed by the `add` reducer on ReviewState.tokens)")
    print("\nAccumulated log (the `add` reducer appends instead of replacing):")
    for line in final.state.log:
        print(f"  - {line}")


if __name__ == "__main__":
    asyncio.run(main())
