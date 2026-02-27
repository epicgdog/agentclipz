import json
import time
from typing import Any, Dict, List, Set

from yutori import YutoriClient


POLL_INTERVAL_SECONDS = 5
OUTPUT_MD_PATH = "streamer_context.md"


def wait_for_task_completion(client: YutoriClient, task_id: str) -> Dict[str, Any]:
    """
    Poll the Yutori Research API until the task completes and return the final result.
    """
    seen_update_ids: Set[str] = set()

    while True:
        result = client.research.get(task_id)
        status = result.get("status")

        print(f"[yutori] Task {task_id} status: {status}")

        # Log any new research updates as they arrive so you can see
        # what the agent is discovering over time.
        updates = result.get("updates") or []
        for update in updates:
            update_id = update.get("id")
            if update_id and update_id in seen_update_ids:
                continue
            if update_id:
                seen_update_ids.add(update_id)

            content = (update.get("content") or "").strip()
            timestamp = update.get("timestamp")
            print("\n[yutori] --- New research update ---")
            if timestamp is not None:
                print(f"[yutori] Timestamp: {timestamp}")
            if content:
                print(content)

            citations = update.get("citations") or []
            if citations:
                print("[yutori] Citations (sample):")
                for citation in citations[:5]:
                    url = citation.get("url")
                    if url:
                        print(f"- {url}")

        if status in ("succeeded", "failed"):
            return result

        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> None:
    # YutoriClient will use (in order):
    # 1. Explicit api_key argument (not used here)
    # 2. YUTORI_API_KEY environment variable, if set
    # 3. Credentials saved via `yutori auth login`
    client = YutoriClient()

    query = (
        "Do deep web research on the streamer 'JasonTheWeen'. "
        "Include platforms, links, typical content, style and tone, schedule (if available), "
        "notable moments or clips, community vibe, and any other relevant context for an AI assistant "
        "that will be helping this streamer. "
        "Put extra emphasis on RECENT trends, memes, viral clips, collaborations, and changes in content "
        "from roughly the last 6–12 months, but still include evergreen background context."
    )

    print(f"[yutori] Creating research task for query: {query}")

    task = client.research.create(
        query=query,
    )

    task_id = task["task_id"]
    print(f"[yutori] Created research task with id: {task_id}")

    final_result = wait_for_task_completion(client, task_id)

    print("[yutori] Final task result (JSON):")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))

    # Turn the result into a markdown context file focused on the streamer.
    result_markdown: str = (final_result.get("result") or "").strip()
    updates: List[Dict[str, Any]] = final_result.get("updates") or []

    lines: List[str] = []
    lines.append("# Streamer Research: JasonTheWeen")
    lines.append("")
    lines.append("Generated via Yutori Research API.")
    lines.append("")

    if result_markdown:
        lines.append("## Overview & Summary")
        lines.append("")
        lines.append(result_markdown)
        lines.append("")

    if updates:
        lines.append("## Detailed Updates")
        lines.append("")
        for idx, update in enumerate(updates, start=1):
            content = (update.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"### Update {idx}")
            lines.append("")
            lines.append(content)
            lines.append("")

            citations = update.get("citations") or []
            if citations:
                lines.append("**Citations (sample):**")
                for citation in citations[:5]:
                    url = citation.get("url")
                    if url:
                        lines.append(f"- {url}")
                lines.append("")

    if not result_markdown and not updates:
        lines.append("_No result content returned by the API._")
        lines.append("")

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[yutori] Saved streamer markdown context to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()

