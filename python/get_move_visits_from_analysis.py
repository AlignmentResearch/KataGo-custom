import argparse
import itertools
import json
from collections import Counter, namedtuple
from dataclasses import dataclass
from typing import Any

DESCRIPTION = f"""Given output JSONs from KataGo's `analysis` command, prints \
the moves visited by MCTS and how many visits each move received. \
`includeTree` must be set to `true` in the input JSONs to `analysis`.

Example usage:
```
    # Run KataGo `analysis` and pipe its output to a file:
    /engines/KataGo-custom/cpp/katago analysis \\
      -analysis-threads 10 \\
      -model /nas/ucb/ttseng/go_attack/victims/kata1-b40c256-s11840935168-d2898845681.bin.gz \\
      -config /go_attack/configs/gtp-raw.cfg \\
      2>&1 | tee /tmp/out.txt
    # Give an input JSON:
    {{"id":"foo","moves":[["B","Q16"],["W","Q4"],["B","D4"]],"rules":"tromp-taylor","komi":6.5,"boardXSize":19,"boardYSize":19,"analyzeTurns":[3,4],"includeTree":true}}
    # Analyze the output with this script:
    python {__file__} /tmp/out.txt
```
To convert an SGF file to an `analysis` input JSON, use `sgf_to_analysis_input.py`.
"""


@dataclass(frozen=True)
class MoveTrace:
    """Indicates the move of an MCTS node and the move of the node's parent."""
    move: str | None
    parent_move: str | None


def sum_lists_elementwise(l1: list[Any], l2: list[Any]) -> list[Any]:
    """Sums two lists element-wise."""
    result = []
    for x, y in itertools.zip_longest(l1, l2):
        if x is None:
            result.append(y)
        elif y is None:
            result.append(x)
        else:
            result.append(x + y)
    return result


def get_move_visit_stats(
    data: dict[str, Any], parent_move: str | None = None
) -> list[Counter[MoveTrace]]:
    """TODO comment, describe return val"""
    if "isSymmetryOf" in data:
        # Skip nodes that were not actually searched and instead had all of
        # its stats copied from another node with a symmetric move. If we don't
        # skip these nodes then we may see the total visit count across a depth
        # level increase as we go deeper down the MCTS tree.
        return []

    move = data.get("move")
    visits = data.get("visits")
    children_data = data.get("children")

    # Handle the root node, which has different field names.
    if visits is None:
        visits = data["rootInfo"]["visits"]
    if children_data is None:
        children_data = data["moveInfos"]

    child_stats = []
    for child_data in children_data:
        child_stats = sum_lists_elementwise(
            child_stats, get_move_visit_stats(child_data, parent_move=move)
        )

    stats = [
        Counter({MoveTrace(move=move, parent_move=parent_move): visits})
    ] + child_stats
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "files",
        help="input files where each file consists of a JSON object per line",
        nargs="+",
    )
    args = parser.parse_args()

    input_files = args.files

    for filename in input_files:
        print(f"\nFile {filename}")
        with open(filename) as f:
            for json_line in f:
                rollout_data = []
                data = json.loads(json_line)

                print(f"\nID={data['id']} turn={data['turnNumber']}")

                visit_stats = get_move_visit_stats(data)
                for depth, depth_stats in enumerate(visit_stats):
                    total_count = 0
                    for count in depth_stats.values():
                        total_count += count
                    print(f"Depth {depth}: total visits={total_count} ===========")
                    print(depth_stats)
