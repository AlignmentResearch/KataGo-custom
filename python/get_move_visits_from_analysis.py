import argparse
import itertools
import json
from collections import Counter, namedtuple
from typing import Any

MoveTrace = namedtuple("MoveTrace", "move parent_move")


def sum_lists_elementwise(l1: list[Any], l2: list[Any]) -> list[Any]:
    """TODO comment"""
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
        return []

    move = data.get("move")
    visits = data.get("visits")
    if visits is None:
        # Handle the root node.
        visits = data["rootInfo"]["visits"]
    children_data = data.get("children")
    if children_data is None:
        # Handle the root node.
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
    description = "TODO"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "files",
        help="input files where each file must consist of a json object per line",
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
