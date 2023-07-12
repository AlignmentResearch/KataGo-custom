import argparse
import itertools
import json
from collections import Counter, namedtuple
from typing import Any

MoveTrace = namedtuple("MoveTrace", "move parent_move")


def sum_elementwise(l1, l2, make_zero):
    """TODO comment and type"""
    # TODO is make_zero lambda necessary vs just zero?
    return [x + y for x, y in itertools.zip_longest(l1, l2, fillvalue=make_zero())]


def sum_counters(l1, l2):
    """TODO comment and type"""
    return sum_elementwise(l1, l2, lambda: Counter())


# def get_move_visit_stats(data: dict[str, Any]) -> list[Counter[tuple(str | None), int]]:
def get_move_visit_stats(data: dict[str, Any]):
    """TODO comment, describe return val"""
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
        child_stats = sum_counters(child_stats, get_move_visit_stats(child_data))

    stats = [Counter({MoveTrace(move=move, parent_move=None): visits})] + child_stats
    if len(stats) > 1:
        stats[1] = Counter(
            {
                MoveTrace(move=child_move, parent_move=move): child_visits
                for (child_move, _), child_visits in stats[1].items()
            }
        )
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

                print(
                    f"\nID={data['id']} turn={data['turnNumber']} ####################"
                )

                visit_stats = get_move_visit_stats(data)
                for depth, depth_stats in enumerate(visit_stats):
                    total_count = 0
                    for count in depth_stats.values():
                        total_count += count
                    print(f"Depth {depth}: visits={total_count} ===========")
                    print(depth_stats)
