#%%
import argparse
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable
import plotly.graph_objects as go
from collections import defaultdict
#%%

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
class MCTSNode:
    """Indicates the move of an MCTS node and the node's parent."""
    move: str | None
    node_id: int
    parent_node_id: int


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
    data: dict[str, Any],  node_id_generator: Iterable[int] = itertools.count(), parent_node_id: int | None = None) -> list[Counter[MCTSNode]]:
    """Returns visit counts for each move in the MCTS tree.

    Args:
        data: KataGo `analysis` JSON output describing an MCTS tree
        parent_move: The move used to reach the top-level node described by
          `data`.
        node_id_generator: Iterator generating unique integers.

    Returns:
        A list L where L[i] contains depth-i moves and their associated visit
        counts.
    """
    if "isSymmetryOf" in data:
        # Skip nodes that were not actually searched and instead had all of
        # its stats copied from another node with a symmetric move. If we don't
        # skip these nodes then we may see the total visit count across a depth
        # level increase as we go deeper down the MCTS tree.
        return []

    move = data.get("move")
    visits = data.get("visits")
    children_data = data.get("children")
    node_id = node_id_generator.__next__()

    # Handle the root node, which has different field names.
    if visits is None:
        visits = data["rootInfo"]["visits"]
    if children_data is None:
        children_data = data["moveInfos"]

    child_stats = []
    for child_data in children_data:
        child_stats = sum_lists_elementwise(
            child_stats, get_move_visit_stats(child_data, node_id_generator=node_id_generator, parent_node_id=node_id)
        )

    stats = [
        Counter({MCTSNode(move=move, node_id=node_id, parent_node_id=parent_node_id): visits})
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

#%%
label = ["root"]
node_id_to_index = {"root": 0}
x, y = [0.0], [0.5]
target_counts = defaultdict(int)
other_counts = [0 for _ in range(len(visit_stats) - 1)]
source, target, value = [], [], []
max_depth = len(visit_stats) - 1
THRESHOLD = 3000
for depth, depth_stats in enumerate(visit_stats[1:]):
    label.append(f"{depth}: other")
    node_id_to_index[f"{depth}: other"] = len(label) - 1
    x.append((depth + 1)/ (max_depth + 0.1))
    y.append(0.0001)

    for index, (node, count) in enumerate(depth_stats.items()):
        if node.node_id not in node_id_to_index:
            label.append(f"{depth}: {node.move}")
            node_id_to_index[node.node_id] = len(label) - 1
            x.append((depth + 1)/ (max_depth + 0.1))
            y.append((index + 1) / (len(depth_stats) + 3))

    for node, count in depth_stats.items():
        target_counts[node.node_id] += count

    for node, count in depth_stats.items():
        node_id = node.node_id
        parent_node_id = node.parent_node_id
        if parent_node_id == 0:
            parent_node_id = "root"
        elif target_counts[parent_node_id] <= THRESHOLD:
            parent_node_id = f"{depth - 1}: other"
        if target_counts[node_id] <= THRESHOLD:
            node_id = f"{depth}: other"

        source.append(node_id_to_index[parent_node_id])
        target.append(node_id_to_index[node_id])
        value.append(count)

fig = go.Figure(data=[go.Sankey(
    # arrangement = "freeform",
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = label,
    #   x = x,
    #   y = y,
      color = "blue"
    ),
    link = dict(
      source = source,
      target = target,
      value = value
  ))])

fig.update_layout(title_text=input_files[0], font_size=10, height=800, width=1500)
fig.show()
