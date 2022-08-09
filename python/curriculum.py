#!/usr/bin/python3

"""Curriculum module for using in victimplay."""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from sgfmill import sgf

Config = Mapping[str, Any]


@dataclass(frozen=True)
class AdvGameInfo:
    """Class for storing game result from the adversary perspective."""

    victim: str
    game_hash: str
    winner: Optional[bool]
    diff_score: float
    diff_score_wo_komi: float


@dataclass(frozen=True)
class PlayerStat:
    """Class for storing game statistics.

    Statistics either represents the adversary
    or stores victim changing criteria.
    For victim only one criteria can be enabled, all others should be None.
    """

    name: Optional[str] = None
    win_rate: Optional[float] = None
    score_diff: Optional[float] = None
    score_wo_komi_diff: Optional[float] = None
    policy_loss: Optional[float] = None

    def get_stat_members(self) -> Dict[str, float]:
        d = asdict(self)
        del d["name"]
        return d

    def can_be_victim_criteria(self) -> bool:
        criteria = self.get_stat_members()
        num_enabled = len([v for k, v in criteria.items() if v is not None])
        return num_enabled == 1

    # check if adv_stat has a greater value of enabled criteria
    def check_if_gt(self, adv_stat) -> bool:
        criteria = self.get_stat_members()
        adv_vals = adv_stat.get_stat_members()
        for k, v in criteria.items():
            if v is not None:
                logging.info(
                    "{}: {} (adv) <-> {} (threshold)".format(k, adv_vals[k], v),
                )
                if adv_vals[k] > v:
                    return True
        return False


def get_game_info(sgf_str: str) -> Optional[AdvGameInfo]:
    sgf_game = sgf.Sgf_game.from_string(sgf_str)

    b_name = sgf_game.get_player_name("b")
    w_name = sgf_game.get_player_name("w")

    if "victim-" in b_name and "__" not in b_name:
        # it has 'victim' in its name and it is not a colored evaluator
        victim_name = b_name
    elif "victim-" in w_name and "__" not in w_name:
        victim_name = w_name
    else:
        logging.warning("No player with 'victim-' prefix: '%s'", sgf_str)
        return None

    victim_color = {b_name: "b", w_name: "w"}[victim_name]
    adv_color = {"b": "w", "w": "b"}[victim_color]

    # adv_raw_name = {"b": b_name, "w": w_name}[adv_color]

    #  currently the model name will not be just 'victim'
    #  this code was NOT tested so commented out yet
    #  adv_name = (
    #    adv_raw_name.split('__' + victim_name)[0]
    #    if adv_color == "b"
    #    else adv_raw_name.split(victim_name + '__')[1]
    #  )
    #  adv_steps = (
    #    0
    #    if adv_name == "random"
    #    else int(re.search(r"\-s([0-9]+)\-", adv_name).group(1))
    #  )

    win_color = sgf_game.get_winner()
    lose_color = {"b": "w", "w": "b", None: None}[win_color]
    komi = sgf_game.get_komi()
    adv_komi = {"w": komi, "b": -komi}[adv_color]

    win_score = 0
    result = "undefined"
    game_hash = None
    try:
        game_root = sgf_game.get_root()
        game_c = game_root.get("C")
        game_hash = game_c.split(",")[2].split("=")[1]

        result = game_root.get("RE")
        win_score = result.split("+")[1]
        win_score = float(win_score)
    except KeyError:
        logging.warning("No result (RE tag) present in SGF game: '%s'", sgf_str)
        return None
    except IndexError:
        logging.warning("No winner in result '%s'", result)
        return None
    except ValueError:
        logging.warning("Game score is not numeric: '%s'", win_score)
        return None

    if game_hash is None:
        logging.warning("Game hash is None!")
        return None

    if win_color is None:
        adv_minus_victim_score = 0
        adv_minus_victim_score_wo_komi = 0
    else:
        adv_minus_victim_score = {
            win_color: win_score,
            lose_color: -win_score,
        }[adv_color]
        adv_minus_victim_score_wo_komi = adv_minus_victim_score - adv_komi

    winner = None
    if win_color is not None:
        winner = win_color == adv_color

    assert victim_name.startswith("victim-")
    victim_name = victim_name[7:]
    return AdvGameInfo(
        victim_name,
        game_hash,
        winner,
        adv_minus_victim_score,
        adv_minus_victim_score_wo_komi,
    )


def get_files_sorted_by_modification_time(
    folder: str,
    extension: Optional[str] = None,
) -> Sequence[str]:
    all_sgfs = []
    for path, dirnames, filenames in os.walk(folder, followlinks=True):
        for f in filenames:
            if extension is None or os.path.splitext(f)[1] == extension:
                file_path = os.path.join(path, f)
                all_sgfs.append([file_path, os.path.getmtime(file_path)])
    all_sgfs.sort(key=lambda x: x[1], reverse=True)
    all_sgfs = [x[0] for x in all_sgfs]
    return all_sgfs


def recompute_statistics(
    games: List[AdvGameInfo],
    games_for_compute: int,
    current_victim_name: str,
) -> Optional[PlayerStat]:
    # don't have enough data
    if len(games) < games_for_compute:
        logging.info("Incomplete statistics, got only {} games".format(len(games)))
        return None

    sum_wins = 0
    sum_ties = 0
    sum_score = 0
    sum_score_wo_komi = 0
    logging.info("Computing {} games".format(len(games)))
    games_cur_victim = [g for g in games if g.victim == current_victim_name]
    if len(games_cur_victim) < len(games):
        logging.info(
            "Incomplete statistics for current victim, got only {} games".format(
                len(games_cur_victim),
            ),
        )
        return None

    for game in games:
        # game.winner can be None (for ties), but a tie is still not a win
        if game.winner:
            sum_wins += 1
        elif game.winner is None:
            sum_ties += 1
        sum_score += game.diff_score
        sum_score_wo_komi += game.diff_score_wo_komi

    logging.info(
        "Got {} wins and {} ties from {} games".format(sum_wins, sum_ties, len(games)),
    )
    win_rate = float(sum_wins) / len(games)
    mean_diff_score = float(sum_score) / len(games)
    mean_diff_score_wo_komi = float(sum_score_wo_komi) / len(games)

    return PlayerStat(
        win_rate=win_rate,
        score_diff=mean_diff_score,
        score_wo_komi_diff=mean_diff_score_wo_komi,
    )


class Curriculum:
    """Curriculum object.

    Curriculum is used for updating victims for victimplay based on
    the criteria specified in the provided config.
    """

    def __init__(
        self,
        victims_input_dir: str,
        victims_output_dir: str,
        config: Optional[Sequence[Config]] = None,
        config_json: Optional[str] = None,
        config_json_file: Optional[str] = None,
    ):
        """Initial curriculum setup.

        Construct and initialize curriculum.

        @param victims_input_dir: The folder with all victim model
            files specified in the config.
        @param victims_output_dir: The folder where we copy victims for selfplay.
        @param config: List of victims.
        @param config_json: Serialized JSON list of victims.
        @param config_json_file: JSON file with list of victims.
        """
        self.MAX_VICTIM_COPYING_EFFORTS = 10
        self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT = 10

        self.stat_files = []
        self.sgf_games = []
        self.game_hashes = dict()

        if config_json_file is not None:
            logging.info("Curriculum: loading JSON config from '%s'", config_json_file)
            with open(config_json_file) as f:
                config = json.load(f)
        elif config_json is not None:
            logging.info("Curriculum: loading JSON config from a string")
            config = json.loads(config_json)
        elif config is not None:
            logging.info("Using python list as a config")

        if not config:
            raise ValueError("Empty config for the curriculum play!")

        self.victims_input_dir = victims_input_dir
        self.victims_output_dir = victims_output_dir

        self.victim_idx = 0
        self.finished = False
        self.victims = []
        for line in config:
            cond = PlayerStat(
                name=line["name"],
                win_rate=line["win_rate"],
                score_diff=line["diff_score"],
                score_wo_komi_diff=line["diff_score_wo_komi"],
                policy_loss=line["policy_loss"],
            )
            if not cond.can_be_victim_criteria():
                raise ValueError(
                    "Incorrect victim change criteria for victim '{}': "
                    "exactly one value should be non-None".format(line["name"]),
                )
            self.victims.append(cond)

        logging.info("Loaded curriculum with the following params:")
        logging.info("\n".join([str(x) for x in config]))

        logging.info("Finding the latest victim...")
        victim_files = get_files_sorted_by_modification_time(self.victims_output_dir)
        if victim_files:
            try:
                fname = os.path.basename(victim_files[0])
                self.victim_idx = [v.name for v in self.victims].index(fname)
            except ValueError:
                logging.warning(
                    "Victim '{}' is not found in '{}'".format(
                        victim_files[0],
                        self.victims_output_dir,
                    ),
                )

        logging.info(
            "Copying the latest victim '{}'...".format(self._cur_victim.name),
        )
        self.__try_victim_copy()
        logging.info("Curriculum initial setup is complete")

    @property
    def _cur_victim(self) -> PlayerStat:
        return self.victims[self.victim_idx]

    def __try_victim_copy(self, force_if_exists=False):
        num_efforts = 0
        victim_name = self._cur_victim.name
        victim_path = os.path.join(self.victims_output_dir, victim_name)
        if not force_if_exists and os.path.exists(victim_path):
            return
        for _ in range(self.MAX_VICTIM_COPYING_EFFORTS):
            try:
                shutil.copy(
                    os.path.join(self.victims_input_dir, victim_name),
                    self.victims_output_dir,
                )
                return
            except OSError:
                logging.warning(
                    "Cannot copy victim '{}', maybe "
                    "filesystem problem? Waiting {} sec...".format(
                        self._cur_victim.name,
                        self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT,
                    ),
                )
                num_efforts += 1
                time.sleep(self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT)

        raise RuntimeError(
            "Problem copying victim '{}', curriculum stopped".format(
                self._cur_victim.name,
            ),
        )

    def try_move_on(
        self,
        adv_stat: Optional[PlayerStat] = None,
        policy_loss: Optional[float] = None,
    ):
        if self.finished:
            return

        logging.info("Checking whether we need to move to the next victim...")
        want_victim_update = False
        if adv_stat is not None and self._cur_victim.check_if_gt(adv_stat):
            want_victim_update = True
        if policy_loss is not None:
            raise NotImplementedError("Policy loss check is not implemented yet")

        if not want_victim_update:
            return

        self.victim_idx += 1
        if self.victim_idx == len(self.victims):
            self.finished = True
            return

        logging.info("Moving to the next victim '{}'".format(self._cur_victim.name))
        self.__try_victim_copy(True)

    def update_sgf_games(self, selfplay_dir: str, games_for_compute: int):
        all_sgfs = get_files_sorted_by_modification_time(selfplay_dir, ".sgfs")

        useful_files = set()
        cur_games = []
        for sgf_file in all_sgfs:
            if sgf_file not in self.game_hashes:
                self.game_hashes[sgf_file] = set()

            with open(sgf_file) as f:
                logging.debug("Processing SGF file '{}'".format(sgf_file))
                all_lines = list(f.readlines())

                for line in reversed(all_lines):
                    sgf_string = line.strip()
                    game_stat = get_game_info(sgf_string)
                    if game_stat is None:
                        continue

                    # game hash was found, so consider that the rest of them are older
                    # so stop scanning this file
                    if game_stat.game_hash in self.game_hashes[sgf_file]:
                        break

                    self.game_hashes[sgf_file].add(game_stat.game_hash)
                    cur_games.append(game_stat)
                    useful_files.add(sgf_file)

        # now have cur_games sorted from newer to older
        logging.info(
            "Got {} new games from {} files".format(len(cur_games), len(useful_files)),
        )
        for f in useful_files:
            logging.info("Useful SGF file: '{}'".format(str(f)))

        # insert new games in the beginning
        self.sgf_games[:0] = cur_games

        # leave only games_for_compute games for statistics computation
        # so delete some old games
        if len(self.sgf_games) > games_for_compute:
            del self.sgf_games[games_for_compute:]

    """
    Run curriculum checking.
    @param selfplay_dir: Folder with selfplay results.
    @param games_for_compute: Number of games to compute statistics.
    @param checking_periodicity: Checking interval in seconds.
    """

    def checking_loop(
        self,
        selfplay_dir: str,
        games_for_compute: int,
        checking_periodicity: int,
    ):
        logging.info("Starting curriculum loop")
        while True:
            self.update_sgf_games(selfplay_dir, games_for_compute)
            adv_stat = recompute_statistics(
                self.sgf_games,
                games_for_compute,
                self._cur_victim.name,
            )
            if adv_stat is not None:
                self.try_move_on(adv_stat=adv_stat)
                if self.finished:
                    logging.info("Curriculum is done. Stopping")
                    break
            logging.info(
                "Curriculum is alive, current victim : {}".format(
                    self._cur_victim.name,
                ),
            )
            time.sleep(checking_periodicity)


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename="/outputs/curriculum.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser(
        description="Run victim replacement based on win rate.",
    )
    parser.add_argument(
        "-selfplay-dir",
        required=True,
        help="Directory with selfplay data",
    )
    parser.add_argument(
        "-input-models-dir",
        required=True,
        help="Input dir with victim model files",
    )
    parser.add_argument(
        "-output-models-dir",
        required=True,
        help="Output dir for adding new victims",
    )
    parser.add_argument(
        "-games-for-compute",
        type=int,
        required=False,
        default=1000,
        help="Number of last games for statistics computation",
    )
    parser.add_argument(
        "-checking-periodicity",
        type=int,
        required=False,
        default=60,
        help="Statistics computation periodicity in seconds",
    )
    parser.add_argument(
        "-config-json-string",
        required=False,
        help="Curriculum JSON config with " "victims sequence (JSON content)",
    )
    parser.add_argument(
        "-config-json-file",
        default="configs/curriculum_conf.json",
        help="Curriculum JSON config with " "victims sequence (JSON file path)",
    )

    args = parser.parse_args()

    if args.config_json_file is not None:
        curriculum = Curriculum(
            args.input_models_dir,
            args.output_models_dir,
            config_json_file=args.config_json_file,
        )
    elif args.config_json_string is not None:
        curriculum = Curriculum(
            args.input_models_dir,
            args.output_models_dir,
            config_json=args.config_json_string,
        )
    else:
        raise ValueError(
            "Curriculum: either path to JSON config or "
            "JSON config string must be provided",
        )

    curriculum.checking_loop(
        args.selfplay_dir,
        args.games_for_compute,
        args.checking_periodicity,
    )

    logging.info("Curriculum finished!")
