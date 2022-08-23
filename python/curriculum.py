#!/usr/bin/python3

"""Curriculum module for using in victimplay."""

import argparse
import json
import logging
import operator
import os
import pathlib
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

    Statistics is being represented from the adversary perspective.
    """

    win_rate: Optional[float] = None
    score_diff: Optional[float] = None
    score_wo_komi_diff: Optional[float] = None
    policy_loss: Optional[float] = None
    window_size: Optional[int] = None

    def members(self):
        d = asdict(self)
        del d["window_size"]
        return d

    def count(self):
        return len([v for k, v in self.members().items() if v is not None])


@dataclass
class VictimCriteria:
    """Criteria for the victim change.

    Victim is represented by the model name,
    moving forward/backward criteria and max visits.
    For victim only one criterion can be enabled, all others should be None.
    This dataclass cannot be frozen anymore since it modifies
    forward/backward types after __init__, a workaround for proper
    constructing the dataclass from JSON.
    Probably later python will add this functionality out-of-the-box.
    """

    name: str
    forward: PlayerStat
    backward: Optional[PlayerStat] = None
    max_visits_victim: Optional[int] = None
    max_visits_adv: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.forward, dict):
            self.forward = PlayerStat(**self.forward)
        if isinstance(self.backward, dict):
            self.backward = PlayerStat(**self.backward)

    def validate(self):
        if self.name is None:
            raise ValueError("Empty victim name in the config!")
        if self.forward is None:
            raise ValueError("Empty forward criteria in the config!")
        if self.forward.count() != 1:
            raise ValueError(
                f"Exactly one forward criterion must be non-None for '{self.name}'",
            )
        if self.backward is not None and self.backward.count() != 1:
            raise ValueError(
                f"Exactly one backward criterion must be non-None for '{self.name}'",
            )

    def __check_criterion(self, adv_stat: PlayerStat, log_prefix: str, relation):
        criteria = self.forward.members()
        adv_vals = adv_stat.members()
        for k, v in criteria.items():
            if v is not None:
                logging.info(
                    f"{log_prefix}: {k}: {adv_vals[k]} (adv) <-> {v} (threshold)",
                )
                if relation(adv_vals[k], v):
                    return True
        return False

    # check if adv_stat has a greater value of enabled criteria
    def want_forward(self, adv_stat: PlayerStat) -> bool:
        return self.__check_criterion(adv_stat, "Forward", operator.gt)

    def want_backward(self, adv_stat: PlayerStat) -> bool:
        return self.__check_criterion(adv_stat, "Backward", operator.lt)

    def __eq__(self, other):
        """Check current victim against the latest parameters.

        Parameters can be either VictimCriteria or a dict.
        """
        members = ["name", "max_visits_victim", "max_visits_adv"]
        d0 = asdict(self)
        if isinstance(other, VictimCriteria):
            d1 = asdict(other)
        else:
            d1 = other

        try:
            for m in members:
                if d0[m] != d1[m]:
                    return False
            return True
        except TypeError:
            logging.warning("Incorrect RHS in VictimCriteria ==")
        except KeyError:
            logging.warning("Missed RHS key in VictimCriteria ==")
        return False


def get_game_info(sgf_str: str) -> Optional[AdvGameInfo]:
    try:
        sgf_game = sgf.Sgf_game.from_string(sgf_str)
    except IndexError:
        logging.warning("Error parsing game: '%s'", sgf_str)
        return None

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
    folder: pathlib.Path,
    extension: Optional[str] = None,
    ignore_extensions: Optional[List[str]] = None,
) -> Sequence[str]:
    all_files = []
    for path, dirnames, filenames in os.walk(folder, followlinks=True):
        for f in filenames:
            ext = os.path.splitext(f)[1]
            if ignore_extensions is not None and ext in ignore_extensions:
                continue
            if extension is None or os.path.splitext(f)[1] == extension:
                file_path = os.path.join(path, f)
                all_files.append([file_path, os.path.getmtime(file_path)])
    # sort from newest to oldest
    all_files.sort(key=lambda x: x[1], reverse=True)

    # leave file names only
    all_files = [x[0] for x in all_files]
    return all_files


def recompute_statistics(
    games: List[AdvGameInfo],
    games_for_compute: int,
    current_victim_name: str,
) -> Optional[PlayerStat]:
    # don't have enough data
    if len(games) < games_for_compute:
        logging.info(f"Incomplete statistics, got only {len(games)} games")
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

    # take only required amount of the newest games
    if len(games) > games_for_compute:
        games = games[:games_for_compute]

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
        victims_input_dir: pathlib.Path,
        victims_output_dir: pathlib.Path,
        config: Optional[Sequence[Config]] = None,
        config_json: Optional[str] = None,
        config_json_file: Optional[pathlib.Path] = None,
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
        self.SELFPLAY_CONFIG_OVERRIDE_NAME = "victim.cfg"

        self.stat_files = []
        self.sgf_games = []
        self.game_hashes = dict()

        if config_json_file is not None:
            logging.info(f"Curriculum: loading JSON config from '{config_json_file}'")
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
        self.selfplay_config_override_path = os.path.join(
            self.victims_output_dir,
            self.SELFPLAY_CONFIG_OVERRIDE_NAME,
        )
        self.victims_output_dir_tmp = victims_output_dir.with_name(
            victims_output_dir.name + "_tmp",
        )

        self.victim_idx = 0
        self.finished = False
        self.victims: List[VictimCriteria] = []
        for line in config:
            cond = VictimCriteria(**line)
            cond.validate()
            self.victims.append(cond)

        logging.info("Loaded curriculum with the following params:")
        logging.info("\n".join([str(x) for x in config]))

        logging.info("Finding the latest victim...")
        victim_files = get_files_sorted_by_modification_time(
            self.victims_output_dir,
            ignore_extensions=[".cfg", ".conf"],
        )
        if victim_files:
            last_victim_name = os.path.basename(victim_files[0])
            victim_params = {
                "name": last_victim_name,
                "max_visits_victim": None,
                "max_visits_adv": None,
            }

            # find current maxVisits settings
            if os.path.exists(self.selfplay_config_override_path):
                with open(self.selfplay_config_override_path) as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        name, val = line.split("=")
                        if name == "maxVisits0":
                            victim_params["max_visits_victim"] = int(val)
                        elif name == "maxVisits1":
                            victim_params["max_visits_adv"] = int(val)

            # determine current victim-with-max-visits index
            victim_found = False
            for cur_idx in range(len(self.victims)):
                if self.victims[cur_idx] == victim_params:
                    self.victim_idx = cur_idx
                    victim_found = True
                    break

            if not victim_found:
                logging.warning(
                    "Victim '{}' is not found in '{}', starting from scratch".format(
                        str(victim_params),
                        self.victims_output_dir,
                    ),
                )

        logging.info(
            "Copying the latest victim '{}'...".format(self._cur_victim),
        )
        self.__try_victim_copy()
        logging.info("Curriculum initial setup is complete")

    @property
    def _cur_victim(self) -> VictimCriteria:
        return self.victims[self.victim_idx]

    def __update_victim_config(self):
        tmp_path = self.victims_output_dir_tmp / self.SELFPLAY_CONFIG_OVERRIDE_NAME
        with open(tmp_path, "w") as f:
            if self._cur_victim.max_visits_victim is not None:
                f.write(f"maxVisits0={self._cur_victim.max_visits_victim}")
            if self._cur_victim.max_visits_adv is not None:
                f.write(f"maxVisits1={self._cur_victim.max_visits_adv}")
        shutil.move(str(tmp_path), self.selfplay_config_override_path)

    def __try_victim_copy(self, force_if_exists=False):
        victim_name = self._cur_victim.name
        victim_path = self.victims_output_dir / victim_name
        victim_path_tmp = self.victims_output_dir_tmp / victim_name

        if not force_if_exists and os.path.exists(victim_path):
            return

        # Attempt to copy
        for _ in range(self.MAX_VICTIM_COPYING_EFFORTS):
            try:
                # Make sure directories exist
                os.makedirs(self.victims_output_dir, exist_ok=True)
                os.makedirs(self.victims_output_dir_tmp, exist_ok=True)
                self.__update_victim_config()

                # We copy to a tmp directory then move to make the overall
                # operation atomic, which is needed to avoid race conditions
                # with the C++ code.
                shutil.copy(
                    self.victims_input_dir / victim_name,
                    victim_path_tmp,
                )
                shutil.move(str(victim_path_tmp), victim_path)
                return
            except OSError:
                logging.warning(
                    "Cannot copy victim '{}', maybe "
                    "filesystem problem? Waiting {} sec...".format(
                        self._cur_victim.name,
                        self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT,
                    ),
                )
                time.sleep(self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT)

        raise RuntimeError(
            "Problem copying victim '{}', curriculum stopped".format(
                self._cur_victim.name,
            ),
        )

    def try_move_on(self, games_for_compute_by_default):
        if self.finished:
            return

        cur_victim = self._cur_victim
        games_for_compute = games_for_compute_by_default
        if self._cur_victim.forward.window_size is not None:
            games_for_compute = cur_victim.forward.window_size
        adv_stat_forward = recompute_statistics(
            self.sgf_games,
            games_for_compute,
            cur_victim.name,
        )

        adv_stat_backward = None
        if cur_victim.backward is not None:
            games_for_compute = games_for_compute_by_default
            if cur_victim.backward.window_size is not None:
                games_for_compute = cur_victim.backward.window_size
            adv_stat_backward = recompute_statistics(
                self.sgf_games,
                games_for_compute,
                cur_victim.name,
            )

        logging.info("Checking whether we need to move to the next victim...")
        if adv_stat_forward and cur_victim.want_forward(adv_stat_forward):
            self.victim_idx += 1
            if self.victim_idx == len(self.victims):
                self.finished = True
                return
        elif adv_stat_backward and cur_victim.want_backward(adv_stat_backward):
            if self.victim_idx == 0:  # nowhere to go
                return
            else:
                self.victim_idx -= 1
        else:  # don't want to move anywhere
            return

        logging.info(f"Moving to the next victim '{asdict(self._cur_victim)}'")
        self.__try_victim_copy(True)
        self.sgf_games.clear()  # drop statistics for old games

    def update_sgf_games(self, selfplay_dir: pathlib.Path, max_stat_len: int):
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
        if len(self.sgf_games) > max_stat_len:
            del self.sgf_games[max_stat_len:]

    """
    Run curriculum checking.
    @param selfplay_dir: Folder with selfplay results.
    @param games_for_compute: Number of games to compute statistics.
    @param checking_periodicity: Checking interval in seconds.
    """

    def checking_loop(
        self,
        selfplay_dir: pathlib.Path,
        games_for_compute: int,
        checking_periodicity: int,
    ):
        logging.info("Starting curriculum loop")
        while True:
            max_stat_len = games_for_compute
            ws = self._cur_victim.forward.window_size
            if ws is not None and ws > max_stat_len:
                max_stat_len = ws
            if self._cur_victim.backward is not None:
                ws = self._cur_victim.backward.window_size
                if ws is not None and ws > max_stat_len:
                    max_stat_len = ws
            self.update_sgf_games(selfplay_dir, max_stat_len)
            self.try_move_on(games_for_compute)
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
        type=pathlib.Path,
        required=True,
        help="Directory with selfplay data",
    )
    parser.add_argument(
        "-input-models-dir",
        type=pathlib.Path,
        required=True,
        help="Input dir with victim model files",
    )
    parser.add_argument(
        "-output-models-dir",
        type=pathlib.Path,
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
        type=pathlib.Path,
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

    try:
        curriculum.checking_loop(
            args.selfplay_dir,
            args.games_for_compute,
            args.checking_periodicity,
        )
    # we really want to silence 'B902: blind except'
    # because we want a stacktrace and error description in logs
    except BaseException as e:  # noqa: B902
        logging.exception("Curriculum error: {}".format(e))
        raise

    logging.info("Curriculum finished!")
