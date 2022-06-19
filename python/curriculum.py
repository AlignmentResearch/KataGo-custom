#!/usr/bin/python3

import argparse
import json
import os
import shutil
import time
from threading import Thread
from typing import Optional, List, Tuple, Union

from sgfmill import sgf

# format: victim name, conditions for moving on:
# win rate, diff score, diff score without komi, policy_loss
# for quick copy as a JSON
"""
[
  ["kata1-b6c96-s41312768-d6061202.txt.gz", 0.75, null, null, null],
  ["kata1-b40c256-s7186724608-d1743537710.bin.gz", 0.75, null, null, null],
  ["g170-b30c320x2-s4824661760-d1229536699.bin.gz", 0.75, null, null, null]
]
"""


class AdvGameInfo:
    def __init__(self, winner: Optional[bool], diff_score: float, diff_score_wo_komi: float):
        self.winner = winner  # can be None!
        self.diff_score = diff_score
        self.diff_score_wo_komi = diff_score_wo_komi


class PlayerStat:
    # for victim only one criteria can be enabled, all others should be None
    def __init__(self,
                 player_name: str = None,
                 win_rate: float = None,
                 score_diff: float = None,
                 score_wo_komi_diff: float = None,
                 policy_loss: float = None):
        self.name = player_name
        self.win_rate = win_rate
        self.score_diff = score_diff
        self.score_wo_komi_diff = score_wo_komi_diff
        self.policy_loss = policy_loss

    def get_stat_members(self) -> List[float, float, float, float]:
        return [self.win_rate, self.score_diff, self.score_wo_komi_diff, self.policy_loss]

    def can_be_victim_criteria(self) -> bool:
        criteria = self.get_stat_members()
        num_enabled = len([x for x in criteria if x is not None])
        return num_enabled == 1

    # check if adv_stat has a greater value of enabled criteria
    def check_if_gt(self, adv_stat) -> bool:
        criteria = self.get_stat_members()
        adv_vals = adv_stat.get_stat_members()
        for c, a in zip(criteria, adv_vals):
            if c is not None and a > c:
                return True
        return False


def get_game_info(sgf_str: str) -> AdvGameInfo:
    sgf_game = sgf.Sgf_game.from_string(sgf_str)

    b_name = sgf_game.get_player_name("b")
    w_name = sgf_game.get_player_name("w")

    if "victim-" in b_name and "__" not in b_name:
        # it has 'victim' in its name and it is not a colored evaluator
        victim_name = b_name
    else:
        victim_name = w_name

    victim_color = {b_name: "b", w_name: "w"}[victim_name]
    adv_color = {"b": "w", "w": "b"}[victim_color]

    adv_raw_name = {"b": b_name, "w": w_name}[adv_color]

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

    if win_color is None:
        adv_minus_victim_score = 0
        adv_minus_victim_score_wo_komi = 0
    else:
        win_score = float(sgf_game.get_root().get("RE").split("+")[1])
        adv_minus_victim_score = {
            win_color: win_score,
            lose_color: -win_score,
        }[adv_color]
        adv_minus_victim_score_wo_komi = adv_minus_victim_score - adv_komi

    winner = None
    if win_color is not None:
        winner = win_color == adv_color

    return AdvGameInfo(winner, adv_minus_victim_score, adv_minus_victim_score_wo_komi)


def read_sgf_files(selfplay_dir: str, games_for_compute: int) -> Tuple[list, int]:
    all_sgfs = []
    for path, dirnames, filenames in os.walk(selfplay_dir, followlinks=True):
        for f in filenames:
            if os.path.splitext(f)[1] == '.sgfs':
                all_sgfs.append([os.path.join(path, f), os.path.getmtime(path)])
    all_sgfs.sort(key=lambda x: x[1], reverse=True)

    sgf_strings = []
    files_checked = 0
    for sgf_file in all_sgfs:
        with open(sgf_file[0]) as f:
            files_checked += 1
            for line in f.readlines():
                sgf_strings.append(line.strip())
                if len(sgf_strings) >= games_for_compute:
                    return sgf_strings, files_checked
    return sgf_strings, files_checked


def recompute_statistics(selfplay_dir: str, games_for_compute: int) -> Optional[PlayerStat]:
    sgf_strings, files_checked = read_sgf_files(selfplay_dir, games_for_compute)

    # don't have enough data
    if len(sgf_strings) < games_for_compute:
        return None

    sum_wins = 0
    sum_loses = 0
    sum_score = 0
    sum_score_wo_komi = 0
    for sgf_str in sgf_strings:
        game = get_game_info(sgf_str)
        if game.winner:
            sum_wins += 1
        elif game.winner is not None:
            sum_loses = 0
        sum_score += game.diff_score
        sum_score_wo_komi += game.diff_score_wo_komi

    win_rate = float(sum_wins) / len(sgf_strings)
    mean_diff_score = float(sum_score) / len(sgf_strings)
    mean_diff_score_wo_komi = float(sum_score_wo_komi) / len(sgf_strings)

    print("Files checked: %d" % files_checked, flush=True)

    return PlayerStat(
        win_rate=win_rate,
        score_diff=mean_diff_score,
        score_wo_komi_diff=mean_diff_score_wo_komi,
    )


class Curriculum:
    def __init__(self,
                 victims_input_dir: str,
                 victims_output_dir: str,
                 config: list = None,
                 config_json: str = None,
                 config_json_file: str = None):
        self.MAX_VICTIM_COPYING_EFFORTS = 10
        self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT = 10

        if config_json_file is not None:
            print("Curriculum: loading JSON config")
            with open(config_json_file) as f:
                config = json.load(f)
        elif config_json is not None:
            print("Curriculum: loading JSON string")
            config = json.loads(config_json)
        elif config is not None:
            print("Using python list as a config")

        if len(config) < 1:
            raise ValueError("Empty config for the curriculum play!")

        self.victims_input_dir = victims_input_dir
        self.victims_output_dir = victims_output_dir

        self.victim_idx = 0
        self.finished = False
        self.victims = []
        for line in config:
            cond = PlayerStat(line[0], win_rate=line[1], score_diff=line[2], score_wo_komi_diff=line[3],
                              policy_loss=line[4])
            if not cond.can_be_victim_criteria():
                raise ValueError(
                    "Incorrect victim change criteria for victim '{}': exactly one value should be non-None".format(
                        line[0]))
            self.victims.append(cond)

        print("Loaded curriculum with the following params:")
        print(*config, sep='\n')

    def __cur_victim(self) -> PlayerStat:
        return self.victims[self.victim_idx]

    def try_move_on(self, adv_stat: PlayerStat = None, policy_loss: float = None):
        if self.finished:
            return

        print("Checking whether we need to move to the next victim...")
        want_victim_update = False
        if adv_stat is not None and self.__cur_victim().check_if_gt(adv_stat):
            want_victim_update = True
        if policy_loss is not None:
            raise NotImplementedError("Policy loss check is not implemented yet")

        if not want_victim_update:
            return

        self.victim_idx += 1
        if self.victim_idx == len(self.victims):
            self.finished = True
            return

        print("Moving to the next victim '{}'".format(self.__cur_victim().name))
        num_efforts = 0
        for _ in range(self.MAX_VICTIM_COPYING_EFFORTS):
            try:
                shutil.copy(os.path.join(self.victims_input_dir, self.__cur_victim().name), self.victims_output_dir)
                return
            except:
                print("Cannot copy victim '{}', maybe filesystem problem? Waiting {} sec...".format(
                    self.__cur_victim().name, self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT))
                num_efforts += 1
                time.sleep(self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT)

        raise RuntimeError("Problem copying victim '{}', curriculum stopped".format(self.__cur_victim().name))

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
            checking_periodicity: int):
        print("Starting curriculum loop")
        while True:
            adv_stat = recompute_statistics(selfplay_dir, games_for_compute)
            if adv_stat is not None:
                curriculum.try_move_on(adv_stat=adv_stat)
                if curriculum.finished:
                    print("Curriculum is done. Stopping")
                    break
            print("Curriculum is alive, current victim idx: {}".format(self.victim_idx))
            time.sleep(checking_periodicity)

    """
    Run curriculum checking and return control to the main script.
    @param selfplay_dir: Folder with selfplay results.
    @param games_for_compute: Number of games to compute statistics.
    @param checking_periodicity: Checking interval in seconds.
    @return: created thread (for using thread.join() in the main script)
    """
    def run_thread(
            self,
            selfplay_dir: str,
            games_for_compute: int,
            checking_periodicity: int) -> Thread:
        thread = Thread(target=self.checking_loop,
                        args=(selfplay_dir, games_for_compute, checking_periodicity))
        thread.start()
        print("Curriculum models update thread started")
        return thread


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run victim replacement based on win rate.')
    parser.add_argument('-selfplay-dir', required=True, help='Directory with selfplay data')
    parser.add_argument('-input-models-dir', required=True, help='Input dir with victim model files')
    parser.add_argument('-output-models-dir', required=True, help='Output dir for adding new victims')
    parser.add_argument('-games-for-compute', type=int, required=False, default=1000,
                        help='Number of last games for statistics computation')
    parser.add_argument('-checking-periodicity', type=int, required=False, default=60,
                        help='Statistics computation periodicity in seconds')
    parser.add_argument('-config-json-string', required=False,
                        help='Curriculum JSON config with victims sequence (JSON content)')
    parser.add_argument('-config-json-file', default='configs/curriculum_conf.json',
                        help='Curriculum JSON config with victims sequence (JSON file path)')

    args = parser.parse_args()

    if args.config_json_file is not None:
        curriculum = Curriculum(args.input_models_dir, args.output_models_dir, config_json_file=args.config_json_file)
    elif args.config_json_string is not None:
        curriculum = Curriculum(args.input_models_dir, args.output_models_dir, config_json=args.config_json_string)
    else:
        raise ValueError("Curriculum: either path to JSON config or JSON config string must be provided")

    curriculum.checking_loop(
        args.selfplay_dir,
        args.games_for_compute,
        args.checking_periodicity)

    print("Curriculum finished!")
