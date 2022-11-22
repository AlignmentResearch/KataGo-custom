#!/bin/bash -eu
if [[ $# -lt 1 || $# -gt 2 ]]
then
    echo "Usage: $0 BASE_DIR [PREDICTOR_DIR]"
    echo "Currently expects to be run from within the 'cpp' directory of the KataGo repo."
    echo "BASE_DIR is the root of the training run, containing selfplay data, models and related directories."
    echo "PREDICTOR_DIR is the path containing predictor models, if applicable."
    exit 0
fi

BASE_DIR="$1"
PREDICTOR_DIR=${2:-}
MODELS_DIR="$BASE_DIR"/models
VICTIMS_DIR="$BASE_DIR"/victims
OUTPUT_DIR="$BASE_DIR"/eval
mkdir -p "$OUTPUT_DIR"/logs
mkdir -p "$OUTPUT_DIR"/sgfs

LAST_STEP=0
SLEEP_INTERVAL=30
while true
do
    if [[ ! -d "$MODELS_DIR" || ! -d "$VICTIMS_DIR" ]]
    then
        echo "Waiting for $MODELS_DIR and $VICTIMS_DIR to exist..."
        sleep 10
        continue
    fi
    # https://stackoverflow.com/questions/1015678/get-most-recent-file-in-a-directory-on-linux
    VICTIM=$(ls -Art "$VICTIMS_DIR" | tail -n 1)

    # Get directories in models/ in *natural sorted order* and see if any are new.
    # This means that the numeric components of the directory names will be sorted based
    # on their numeric value, not as strings, so early models will be evaluated first.
    MODELS=$(ls -v "$MODELS_DIR")

    if [[ -z "$MODELS" || -z "$VICTIM" ]]; then
        echo "Waiting for an adversary and a victim to exist..."
        sleep $SLEEP_INTERVAL
        continue
    fi

    for MODEL_DIR in $MODELS; do
        if [[ "$MODEL_DIR" =~ -s([0-9]+) ]]; then
            # The first capture group is the step number
            STEP=${BASH_REMATCH[1]}

            # Have we evaluated this model yet?
            if [ "$STEP" -gt "$LAST_STEP" ]; then
                # https://stackoverflow.com/questions/12152626/how-can-i-remove-the-extension-of-a-filename-in-a-shell-script
                VICTIM_NAME=$(echo "$VICTIM" | cut -f 1 -d '.')
                EXTRA_CONFIG="numGamesTotal=100"

                if [ -n "$PREDICTOR_DIR" ]; then
                    # https://stackoverflow.com/questions/4561895/how-to-recursively-find-the-latest-modified-file-in-a-directory
                    PREDICTOR=$(find $PREDICTOR_DIR -name *.bin.gz -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
                    EXTRA_CONFIG+=",predictorPath=$PREDICTOR"
                fi

                # Run the evaluation
                echo "Evaluating model $MODEL_DIR against victim $VICTIM_NAME"
                /engines/KataGo-custom/cpp/katago match \
                    -config /go_attack/configs/match-1gpu.cfg \
                    -config "$VICTIMS_DIR"/victim.cfg \
                    -override-config "$EXTRA_CONFIG" \
                    -override-config nnModelFile0="$VICTIMS_DIR"/"$VICTIM" \
                    -override-config nnModelFile1="$MODELS_DIR"/"$MODEL_DIR"/model.bin.gz \
                    -sgf-output-dir "$OUTPUT_DIR"/sgfs/"$VICTIM_NAME"_"$MODEL_DIR" \
                    2>&1 | tee "$OUTPUT_DIR"/logs/"$VICTIM_NAME"_"$MODEL_DIR".log

                # Update the last step
                LAST_STEP="$STEP"
            fi
        fi
    done
    sleep $SLEEP_INTERVAL
done
