#!/bin/bash -eu
if [[ $# -ne 1 ]]
then
    echo "Usage: $0 BASE_DIR"
    echo "Currently expects to be run from within the 'cpp' directory of the KataGo repo."
    echo "BASE_DIR containing selfplay data and models and related directories"
    exit 0
fi

BASE_DIR="$1"
MODELS_DIR="$BASE_DIR"/models
VICTIMS_DIR="$BASE_DIR"/victims
OUTPUT_DIR="$BASE_DIR"/eval
mkdir -p "$OUTPUT_DIR"/logs
mkdir -p "$OUTPUT_DIR"/sgfs

LAST_STEP=0
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

    if [[ ! -z "$MODELS" || ! -z "$VICTIM" ]]; then
        echo "Waiting for an adversary and a victim to exist..."
        sleep 30
        continue
    fi

    for MODEL_DIR in $MODELS ; do
        if [[ "$MODEL_DIR" =~ -s([0-9]+) ]] ; then
            # The first capture group is the step number
            STEP=${BASH_REMATCH[1]}

            # Have we evaluated this model yet?
            if [ "$STEP" -gt "$LAST_STEP" ] ; then
                # https://stackoverflow.com/questions/12152626/how-can-i-remove-the-extension-of-a-filename-in-a-shell-script
                VICTIM_NAME=$(echo "$VICTIM" | cut -f 1 -d '.')

                # Run the evaluation
                echo "Evaluating model $MODEL_DIR against victim $VICTIM_NAME"
                /engines/KataGo-custom/cpp/katago match \
                    -config /go_attack/configs/match-1gpu.cfg \
                    -override-config numGamesTotal=100 \
                    -override-config nnModelFile0="$VICTIMS_DIR"/"$VICTIM" \
                    -override-config nnModelFile1="$MODELS_DIR"/"$MODEL_DIR"/model.bin.gz \
                    -sgf-output-dir "$OUTPUT_DIR"/sgfs/"$VICTIM_NAME"_"$MODEL_DIR" \
                    2>&1 | tee "$OUTPUT_DIR"/logs/"$VICTIM_NAME"_"$MODEL_DIR".log

                # Update the last step
                LAST_STEP="$STEP"
            fi
        fi
    done
    sleep 30
done