#!/bin/bash -eu
set -o pipefail
{
# Takes any models in modelstobetested/ and uploads them, then moves them to modelsuploaded/
# Should be run periodically.

if [[ $# -ne 4 ]]
then
    echo "Usage: $0 RUNNAME BASEDIR CONNECTION_CONFIG"
    echo "Currently expects to be run from within the 'python' directory of the KataGo repo, or otherwise in the same dir as upload_model.py."
    echo "RUNNAME should match what the server uses as the run name, try to pick something globally unique. Will prefix model names in uploaded files."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "CONNECTION_CONFIG config containing serverUrl, username, password"
    echo "RATING_ONLY if 1, upload for rating only, else upload for selfplay too"
    exit 0
fi
RUNNAME="$1"
shift
BASEDIR="$1"
shift
CONNECTION_CONFIG="$1"
shift
RATING_ONLY="$1"
shift

#------------------------------------------------------------------------------

mkdir -p "$BASEDIR"/modelstobetested
mkdir -p "$BASEDIR"/modelsuploaded

function uploadStuff() {
    FROMDIR="$1"
    TODIR="$2"

    #Sort by timestamp so that we process in order of oldest to newest if there are multiple
    for FILEPATH in $(find "$BASEDIR"/"$FROMDIR"/ -mindepth 1 -maxdepth 1 -printf "%T@ %p\n" | sort -n | cut -d ' ' -f 2)
    do
        if [ ${FILEPATH: -10} == ".uploading" ]
        then
            echo "Skipping upload tmp file:" "$FILEPATH"
        else
            echo "Found model to export:" "$FILEPATH"
            NAME="$(basename "$FILEPATH")"

            SRC="$BASEDIR"/"$FROMDIR"/"$NAME"
            TMPDST="$BASEDIR"/"$FROMDIR"/"$NAME".uploading
            TARGETDIR="$BASEDIR"/"$TODIR"
            TARGET="$BASEDIR"/"$TODIR"/"$NAME"

            if [ -d "$BASEDIR"/modelsuploaded/"$NAME" ]
            then
                echo "Model with same name already exists, so skipping:" "$SRC"
            else
                rm -rf "$TMPDST"
                mkdir "$TMPDST"

                TOBEZIPPED="$TMPDST"/"$RUNNAME"-"$NAME"
                mkdir "$TOBEZIPPED"

                # Build zip containing the ckpt
                cp "$SRC"/model.ckpt "$TOBEZIPPED"/model.ckpt
                (cd "$TMPDST"; zip -r "$RUNNAME"-"$NAME".zip "$RUNNAME"-"$NAME"/)
                rm "$TOBEZIPPED"/*
                rmdir "$TOBEZIPPED"

                cp "$SRC"/model.bin.gz "$TMPDST"/"$RUNNAME"-"$NAME".bin.gz
                cp "$SRC"/metadata.json "$TMPDST"/metadata.json
                cp "$SRC"/log.txt "$TMPDST"/log.txt

                #Sleep a little to allow some tolerance on the filesystem
                sleep 3

                SUCCESSFUL=0
                BACKOFF=10
                while [ $SUCCESSFUL -ne 1 ]
                do
                    set +e
                    set -x
                    python3 ./upload_model.py \
                            -run-name "$RUNNAME" \
                            -model-name "$RUNNAME"-"$NAME" \
                            -model-file "$TMPDST"/"$RUNNAME"-"$NAME".bin.gz \
                            -model-zip "$TMPDST"/"$RUNNAME"-"$NAME".zip \
                            -upload-log-file "$TMPDST"/upload_log.txt \
                            -metadata-file "$TMPDST"/metadata.json \
                            -parents-dir "$TARGETDIR" \
                            -connection-config "$CONNECTION_CONFIG" \
                            -rating-only "$RATING_ONLY"
                    RESULT=$?
                    set +x
                    set -e
                    if [ $RESULT -ne 0 ]
                    then
                        echo "Sleeping $BACKOFF seconds before trying again"
                        sleep "$BACKOFF"
                        BACKOFF=$(( BACKOFF * 4 / 3 ))
                        if [ $BACKOFF -gt 10000 ]
                        then
                            BACKOFF=10000
                        fi
                    else
                        SUCCESSFUL=1
                    fi
                done

                #Sleep a little to allow some tolerance on the filesystem
                sleep 3

                mv "$TMPDST" "$TARGET"
                rm -r "$SRC"

                #Sleep a little to allow some tolerance on the filesystem
                sleep 3

                echo "Done uploading to server:" "$NAME" " and moved to" "$TARGET"
            fi
        fi
    done
}

uploadStuff "modelstobetested" "modelsuploaded"

exit 0
}
