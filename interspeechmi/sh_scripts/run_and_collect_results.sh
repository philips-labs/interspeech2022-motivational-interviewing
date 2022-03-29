#!/bin/bash

SH_SCRIPT_PATH=$(readlink -f "$0")
SH_SCRIPT_DIR=$(dirname "$SH_SCRIPT_PATH")

ALL_CODE_DIR=$(dirname "$SH_SCRIPT_DIR")
PY_CODE_DIR="$ALL_CODE_DIR/py_scripts"

python "$PY_CODE_DIR/preprocess.py"

python "$PY_CODE_DIR/current_turn_code_classification.py"
python "$PY_CODE_DIR/get_utterance_classifier_performance.py"

python "$PY_CODE_DIR/next_turn_code_forecast.part1.py"
python "$PY_CODE_DIR/next_turn_code_forecast.part2.py"
python "$PY_CODE_DIR/next_turn_code_forecast.part3.py"
python "$PY_CODE_DIR/next_turn_code_forecast.part4.py"
python "$PY_CODE_DIR/next_turn_code_forecast.part5.py"
python "$PY_CODE_DIR/collect_all_results.py"

