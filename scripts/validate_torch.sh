#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPHAFAST_PYTHON_BIN="${ALPHAFAST_VALIDATION_PYTHON:-python3}"

usage() {
    cat <<'EOF'
Usage:
  scripts/validate_torch.sh weight \
      --weights-file PATH \
      --mapping-report PATH \
      [--result-file PATH]

  scripts/validate_torch.sh e2e \
      --reference PATH \
      --candidate PATH \
      --tolerances PATH \
      [--result-file PATH]

  scripts/validate_torch.sh ranking \
      --reference PATH \
      --candidate PATH \
      [--frozen-features PATH --input-json PATH] \
      [--tolerances PATH] \
      [--result-file PATH]

  scripts/validate_torch.sh accuracy \
      --output-root PATH \
      --ground-truth-root PATH \
      [--tool PATH] \
      [--profile smoke|full] \
      [--thresholds PATH] \
      [--result-file PATH]

  scripts/validate_torch.sh accuracy_delta \
      --reference-report PATH \
      --candidate-report PATH \
      [--tolerances PATH] \
      [--result-file PATH]

  scripts/validate_torch.sh suite \
      --manifest PATH \
      [--result-file PATH]

Exit codes:
  0  validation passed
  1  candidate failed validation
  2  validation could not run
EOF
}

if [ "$#" -eq 0 ]; then
    usage >&2
    exit 2
fi

STAGE="$1"
shift

case "$STAGE" in
    weight)
        exec "$ALPHAFAST_PYTHON_BIN" \
            "$SCRIPT_DIR/validation/validate_weights.py" "$@"
        ;;
    e2e)
        exec "$ALPHAFAST_PYTHON_BIN" \
            "$SCRIPT_DIR/validation/validate_e2e.py" "$@"
        ;;
    ranking)
        exec "$ALPHAFAST_PYTHON_BIN" \
            "$SCRIPT_DIR/validation/validate_ranking.py" "$@"
        ;;
    accuracy)
        exec "$ALPHAFAST_PYTHON_BIN" \
            "$SCRIPT_DIR/validation/validate_accuracy.py" "$@"
        ;;
    accuracy_delta)
        exec "$ALPHAFAST_PYTHON_BIN" \
            "$SCRIPT_DIR/validation/validate_accuracy_delta.py" "$@"
        ;;
    suite)
        exec "$ALPHAFAST_PYTHON_BIN" \
            "$SCRIPT_DIR/validation/run_suite.py" "$@"
        ;;
    --help|-h)
        usage
        ;;
    *)
        echo "ERROR: validation stage is not implemented: $STAGE" >&2
        usage >&2
        exit 2
        ;;
esac
