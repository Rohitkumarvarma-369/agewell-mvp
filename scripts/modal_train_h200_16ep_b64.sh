#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

uv run modal run --timestamps -m agewell.modal_apps.phase5_eval::train_h200 \
  --teacher-run-name "${TEACHER_RUN_NAME:-phase5_teacher_16ep_b64_h200}" \
  --student-run-name "${STUDENT_RUN_NAME:-phase5_student_from_16ep_teacher_b64_h200}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --max-epochs "${MAX_EPOCHS:-16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --tabpfn-estimators "${TABPFN_ESTIMATORS:-8}" \
  --precision "${PRECISION:-bf16-mixed}" \
  --log-every-n-steps "${LOG_EVERY_N_STEPS:-10}" \
  --warm-batch-size "${WARM_BATCH_SIZE:-256}" \
  "${WARM_CACHE_FLAG:---warm-cache}"
