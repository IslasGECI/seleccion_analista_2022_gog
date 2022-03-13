#!/usr/bin/env python

from pollos_petrel import (
    predict_target_dummy_model,
    predict_target_linear_model,
    predict_target_power_model,
    write_submission,
    Path_To_Submission,
)

write_submission(Path_To_Submission().DummyModel, predict_target_dummy_model)
write_submission(Path_To_Submission().LinearModel, predict_target_linear_model)
write_submission(Path_To_Submission().PowerModel, predict_target_power_model)
