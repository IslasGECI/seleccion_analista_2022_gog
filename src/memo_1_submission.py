#!/usr/bin/env python

from pollos_petrel import predict_age_pollos_petrel, write_submission, Path_To_Submission

write_submission(Path_To_Submission().LinearModel, predict_age_pollos_petrel)
