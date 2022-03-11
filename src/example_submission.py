#!/usr/bin/env python

from pollos_petrel import add_mean_as_target, write_submission
from .dummy_model import Path_To_Submission

write_submission(Path_To_Submission().DummyModel, add_mean_as_target)
