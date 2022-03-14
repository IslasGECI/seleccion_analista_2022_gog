#!/usr/bin/env python

from pollos_petrel import Model, write_submission

write_submission(Model().DummyModel)
write_submission(Model().LinearModel)
write_submission(Model().PowerModel)
