from __future__ import annotations

import logging
import unittest
from context_memory.core.logging import get_logger, setup_logging, timed_operation


class LoggingAndLatencyTests(unittest.TestCase):
    def test_setup_logging_configures_logger(self) -> None:
        setup_logging(level=logging.DEBUG)
        logger = get_logger("test.module")
        self.assertEqual(logger.name, "test.module")

    def test_timed_operation_success(self) -> None:
        logger = get_logger("test.timer")
        with timed_operation(logger, "test_op", {"meta": "val"}) as ctx:
            ctx["custom_result"] = 123
            x = sum([1, 2, 3])
        self.assertEqual(x, 6)
        self.assertEqual(ctx["custom_result"], 123)

    def test_timed_operation_records_failure_and_reraises(self) -> None:
        logger = get_logger("test.timer.fail")
        with self.assertRaises(ValueError):
            with timed_operation(logger, "failing_op"):
                raise ValueError("expected failure")


if __name__ == "__main__":
    unittest.main()
