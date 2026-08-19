from __future__ import annotations

import unittest

from context_memory.core.enums import IngestionJobState as S
from context_memory.core.enums import is_legal_job_transition


class JobTransitionTests(unittest.TestCase):
    def test_happy_path_single_steps_are_legal(self) -> None:
        self.assertTrue(is_legal_job_transition(S.PENDING_GRAPH, S.PENDING_EMBEDDINGS, None))
        self.assertTrue(is_legal_job_transition(S.PENDING_EMBEDDINGS, S.VERIFYING, None))
        self.assertTrue(is_legal_job_transition(S.VERIFYING, S.COMPLETED, None))

    def test_backward_single_step_is_illegal(self) -> None:
        self.assertFalse(is_legal_job_transition(S.VERIFYING, S.PENDING_GRAPH, None))
        self.assertFalse(is_legal_job_transition(S.PENDING_EMBEDDINGS, S.PENDING_GRAPH, None))

    def test_reissuing_current_state_is_legal_idempotent_replay(self) -> None:
        self.assertTrue(is_legal_job_transition(S.PENDING_EMBEDDINGS, S.PENDING_EMBEDDINGS, None))

    def test_skipping_ahead_from_non_terminal_is_legal(self) -> None:
        # Supports "redo whole pipeline" orchestration re-issuing a state already reached.
        self.assertTrue(is_legal_job_transition(S.PENDING_GRAPH, S.VERIFYING, None))

    def test_any_non_terminal_state_can_fail(self) -> None:
        for state in (S.PENDING_GRAPH, S.PENDING_EMBEDDINGS, S.VERIFYING):
            self.assertTrue(is_legal_job_transition(state, S.RETRYABLE_FAILED, None))
            self.assertTrue(is_legal_job_transition(state, S.TERMINAL_FAILED, None))
            self.assertTrue(is_legal_job_transition(state, S.MANUAL_REPAIR, None))

    def test_completed_has_no_next_state(self) -> None:
        for state in S:
            self.assertFalse(is_legal_job_transition(S.COMPLETED, state, None))

    def test_retryable_resumes_forward_from_last_verified_state(self) -> None:
        self.assertTrue(is_legal_job_transition(S.RETRYABLE_FAILED, S.PENDING_EMBEDDINGS, S.PENDING_GRAPH))
        self.assertTrue(is_legal_job_transition(S.RETRYABLE_FAILED, S.VERIFYING, S.PENDING_EMBEDDINGS))
        self.assertFalse(is_legal_job_transition(S.RETRYABLE_FAILED, S.PENDING_GRAPH, S.PENDING_EMBEDDINGS))

    def test_retryable_resumes_from_pending_graph_when_no_prior_checkpoint(self) -> None:
        self.assertTrue(is_legal_job_transition(S.RETRYABLE_FAILED, S.PENDING_GRAPH, None))

    def test_retryable_can_escalate_to_terminal_or_manual_repair(self) -> None:
        self.assertTrue(is_legal_job_transition(S.RETRYABLE_FAILED, S.TERMINAL_FAILED, None))
        self.assertTrue(is_legal_job_transition(S.RETRYABLE_FAILED, S.MANUAL_REPAIR, None))

    def test_terminal_failed_only_exits_via_manual_repair(self) -> None:
        self.assertTrue(is_legal_job_transition(S.TERMINAL_FAILED, S.MANUAL_REPAIR, None))
        self.assertFalse(is_legal_job_transition(S.TERMINAL_FAILED, S.PENDING_GRAPH, None))
        self.assertFalse(is_legal_job_transition(S.TERMINAL_FAILED, S.RETRYABLE_FAILED, None))

    def test_manual_repair_can_resume_or_escalate_to_terminal(self) -> None:
        self.assertTrue(is_legal_job_transition(S.MANUAL_REPAIR, S.PENDING_GRAPH, None))
        self.assertTrue(is_legal_job_transition(S.MANUAL_REPAIR, S.TERMINAL_FAILED, None))
        self.assertFalse(is_legal_job_transition(S.MANUAL_REPAIR, S.RETRYABLE_FAILED, None))


if __name__ == "__main__":
    unittest.main()
