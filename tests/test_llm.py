"""Tests fuer llm.LLM.structured() - die Reply-Huelle, die Denk-Trennung und
die Parallelfestigkeit (mehrere structured()-Aufrufe gleichzeitig).

Kein echter Server: ein FakeLLM fuellt complete() mit festen Antworten.
"""

import concurrent.futures
import time
import unittest

from pydantic import BaseModel

import llm


class Tiny(BaseModel):
    x: int


def _bare() -> llm.LLM:
    """Nur fuer _split_thinking - die Basisklasse braucht kein complete()."""
    return llm.LLM()


class ReplyShapeTest(unittest.TestCase):
    def test_structured_returns_reply_with_value_and_metadata(self):
        class E(llm.LLM):
            def complete(self, messages, schema=None):
                return '{"x": 7}', "some reasoning", 42.0

        reply = E().structured([{"role": "user", "content": "hi"}], Tiny)
        self.assertIsInstance(reply, llm.Reply)
        self.assertEqual(reply.value.x, 7)
        self.assertEqual(reply.thinking, "some reasoning")
        self.assertEqual(reply.tokens_per_sec, 42.0)

    def test_rate_is_written_to_the_module_global_for_the_footer(self):
        class E(llm.LLM):
            def complete(self, messages, schema=None):
                return '{"x": 1}', "", 99.5

        E().structured([{"role": "user", "content": "hi"}], Tiny)
        self.assertEqual(llm.last_tokens_per_sec, 99.5)


class EmptyResponseTest(unittest.TestCase):
    def test_thinking_survives_into_the_final_error(self):
        """Der verrannte Denkprozess des letzten Versuchs muss im
        StructuredError landen - story._log_thinking() liest ihn dort im
        Fehlerpfad (getattr(e, 'thinking', ''))."""
        class E(llm.LLM):
            def complete(self, messages, schema=None):
                raise llm.EmptyResponse("empty", thinking="ran away in circles")

        with self.assertRaises(llm.StructuredError) as cm:
            E().structured([{"role": "user", "content": "x"}], Tiny, retries=1)
        self.assertEqual(cm.exception.thinking, "ran away in circles")


class SplitThinkingTest(unittest.TestCase):
    def test_inline_block_is_appended_to_prior(self):
        clean, thinking = _bare()._split_thinking("<think>abc</think>{}", prior="pre")
        self.assertEqual(clean, "{}")
        self.assertEqual(thinking, "pre\nabc")

    def test_no_tags_returns_content_and_prior_untouched(self):
        clean, thinking = _bare()._split_thinking('{"x": 1}', prior="pre")
        self.assertEqual((clean, thinking), ('{"x": 1}', "pre"))

    def test_unclosed_think_yields_empty_clean(self):
        clean, thinking = _bare()._split_thinking("<think>loop forever", prior="")
        self.assertEqual(clean, "")
        self.assertIn("loop forever", thinking)

    def test_bare_closing_tag_splits_head_as_thinking(self):
        clean, thinking = _bare()._split_thinking("reasoning here</think>{}", prior="")
        self.assertEqual(clean, "{}")
        self.assertEqual(thinking, "reasoning here")


class ParallelNoClobberTest(unittest.TestCase):
    def test_concurrent_structured_calls_keep_their_own_thinking(self):
        """Der eigentliche Grund fuer die Reply-Umstellung: laufen fuenf
        structured()-Aufrufe gleichzeitig (agentische DECIDE-Faecherung),
        darf keiner den Denk-Trace eines anderen sehen."""
        class E(llm.LLM):
            def complete(self, messages, schema=None):
                tag = messages[0]["content"]
                time.sleep(0.05)   # Ueberlappung erzwingen
                return '{"x": 1}', f"thinking-{tag}", None

        engine = E()

        def call(tag):
            return engine.structured([{"role": "user", "content": tag}], Tiny)

        tags = [f"t{i}" for i in range(5)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            replies = list(ex.map(call, tags))

        for tag, reply in zip(tags, replies):
            self.assertEqual(reply.thinking, f"thinking-{tag}")


if __name__ == "__main__":
    unittest.main()
