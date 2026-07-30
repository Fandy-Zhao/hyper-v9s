import unittest

from compose.data.records import answer_text, question_text


class RecordAccessorTest(unittest.TestCase):
    def test_flat_record(self):
        record = {"text": "How many?", "answer": "3"}
        self.assertEqual(question_text(record), "How many?")
        self.assertEqual(answer_text(record), "3")

    def test_conversation_record(self):
        record = {"conversations": [
            {"from": "human", "value": "<image>\nWhat shape?"},
            {"from": "gpt", "value": "circle"},
        ]}
        self.assertEqual(question_text(record), "What shape?")
        self.assertEqual(answer_text(record), "circle")

    def test_missing_fields_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "human question"):
            question_text({})
        with self.assertRaisesRegex(ValueError, "teacher answer"):
            answer_text({})
