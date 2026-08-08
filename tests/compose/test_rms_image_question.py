import unittest

from llava.constants import DEFAULT_IMAGE_TOKEN

from compose.eval.rms_stats import _image_question


class RmsImageQuestionTest(unittest.TestCase):
    """Regression: calibration batches must carry exactly one image
    placeholder per sample. The UCIT instruction files embed <image> in the
    human message themselves; rms_stats used to prepend a second token,
    leaving more image tokens than images and crashing with IndexError in
    prepare_inputs_labels_for_multimodal (smoke run 8, S9)."""

    def test_embedded_placeholder_untouched(self):
        text = DEFAULT_IMAGE_TOKEN + "\nWhat is the object in the image?"
        self.assertEqual(_image_question(text), text)

    def test_placeholder_prepended_when_missing(self):
        result = _image_question("How many people?")
        self.assertTrue(result.startswith(DEFAULT_IMAGE_TOKEN + "\n"))
        self.assertEqual(result.count(DEFAULT_IMAGE_TOKEN), 1)

    def test_multiple_placeholders_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "image placeholders"):
            _image_question("<image>\n<image>\nB")


if __name__ == "__main__":
    unittest.main()
