import json
import tempfile
import os
import torch
import unittest

from tunix.data.dataset import SFTDataset
from Tunix.tunix.test_dataset import IGNORE_INDEX


class MockTokenizer:
    """A minimal tokenizer that maps characters to integers for testing."""

    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.padding_side = "right"

    def __call__(self, text, truncation=False, add_special_tokens=False):
        ids = [hash(c) % self.vocab_size for c in text]
        return {"input_ids": ids}


class TestSFTDataset(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.samples = [
            {
                "conversation_id": 0,
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
                ],
                "system_prompt": "你是一个助手。",
            },
            {
                "conversation_id": 1,
                "messages": [
                    {"role": "user", "content": "1+1=?"},
                    {"role": "assistant", "content": "等于2。"},
                    {"role": "user", "content": "那2+2呢？"},
                    {"role": "assistant", "content": "等于4。"},
                ],
                "system_prompt": "你是一个助手。",
            },
        ]
        self.tmpdir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.tmpdir, "train.json")
        with open(self.data_path, "w") as f:
            json.dump(self.samples, f, ensure_ascii=False)

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    # ----------------------------------------------------------------
    # Basic properties
    # ----------------------------------------------------------------
    def test_dataset_length(self):
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        self.assertEqual(len(ds), 2)

    def test_item_keys(self):
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        item = ds[0]
        self.assertIn("input_ids", item)
        self.assertIn("labels", item)
        self.assertIn("attention_mask", item)

    def test_tensor_types(self):
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        item = ds[0]
        for key in ("input_ids", "labels", "attention_mask"):
            self.assertIsInstance(item[key], torch.Tensor)
            self.assertEqual(item[key].dtype, torch.long)

    def test_all_tensors_same_length(self):
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        item = ds[0]
        n = len(item["input_ids"])
        self.assertEqual(len(item["labels"]), n)
        self.assertEqual(len(item["attention_mask"]), n)

    # ----------------------------------------------------------------
    # Loss masking
    # ----------------------------------------------------------------
    def test_loss_mask_user_tokens_ignored(self):
        """User / system tokens should have labels == IGNORE_INDEX."""
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        item = ds[0]  # conversation with 2 messages: user, assistant
        labels = item["labels"]
        input_ids = item["input_ids"]

        # Find the assistant segment separator in input_ids
        # We verify that *some* labels match input_ids (assistant)
        # and *some* are -100 (non-assistant)
        predict_mask = labels != IGNORE_INDEX
        ignore_mask = labels == IGNORE_INDEX
        self.assertTrue(predict_mask.any(), msg="Expected at least one assistant token with loss")
        self.assertTrue(ignore_mask.any(), msg="Expected at least one non-assistant token (user/system)")
        self.assertTrue((labels[predict_mask] == input_ids[predict_mask]).all(),
                        msg="Assistant labels must equal input_ids")

    def test_loss_mask_all_assistant_have_valid_labels(self):
        """All assistant tokens in a multi-turn conversation produce valid labels."""
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        item = ds[1]  # 4 messages: user, assistant, user, assistant
        labels = item["labels"]
        input_ids = item["input_ids"]

        predict_mask = labels != IGNORE_INDEX
        ignore_mask = labels == IGNORE_INDEX
        self.assertTrue(predict_mask.any())
        self.assertTrue(ignore_mask.any())
        self.assertTrue((labels[predict_mask] == input_ids[predict_mask]).all())

    def test_predict_mask_matches_assistant_role(self):
        """Use segment info to check that only assistant segments are predicted."""
        from Tunix.tunix.test_dataset import get_segments

        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        item = ds[1]

        # Recompute segments and tokenize each to find the expected
        # number of assistant tokens
        segments = get_segments(self.samples[1]["messages"], self.samples[1]["system_prompt"])
        assistant_tokens = 0
        total_tokens = 0
        for seg_type, text in segments:
            ids = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
            total_tokens += len(ids)
            if seg_type == "assistant":
                assistant_tokens += len(ids)
        # + eos token
        total_tokens += 1
        assistant_tokens += 1

        predict_count = (item["labels"] != IGNORE_INDEX).sum().item()
        self.assertEqual(predict_count, assistant_tokens)

    # ----------------------------------------------------------------
    # EOS token
    # ----------------------------------------------------------------
    def test_eos_token_appended(self):
        """When add_eos_token=True, the last label should be eos_token_id."""
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024, add_eos_token=True)
        item = ds[0]
        self.assertEqual(item["labels"][-1].item(), self.tokenizer.eos_token_id,
                         msg="Last label should be EOS token")
        self.assertEqual(item["input_ids"][-1].item(), self.tokenizer.eos_token_id,
                         msg="Last input_id should be EOS token")

    def test_no_eos_when_disabled(self):
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024, add_eos_token=False)
        item = ds[0]
        self.assertNotEqual(item["labels"][-1].item(), self.tokenizer.eos_token_id,
                            msg="When add_eos_token=False, last label should NOT be EOS")

    # ----------------------------------------------------------------
    # Truncation
    # ----------------------------------------------------------------
    def test_truncation_from_left(self):
        """When max_length is smaller than the full sequence, data is
        truncated from the left (preserving the end of conversation)."""
        ds_long = SFTDataset(self.data_path, self.tokenizer, max_length=4096)
        ds_short = SFTDataset(self.data_path, self.tokenizer, max_length=5)

        item_long = ds_long[0]
        item_short = ds_short[0]

        # Short version should end the same way as the tail of the long
        # version (because we truncate from the left).
        torch.testing.assert_close(
            item_short["input_ids"],
            item_long["input_ids"][-5:],
            msg="Truncation should preserve the rightmost tokens",
        )
        torch.testing.assert_close(
            item_short["labels"],
            item_long["labels"][-5:],
        )
        torch.testing.assert_close(
            item_short["attention_mask"],
            item_long["attention_mask"][-5:],
        )

    # ----------------------------------------------------------------
    # Multiple items
    # ----------------------------------------------------------------
    def test_different_items(self):
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        item0 = ds[0]
        item1 = ds[1]
        self.assertFalse(torch.equal(item0["input_ids"], item1["input_ids"]),
                         msg="Different samples should have different content")

    # ----------------------------------------------------------------
    # Edge cases
    # ----------------------------------------------------------------
    def test_empty_messages_skipped(self):
        """A conversation with no messages should not produce examples
        (handled in preprocess, but dataset should tolerate empty items)."""
        samples = [{"conversation_id": 2, "messages": []}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(samples, f, ensure_ascii=False)
            tmp = f.name
        try:
            ds = SFTDataset(tmp, self.tokenizer, max_length=1024)
            self.assertEqual(len(ds), 1)
            # The dataset will still produce a sample, but it will be
            # composed only of the system prompt + eos
            item = ds[0]
            self.assertGreater(len(item["input_ids"]), 0)
        finally:
            os.remove(tmp)

    def test_attention_mask_all_ones_within_range(self):
        ds = SFTDataset(self.data_path, self.tokenizer, max_length=1024)
        item = ds[0]
        self.assertTrue((item["attention_mask"] == 1).all(),
                        msg="Without padding all attention_mask entries should be 1")


if __name__ == "__main__":
    unittest.main()
