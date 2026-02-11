import json
from logoslabs.avp import encode_text_to_tensor

with open("outputs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        x = encode_text_to_tensor(item["prediction"])
        # x shape: (1, max_length)
        print(
            item.get("id"),
            "len=", x.shape[-1],
            "min=", float(x.min()),
            "max=", float(x.max()),
            "std=", float(x.std()),
        )