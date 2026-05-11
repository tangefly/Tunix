import json


def convert_sft_dataset(input_path, output_path):
    # 读取原始数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for item in data:

        instruction = item["instruction"]
        output = item["output"]

        new_item = {
            "messages": [
                {
                    "role": "user",
                    "content": instruction,
                },
                {
                    "role": "assistant",
                    "content": output,
                }
            ]
        }

        new_data.append(new_item)

    # 保存新格式
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            new_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Convert finished: {output_path}")


if __name__ == "__main__":

    convert_sft_dataset(
        input_path="/home/tanger/workspace/Tunix/data/ruozhiba_qa.json",
        output_path="/home/tanger/workspace/Tunix/data/ruozhiba.json",
    )