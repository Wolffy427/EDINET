import json
import argparse
from edinet_bench.model import MODEL_TABLE


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result/company_name_prediction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Model name",
        choices=MODEL_TABLE.keys(),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    path = f"{args.output_dir}/{args.model}/prediction.jsonl"

    labels = []
    predictions = []

    with open(path, mode="r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            labels.append(data["label"])
            predictions.append(data["prediction"])

    print(labels[0])
    print(predictions[0])

    correct_num = 0

    for label, prediction in zip(labels, predictions):
        if label in prediction:
            print(label, prediction)
            correct_num += 1

    acc = correct_num / len(labels)
    print(round(acc, 4))
