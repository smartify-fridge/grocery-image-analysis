"""
Scores use_model.py output against hand-labeled fridge photos.

Command to run (after use_model.py):
    python evaluate_fridge.py
    python evaluate_fridge.py --detections model_output/detections.csv --ground-truth fridge_ground_truth.csv

Ground truth is one row per class per photo: image,class,count,verify,notes.
Rows with an empty class are items the model has no class for; they're listed but not scored.

An item counts as correct when the model finds that class in that photo. Counts matter:
if a photo has 2 egg cartons and the model finds 1, that's 1 correct and 1 missed.
With --presence, counts are ignored: each class either is or isn't in the photo.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_ground_truth(path, class_names):
    truth = defaultdict(Counter)
    unscored = defaultdict(list)
    unverified = 0

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = row["image"].strip()
            name = row["class"].strip()

            if not name:
                unscored[image].append(row["notes"].strip())
                continue

            if name not in class_names:
                raise ValueError(f"{path}: '{name}' ({image}) is not one of the model's classes")

            truth[image][name] += int(row["count"])
            if row.get("verify", "").strip().lower() == "yes":
                unverified += 1

    return truth, unscored, unverified


def load_predictions(path):
    predictions = defaultdict(Counter)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if rows and "status" not in rows[0]:
        print(f"WARNING: {path} has no status column (made before duplicate filtering),")
        print("         so every accepted detection is counted and duplicates will show as wrong.\n")
        kept = [r for r in rows if r["accepted"] == "True"]
    else:
        kept = [r for r in rows if r["status"] == "kept"]

    for row in kept:
        predictions[row["image"]][row["classifier_label"]] += 1

    return predictions


def score(truth, predicted):
    correct = sum((truth & predicted).values())
    return correct, sum(predicted.values()) - correct, sum(truth.values()) - correct


def rates(correct, wrong, missed):
    precision = correct / (correct + wrong) if correct + wrong else 0
    recall = correct / (correct + missed) if correct + missed else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Score use_model.py output against labeled fridge photos.")
    parser.add_argument("--detections", default="model_output/detections.csv")
    parser.add_argument("--ground-truth", default="fridge_ground_truth.csv")
    parser.add_argument("--classes", default="trained_model/class_names.json")
    parser.add_argument("--output", default="model_output/evaluation.csv")
    parser.add_argument("--presence", action="store_true",
                        help="ignore counts; score whether each class is found in each photo")
    args = parser.parse_args()

    class_names = set(json.loads(Path(args.classes).read_text(encoding="utf-8")))
    truth, unscored, unverified = load_ground_truth(args.ground_truth, class_names)
    predictions = load_predictions(args.detections)

    if args.presence:
        truth = defaultdict(Counter, {image: Counter(set(c)) for image, c in truth.items()})
        predictions = defaultdict(Counter, {image: Counter(set(c)) for image, c in predictions.items()})

    missing = sorted(set(truth) - set(predictions))
    if missing:
        print(f"WARNING: no detections for {len(missing)} labeled photo(s), scored as all missed: {missing}\n")

    unknown = {label for counts in predictions.values() for label in counts} - class_names
    if unknown:
        print(f"WARNING: predicted labels that aren't in {args.classes} (wrong weights or class list?):")
        print(f"         {sorted(unknown)[:10]}{' ...' if len(unknown) > 10 else ''}\n")

    rows = []
    totals = [0, 0, 0]
    per_class = defaultdict(lambda: [0, 0, 0])

    for image in sorted(truth):
        actual, found = truth[image], predictions.get(image, Counter())
        correct, wrong, missed = score(actual, found)
        totals = [t + v for t, v in zip(totals, (correct, wrong, missed))]
        precision, recall, f1 = rates(correct, wrong, missed)

        for name in set(actual) | set(found):
            c, w, m = score(Counter({name: actual[name]}), Counter({name: found[name]}))
            per_class[name] = [t + v for t, v in zip(per_class[name], (c, w, m))]

        print(f"{image}: {correct} correct, {wrong} wrong, {missed} missed "
              f"(precision {precision:.0%}, recall {recall:.0%})")
        print(f"  actual:     {format_counts(actual)}")
        print(f"  model said: {format_counts(found)}")
        if unscored[image]:
            print(f"  not scored: {'; '.join(unscored[image])}")
        print()

        rows.append({
            "image": image, "correct": correct, "wrong": wrong, "missed": missed,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "actual": format_counts(actual), "predicted": format_counts(found),
        })

    precision, recall, f1 = rates(*totals)
    print("=" * 65)
    print(f"OVERALL on {len(truth)} photos: {totals[0]} correct, {totals[1]} wrong, {totals[2]} missed")
    print(f"  Precision {precision:.1%}  (of the items the model listed, how many were really there)")
    print(f"  Recall    {recall:.1%}  (of the items really there, how many the model found)")
    print(f"  F1        {f1:.1%}")
    if unverified:
        print(f"\nNOTE: {unverified} ground-truth rows are marked verify=yes. Check them before reporting these numbers.")

    print("\nPer class (correct / wrong / missed):")
    for name, (c, w, m) in sorted(per_class.items(), key=lambda x: -(x[1][1] + x[1][2])):
        print(f"  {name:22} {c:3} / {w:3} / {m:3}")

    rows.append({
        "image": "OVERALL", "correct": totals[0], "wrong": totals[1], "missed": totals[2],
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "actual": "", "predicted": "",
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.output}")


def format_counts(counts):
    return ", ".join(f"{name} x{n}" if n > 1 else name for name, n in sorted(counts.items())) or "(nothing)"


if __name__ == "__main__":
    main()
