import torch
import torchmetrics


def average_precision_at_k(ranked_list, k):
    precisions_of_relevant_elements = []
    relevant_count = 0
    for all_count, el in enumerate(ranked_list[:k], start=1):
        if el == 1:
            relevant_count += 1
            precisions_of_relevant_elements.append(relevant_count / all_count)
    return sum(precisions_of_relevant_elements) / len(precisions_of_relevant_elements)


def mean_average_precision_at_k(ranked_lists, k):
    apks = []
    for ranked_list in ranked_lists:
        apk = average_precision_at_k(ranked_list, k)
        print(apk)
        apks.append(apk)
    return sum(apks) / len(apks)


def main():
    ranked_list = [
        [0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],  # TODO div by zero
        [1, 1, 1, 1, 1],
    ]
    print(mean_average_precision_at_k(ranked_list, 5))


if __name__ == "__main__":
    main()
