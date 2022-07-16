import torch
from torchmetrics import Metric


class MeanAveragePrecision(Metric):
    full_state_update=True
    def __init__(self, k=100):
        super(MeanAveragePrecision, self).__init__()
        self.add_state("embeddings", default=torch.Tensor([]))
        self.add_state("vehicle_ids", default=torch.Tensor([]))
        self.k = k

    def _average_precision(self, ranked_list: torch.Tensor):
        precisions_of_relevant_elements = []
        relevant_count = 0
        for all_count, el in enumerate(ranked_list[:self.k], start=1):
            if el == 1:
                relevant_count += 1
                precisions_of_relevant_elements.append(relevant_count / all_count)
        if len(precisions_of_relevant_elements) == 0:
            return 0.0
        return sum(precisions_of_relevant_elements) / len(precisions_of_relevant_elements)

    def _mean_average_precision(self, ranked_lists: torch.Tensor):
        apks = []
        for ranked_list in ranked_lists:
            apk = self._average_precision(ranked_list)
            apks.append(apk)
        return sum(apks) / len(apks)

    def _get_ranked_list(self):
        distances = torch.cdist(x1=self.embeddings, x2=self.embeddings)
        ranked_list = []
        for i, (vid, dists) in enumerate(zip(self.vehicle_ids, distances)):
            values, indices = dists.sort(descending=False)
            ranked_list.append(torch.tensor(self.vehicle_ids[indices] == vid, dtype=torch.int)[1:])
        return torch.stack(ranked_list)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.embeddings = torch.cat([self.embeddings, preds])
        self.vehicle_ids = torch.cat([self.vehicle_ids, target])

    def compute(self):
        pass

    def compute_final(self):
        ranked_list = self._get_ranked_list()
        return self._mean_average_precision(ranked_list)


class RankOne(Metric):
    def __init__(self):
        super(RankOne, self).__init__()
        self.add_state("embeddings", default=torch.Tensor([]))
        self.add_state("vehicle_ids", default=torch.Tensor([]))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.embeddings = torch.cat([self.embeddings, preds])
        self.vehicle_ids = torch.cat([self.vehicle_ids, target])

    def _get_ranked_list(self):
        distances = torch.cdist(x1=self.embeddings, x2=self.embeddings)
        ranked_list = []
        for i, (vid, dists) in enumerate(zip(self.vehicle_ids, distances)):
            values, indices = dists.sort(descending=False)
            ranked_list.append(torch.tensor(self.vehicle_ids[indices] == vid, dtype=torch.int)[1:])
        return torch.stack(ranked_list)

    def compute(self):
        pass

    def compute_final(self):
        ranked_list = self._get_ranked_list()
        results = []
        for rank in ranked_list:
            if rank.sum() == 0:
                results.append(0)
            else:
                idx = (rank == 1).nonzero(as_tuple=True)[0][0] + 1
                results.append(1 / idx)
        return sum(results) / len(results)


class Visualizator(Metric):
    def __init__(self):
        super(Visualizator, self).__init__()
        self.add_state("images", default=torch.Tensor([]))
        self.add_state("embeddings", default=torch.Tensor([]))
        self.add_state("vehicle_ids", default=torch.Tensor([]))

    def update(self, images: torch.Tensor, preds: torch.Tensor, target: torch.Tensor):
        self.images = torch.cat([self.images.cpu(), images.cpu()])
        self.embeddings = torch.cat([self.embeddings, preds])
        self.vehicle_ids = torch.cat([self.vehicle_ids, target])

    def _get_ranked_vid_list(self):
        distances = torch.cdist(x1=self.embeddings, x2=self.embeddings)
        ranked_list = []
        for i, (vid, dists) in enumerate(zip(self.vehicle_ids, distances)):
            values, indices = dists.sort(descending=False)
            ranked_list.append(torch.tensor(self.vehicle_ids[indices] == vid, dtype=torch.int)[1:])
        return torch.stack(ranked_list)

    def compute(self):
        pass

    def get_images(self, idx, n=5):
        distances = torch.cdist(x1=self.embeddings, x2=self.embeddings)
        vid, dists = self.vehicle_ids[idx], distances[idx]
        values, indices = dists.sort(descending=False)
        return self.images[indices][:n], self.vehicle_ids[indices][:n]

def average_precision_at_k(ranked_list, k):
    precisions_of_relevant_elements = []
    relevant_count = 0
    for all_count, el in enumerate(ranked_list[:k], start=1):
        if el == 1:
            relevant_count += 1
            precisions_of_relevant_elements.append(relevant_count / all_count)
    if len(precisions_of_relevant_elements) == 0:
        return 0.0
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
    ]

    embeddings = torch.tensor([
        [[10],[10]], [[55],[55]], [[15],[15]], [[50],[50]], [[45],[45]]
    ])

    vids = torch.tensor([
        [15], [69], [14], [69], [69]
    ])

    metric = MeanAveragePrecision()

    for e, v in zip(embeddings, vids):
        print(e,v)
        metric(e,v)


    print(metric.compute_final())

    # print(mean_average_precision_at_k(ranked_list, 5))


if __name__ == "__main__":
    main()
