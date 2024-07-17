from tqdm import trange

from core.data import *
from core.models import *
from eval.eval_predictive import *


class FewShotMethod:
    # general
    experiment: str
    name: str
    rm: ResourceManager
    # dataset related
    dataset_folder: str
    batch_nums: List[int]
    shuffle: bool
    seed: int
    training_batches: List[MMLUBatch]
    validation_batch: MMLUBatch
    testing_batch: MMLUBatch
    # training related
    topic: str
    model: str
    # eval related
    predictive_evaluator: PredictiveEvaluator
    # misc
    token_manager: CostManager

    def __init__(
        self, experiment: str, name: str, hp: Dict, token_manager: CostManager
    ):
        self.experiment = experiment
        self.name = name
        if hp.get("load_from", None) is not None:
            self.rm = ResourceManager(existing_output_path=hp["load_from"])
        else:
            self.rm = ResourceManager(experiment, f"{name}")

        self.token_manager = token_manager
        self.dataset_folder = hp["dataset_folder"]
        self.topic = hp["topic"]
        self.model = hp["model"]
        self.batch_nums = hp["batch_nums"]
        self.shuffle = hp["shuffle"]
        self.seed = hp["seed"]
        random.seed(self.seed)

        # last two batches are validation and testing
        batches = load_mmlu_batches(
            self.dataset_folder,
            self.topic,
            self.batch_nums,
            shuffle=self.shuffle,
            model=self.model,
        )
        self.training_batches = batches[:-2]
        self.validation_batch = batches[-2]
        self.testing_batch = batches[-1]

        # eval
        self.predictive_evaluator = PredictiveEvaluator(
            self.topic, self.model, self.rm, "few-shot", tm=self.token_manager
        )

        self.rm.dump_dict("hyperparameters", hp)

    def main(self):
        self.train()
        self.predictive_evaluator.plot(
            "eval_validation",
            len(self.training_batches),
            self.validation_batch.get_accuracy(self.model),
        )
        self.rm.dump_dict("cost", self.token_manager.get_info_dict())

    def train(self):
        epoch = len(self.training_batches)
        for i in trange(epoch, desc="Training"):
            # card_str = combined_train_batch.sample(10).get_training_str()
            # print(card_str)
            card_str = self.training_batches[i].get_train_str()

            print(
                f"Epoch {i} Evaluating..., cost so far: {self.token_manager.get_cost()}"
            )
            if self.rm.file_exists(f"eval_validation_epoch_{i}_predictive.json"):
                print(f"Epoch {i} already evaluated, skipping...")
                continue
            self.predictive_evaluator.main(
                f"eval_validation_epoch_{i}_predictive",
                self.validation_batch,
                card_str,
                num_times=1,
            )

            print(f"Epoch {i} Finished. Cost so far: {self.token_manager.get_cost()}")


def load_few_shot_instance(folder: str, new_instance: bool = False) -> FewShotMethod:
    with open(os.path.join(folder, "hyperparameters.json")) as f:
        hp = json.load(f)
    if not new_instance:
        hp["load_from"] = folder
    token_manager = CostManager()
    folder_names = folder.split("/")
    method = FewShotMethod(
        folder_names[-2], folder_names[-1].split("_")[1], hp, token_manager
    )
    return method
