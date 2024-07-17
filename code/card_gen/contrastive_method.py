import threading

# TODO: fit this method to the new contrastive evaluator
from eval.eval_contrastive_full import *
from eval.eval_contrastive_answer import *
from tqdm import trange

from core.data import *
from core.models import *
from core.utils import *
from eval.eval_predictive import *


class ContrastiveMethod:
    # general
    experiment: str
    name: str
    rm: ResourceManager
    cm: CostManager

    # dataset related
    dataset_folder: str
    batch_nums: List[int]
    shuffle: bool
    seed: int
    training_batches: List[Batch]
    testing_batch: Batch

    # training related
    topic: str
    model_1: str
    model_2: str
    card_format: str
    initial_criteria: List[str]
    epoch: int
    cards: List[dict]  # includes the current and previous cards
    card: GenerativeCard
    CoT: bool  # if use CoT while training
    use_refine: bool  # if use refine while training

    # eval related
    async_eval: bool  # if evaluate while training
    eval_threads: List[threading.Thread]
    contrastive_evaluator: ContrastiveAnswerEvaluator

    @staticmethod
    def format_batch_str(batch_str, model_1_completions, model_2_completions):

        for i in range(len(model_1_completions)):
            completion_str = f"### Student A Completion:\n{model_1_completions[i]['completion']}\n\n### Student A Answer: {model_1_completions[i]['choice']}\n\n"
            completion_str += f"### Student B Completion:\n{model_2_completions[i]['completion']}\n\n## Student B Answer: {model_2_completions[i]['choice']}\n\n"
            place_holder = f"student_answer_{i}"
            batch_str = batch_str.replace(f"<{place_holder}>", completion_str)

        return batch_str

    def __init__(self, hp: Dict, cm: CostManager = None):
        self.experiment = hp["experiment"]
        self.name = hp["name"]
        if hp.get("load_from", None) is not None:
            self.rm = ResourceManager(existing_output_path=hp["load_from"])
        else:
            self.rm = ResourceManager(self.experiment, self.name)
        self.cm = cm if cm is not None else CostManager()
        self.meta = hp["dataset_folder"].split("/")[1]
        self.dataset_folder = hp["dataset_folder"]
        self.topic = hp["topic"]

        self.model_1 = hp["model1"]
        self.model_2 = hp["model2"]

        self.batch_nums = hp["batch_nums"]
        self.shuffle = hp["shuffle"]
        self.seed = hp["seed"]

        self.initial_criteria = hp["initial_criteria"]
        self.CoT = hp["CoT"]
        self.async_eval = hp["async_eval"]
        self.eval_threads = []
        self.use_refine = hp["use_refine"]
        self.epoch = hp["epoch"]
        self.evaluator_type = hp["evaluator"]

        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.evaluator_type == "claude":
            self.evaluator = CLAUDE_3_MODEL_NAME

        elif self.evaluator_type == "gpt":
            self.evaluator = GPT_4_MODEL_NAME

        else:
            raise ValueError(f"Unknown evaluator type: {self.evalutor_type}")

        # Format: dict, bullet_point (list), paragraph
        self.card_format = hp["card_format"]

        self.training_batches = {
            self.model_1: load_batches(
                self.dataset_folder,
                self.topic,
                self.model_1,
                "train",
                self.batch_nums[:-1],
                shuffle=self.shuffle,
            ),
            self.model_2: load_batches(
                self.dataset_folder,
                self.topic,
                self.model_2,
                "train",
                self.batch_nums[:-1],
                shuffle=self.shuffle,
            ),
        }

        if self.epoch is None:
            self.epoch = len(self.training_batches[self.model_1])
        self.testing_batch = {
            self.model_1: load_batches(
                self.dataset_folder,
                self.topic,
                self.model_1,
                "test",
                [self.batch_nums[-1]],
                shuffle=self.shuffle,
            )[0],
            self.model_2: load_batches(
                self.dataset_folder,
                self.topic,
                self.model_2,
                "test",
                [self.batch_nums[-1]],
                shuffle=self.shuffle,
            )[0],
        }

        self.cards = []

        # initial card
        if self.card_format == "str":
            self.card = {self.model_1: "", self.model_2: ""}
        elif self.card_format == "bullet_point":
            self.card = {self.model_1: [], self.model_2: []}
        else:
            self.card = {
                self.model_1: GenerativeCard(d={c: "" for c in self.initial_criteria}),
                self.model_2: GenerativeCard(d={c: "" for c in self.initial_criteria}),
            }

        type_ = "mmlu"
        if "mmlu" in self.dataset_folder:
            type_ = "mmlu"
        elif "anthropic" in self.dataset_folder:
            type_ = "anthropic"

        self.contrastive_evaluator = ContrastiveAnswerEvaluator(
            topic=self.topic,
            batches=(
                self.testing_batch[self.model_1],
                self.testing_batch[self.model_2],
            ),
            models=[self.model_1, self.model_2],
            cards=(self.card[self.model_1], self.card[self.model_2]),
            # evaluator_model='groq/llama3-70b-8192',
            evaluator_model="meta-llama/Meta-Llama-3-70B-Instruct",
            # evaluator_model='meta-llama/Meta-Llama-3-8B-Instruct',
            rm=self.rm,
        )

        self.rm.dump_dict("hyperparameters", hp)

    def main(self):
        self.train()

        # wait for all eval threads to finish
        for t in self.eval_threads:
            t.join()
        time.sleep(1.5)  # wait for json to write

        i = 0
        while self.rm.file_exists(f"cost_{i}"):
            continue
        self.rm.dump_dict(f"cost_{i}", self.cm.get_info_dict())

    def train(self):
        print(
            f"Training started on topic {self.topic} and models {self.model_1}, {self.model_2}"
        )

        for e in trange(self.epoch, desc="Training"):
            p_card, p_model = self.progress(e)
            time.sleep(5)

            # temp = copy.deepcopy(self.card) + p_card
            temp = self.add_card(copy.deepcopy(self.card), p_card)

            if self.card_format == "dict":
                word_count = max(temp[self.model_1].words(), temp[self.model_2].words())
                criteria_count = max(
                    temp[self.model_1].criteria_count(),
                    temp[self.model_2].criteria_count(),
                )
                if (
                    word_count > REG_WORD_LIM
                    or criteria_count > REG_CRITERIA_LIM
                    or e == self.epoch - 1
                ):
                    # if False:
                    regressive_card, regressive_model = self.regress(e, p_card)

                else:
                    if e == 0:
                        regressive_card = p_card
                    else:
                        # regressive_card = self.card + p_card
                        regressive_card = self.add_card(self.card, p_card)

                        assert regressive_card == temp

                    regressive_model = p_model

            else:
                if len(p_card) > REG_WORD_LIM or e == self.epoch - 1:
                    regressive_card, regressive_model = self.regress(e, p_card)

                else:
                    if e == 0:
                        regressive_card = p_card
                    else:
                        # regressive_card = self.card + p_card
                        # concat each entry in the dict

                        if type(p_card) == str:
                            p_card = p_card["card"]

                        if type(self.card) == str:
                            self.card = self.card["card"]

                        regressive_card = {}
                        for key in self.card.keys():
                            regressive_card[key] = self.card[key] + ";\n" + p_card[key]

                    regressive_model = p_model

            refine_card, _ = self.refine(e, regressive_card)
            # refine_card = regressive_card

            time.sleep(5)

            final_card = refine_card
            self.card = final_card
            self.cards.append(final_card)

            if self.card_format == "str":
                self.rm.dump_dict(f"cards/epoch_{e}_card", {"card": final_card})
            elif self.card_format == "bullet_point":
                self.rm.dump_dict(f"cards/epoch_{e}_card", final_card)
            else:
                self.rm.dump_dict(
                    f"cards/epoch_{e}_card",
                    {
                        self.model_1: final_card[self.model_1].to_dict(),
                        self.model_2: final_card[self.model_2].to_dict(),
                    },
                )

            # Now, only eval on first and last epoch
            if True:
                if e == self.epoch - 1:
                    predictive_eval_dict = self.contrastive_eval(
                        e, final_card, self.testing_batch, async_eval=self.async_eval
                    )

            # Temporary

            print(f"Epoch {e} Finished. Cost so far: {self.cm.get_cost()}")
            assert len(self.cards) == e + 1 and self.card == self.cards[-1]

    def contrastive_eval(
        self,
        e: int,
        card: Union[GenerativeCard, str, dict],
        batch: Batch,
        num_times: int = 1,
        batch_type: str = "test",
        async_eval: bool = False,
    ) -> Optional[Dict]:
        print(f"Epoch {e} Predicative Evaluating..., cost so far: {self.cm.get_cost()}")
        name = f"eval_{batch_type}_epoch_{e}_predictive"

        self.contrastive_evaluator.cards = (
            str(self.card[self.model_1]),
            str(self.card[self.model_2]),
        )

        if self.rm.file_exists(name):  # try to recover from file
            return self.rm.load_dict(name)
        if async_eval:
            max_worker = 60

            t = threading.Thread(target=self.contrastive_evaluator.main)

            self.eval_threads.append(t)
            t.start()
            return None

        else:
            max_worker = 60
            return self.contrastive_evaluator.main(max_workers=max_worker)

    def preferential_eval(
        self,
        e: int,
        card1: GenerativeCard,
        card2: GenerativeCard,
        batch: Batch,
        num_times: int = 1,
    ) -> Optional[Dict]:
        # TODO: update this
        print(
            f"Epoch {e} Preferential Evaluating..., cost so far: {self.cm.get_cost()}"
        )
        name = f"eval_test_epoch_{e}_preferential"
        if e == 0:
            return None
        if self.rm.file_exists(name):  # try to recover from file
            return self.rm.load_dict(name)
        if self.async_eval:
            raise NotImplementedError
        else:
            # TODO: now preferential evaluation main returns None, may change to dict
            return self.preferential_evaluator.main(
                name, batch, card1, card2, num_times=num_times
            )

    def progress(self, e: int) -> Tuple[GenerativeCard, GPTModel]:
        print(f"Epoch {e} Progressing..., cost so far: {self.cm.get_cost()}")
        # try to recover from file
        if self.rm.file_exists(f"cards/epoch_{e}_progressive_card"):
            new_card = {
                self.model_1: GenerativeCard(
                    d=self.rm.load_dict(f"cards/epoch_{e}_progressive_card")[
                        self.model_1
                    ]
                ),
                self.model_2: GenerativeCard(
                    d=self.rm.load_dict(f"cards/epoch_{e}_progressive_card")[
                        self.model_2
                    ]
                ),
            }

            if self.evaluator_type == "claude":
                p_model = ClaudeModel.from_messages(
                    self.rm.load_dict(f"epoch_{e}_progressive")["conversation"],
                    CLAUDE_3_MODEL_NAME,
                    self.cm,
                )

            elif self.evaluator_type == "gpt":
                p_model = GPTModel.from_messages(
                    self.rm.load_dict(f"epoch_{e}_progressive")["conversation"],
                    GPT_4_MODEL_NAME,
                    self.cm,
                )

            return new_card, p_model

        # progression
        batch_str_1 = self.training_batches[self.model_1][e].get_question_str()
        batch_str_2 = self.training_batches[self.model_2][e].get_question_str()

        # make sure same batch
        assert batch_str_1 == batch_str_2

        batch_str = batch_str_1

        model_1_completions = self.training_batches[self.model_1][
            e
        ].get_student_completions()
        model_2_completions = self.training_batches[self.model_2][
            e
        ].get_student_completions()

        batch_str = self.format_batch_str(
            batch_str, model_1_completions, model_2_completions
        )

        system_prompt = self.rm.get_prompt("contrastive/progressive/system").format(
            topic=self.topic
        )

        user_prompt = self.rm.get_prompt("contrastive/progressive/user").format(
            topic=self.topic, batch=batch_str
        )

        if self.evaluator_type == "claude":
            p_model = ClaudeModel(
                system_prompt, CLAUDE_3_MODEL_NAME, tm=self.cm
            )  # progressive model
        elif self.evaluator_type == "gpt":
            p_model = GPTModel(
                system_prompt, GPT_4_MODEL_NAME, tm=self.cm
            )  # progressive model

        if self.card_format == "str":
            formatting_prompt = self.rm.get_prompt(
                "contrastive/progressive/str_formatting"
            )
            use_json = True

        elif self.card_format == "bullet_point":
            formatting_prompt = self.rm.get_prompt(
                "contrastive/progressive/bp_formatting"
            )
            use_json = True

        else:
            formatting_prompt = self.rm.get_prompt(
                "contrastive/progressive/dict_formatting"
            )
            formatting_prompt = formatting_prompt.format(
                topic=self.topic, criteria=self.card[self.model_1].get_criteria_str()
            )

            use_json = True

        final_prompt = user_prompt + "-" * 50 + formatting_prompt

        json_obj = p_model(final_prompt, use_json=use_json)

        self.rm.dump_dict(
            f"epoch_{e}_progressive",
            {"step": "progressive", "conversation": p_model.messages},
        )

        card_dict = dict()

        if self.card_format == "str":
            new_card = {
                self.model_1: json_obj["card"]["A"],
                self.model_2: json_obj["card"]["B"],
            }
            card_dict = {"card": new_card}

        elif self.card_format == "bullet_point":
            # TODO: fix this
            new_card = json_obj
            card_dict = new_card

        else:
            # # criterion: Student A/B: items
            new_card, card_dict = self.format_card(json_obj)

        self.rm.dump_dict(f"cards/epoch_{e}_progressive_card", card_dict)

        return new_card, p_model

    def regress(
        self, e: int, p_card: GenerativeCard
    ) -> Tuple[GenerativeCard, Optional[GPTModel]]:
        if e == 0:  # no regression for the first epoch
            return p_card, None
        print(f"Epoch {e} Regressing..., cost so far: {self.cm.get_cost()}")
        # try to recover from file
        if self.rm.file_exists(f"cards/epoch_{e}_regressive_card"):
            # r_card = GenerativeCard(d=self.rm.load_dict(f'cards/epoch_{e}_regressive_card'))
            r_card = {
                self.model_1: GenerativeCard(
                    d=self.rm.load_dict(f"cards/epoch_{e}_regressive_card")[
                        self.model_1
                    ]
                ),
                self.model_2: GenerativeCard(
                    d=self.rm.load_dict(f"cards/epoch_{e}_regressive_card")[
                        self.model_2
                    ]
                ),
            }

            if self.evaluator_type == "claude":
                r_model = ClaudeModel.from_messages(
                    self.rm.load_dict(f"epoch_{e}_regressive")["conversation"],
                    CLAUDE_3_MODEL_NAME,
                    self.cm,
                )
            elif self.evaluator_type == "gpt":
                r_model = GPTModel.from_messages(
                    self.rm.load_dict(f"epoch_{e}_regressive")["conversation"],
                    GPT_4_MODEL_NAME,
                    self.cm,
                )

            return r_card, r_model

        # merge the two cards
        system_prompt = self.rm.get_prompt("contrastive/regressive/system").format(
            topic=self.topic
        )
        combined_cards = self.add_card(self.card, p_card)
        # user_prompt = self.rm.get_prompt('regressive/user').format(cards=combined_cards.get_all_card_str())
        card_str = f"Student A: {combined_cards[self.model_1].get_all_card_str()}\nStudent B: {combined_cards[self.model_2].get_all_card_str()}"
        user_prompt = self.rm.get_prompt("contrastive/regressive/user").format(
            cards=card_str
        )
        # user_prompt = self.rm.get_prompt('regressive/user').format(card_1=str(self.card), card_2=str(p_card))

        if self.evaluator_type == "claude":
            r_model = ClaudeModel(system_prompt, CLAUDE_3_MODEL_NAME, tm=self.cm)
        elif self.evaluator_type == "gpt":
            r_model = GPTModel(system_prompt, GPT_4_MODEL_NAME, tm=self.cm)

        if self.card_format == "str":
            formatting_prompt = self.rm.get_prompt(
                "contrastive/regressive/str_formatting"
            )
            use_json = True
        elif self.card_format == "bullet_point":
            formatting_prompt = self.rm.get_prompt(
                "contrastive/regressive/bp_formatting"
            )
            use_json = True
        else:
            formatting_prompt = self.rm.get_prompt(
                "contrastive/regressive/dict_formatting"
            )
            use_json = True

        final_prompt = user_prompt + "-" * 50 + formatting_prompt

        json_obj = r_model(final_prompt, use_json=use_json)
        self.rm.dump_dict(
            f"epoch_{e}_regressive",
            {"step": "regressive", "conversation": r_model.messages},
        )

        card_dict = dict()

        if self.card_format == "str":
            r_card = {
                self.model_1: json_obj["card"]["A"],
                self.model_2: json_obj["card"]["B"],
            }
            card_dict = {"card": r_card}
        elif self.card_format == "bullet_point":
            r_card = json_obj
            card_dict = r_card
        else:
            r_card, card_dict = self.format_card(json_obj)

        self.rm.dump_dict(f"cards/epoch_{e}_regressive_card", card_dict)
        return r_card, r_model

    def refine(
        self, e: int, regressive_card: GenerativeCard
    ) -> Tuple[GenerativeCard, Optional[GPTModel]]:
        if not self.use_refine:
            return regressive_card, None
        print(f"Epoch {e} Refining..., cost so far: {self.cm.get_cost()}")
        # try to recover from file
        if self.rm.file_exists(f"cards/epoch_{e}_refine_card"):
            info_dict = self.rm.load_dict(f"epoch_{e}_refine")
            if info_dict["decision"] == "regression":
                return regressive_card, None
            else:
                r_card = GenerativeCard(
                    d=self.rm.load_dict(f"cards/epoch_{e}_refine_card")
                )
                r_model = GPTModel.from_messages(
                    info_dict["refine_conversation"], GPT_4_MODEL_NAME, self.cm
                )
                return r_card, r_model
        info_dict = {
            "step": "refine",
            "eval_before": {},
            "refine_conversation": {},
            "eval_after": {},
        }
        # evaluate on the current training batch
        batch = self.training_batches[e]
        # choose to use mixtral because it's cheap
        eval_dict = self.predictive_eval(
            e, regressive_card, batch, num_times=1, batch_type="train_before_refine"
        )["iterations"][0]
        info_dict["eval_before"] = eval_dict
        accuracy, specificity, sensitivity, precision = (
            eval_dict["metrics"]["accuracy"],
            eval_dict["metrics"]["specificity"],
            eval_dict["metrics"]["sensitivity"],
            eval_dict["metrics"]["precision"],
        )
        print(
            f"Metrics (on train) before refine: {accuracy=}, {specificity=}, {sensitivity=}, {precision=}"
        )
        # if the eval accuracy is high enough, we skip the refine step
        if accuracy >= 0.8:
            return regressive_card, None
        eval_details = [
            eval_dict["details"].get(str(i), None) for i in range(len(batch))
        ]
        # format the eval information, including qa, model's response, eval result
        s = ""
        for i in range(len(batch)):
            if eval_details[i] is None:
                continue
            s += f"Question:\n{batch.get_question(i)}\n"
            s += f"Ground Truth Answer: {get_choice_str(batch.get_true_answer(i))}\n"
            s += f"Student's Answer: {get_choice_str(batch.get_model_answer(i))}\n"
            s += f"Student's Reasoning:\n{batch.get_model_reasoning(i)}\n"
            evaluator_guess = eval_details[i]["conversation"][-1]["content"]
            if evaluator_guess[
                "I believe the student can correctly answer the question"
            ]:
                s += f"TA's Guess: Yes, the student will answer this question correctly.\n"
            else:
                s += f"TA's Guess: No, the student will not answer this question correctly.\n"
            s += f"TA's Reasoning:\n{evaluator_guess['reasoning']}\n"
            s += f"\n{SMALL_SEPARATOR}\n"
        # pass the eval information to the refine model (GPT4) to refine the card
        system_prompt = self.rm.get_prompt("refine/system").format(topic=self.topic)
        refine_model = GPTModel(system_prompt, GPT_4_MODEL_NAME)
        user_prompt = self.rm.get_prompt("refine/user").format(s=s)
        json_obj = refine_model(user_prompt, use_json=True)
        refine_card = GenerativeCard(d=json_obj)
        self.rm.dump_dict(f"cards/epoch_{e}_refine_card", refine_card.to_dict())
        info_dict["refine_conversation"] = refine_model.messages
        # eval on the refined card
        eval_dict = self.predictive_eval(
            e, refine_card, batch, num_times=1, batch_type="train_after_refine"
        )["iterations"][0]
        info_dict["eval_after"] = refine_model.messages
        new_accuracy, new_specificity, new_sensitivity, new_precision = (
            eval_dict["metrics"]["accuracy"],
            eval_dict["metrics"]["specificity"],
            eval_dict["metrics"]["sensitivity"],
            eval_dict["metrics"]["precision"],
        )
        print(
            f"Metrics (on train) after refine: {new_accuracy=}, {new_specificity=}, {new_sensitivity=}, {new_precision=}"
        )
        # if the eval accuracy is lower than before, we keep the old card
        info_dict["decision"] = "regression" if new_accuracy < accuracy else "refine"
        self.rm.dump_dict(f"epoch_{e}_refine", info_dict)
        if new_accuracy < accuracy:
            return regressive_card, refine_model
        else:
            return refine_card, refine_model
        # TODO: if low then do refine again?

    def summarize(self, card: GenerativeCard, indices: List[int] = None) -> str:
        if indices is None:
            indices = list(range(len(card.criteria)))
        s = ""
        for i in indices:
            if card.get_summarization(i) is not None:
                s += card.get_summarization(i) + "\n"
            system_prompt = self.rm.get_prompt("summarize/system_prompt")
            model = select_model(
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", system_prompt
            )
            user_prompt = self.rm.get_prompt("summarize/user_prompt")
            response = model(user_prompt)
            card.set_summarization(i, response)
            s += response + "\n"
        return s

    def format_card(self, json_obj):
        model_1_card = {}
        model_2_card = {}
        # print(json_obj)
        for value, key in json_obj.items():
            A_dict = key["Student A"]
            B_dict = key["Student B"]

            # replace all `student a with It`
            for k, v in A_dict.items():
                v = v.replace("Student A", "It")
                v = v.replace("student A", "it")
                A_dict[k] = v

            for k, v in B_dict.items():
                v = v.replace("Student B", "It")
                v = v.replace("student B", "it")
                B_dict[k] = v

            model_1_card[value] = A_dict
            model_2_card[value] = B_dict

        new_card = {
            self.model_1: GenerativeCard(d=model_1_card),
            self.model_2: GenerativeCard(d=model_2_card),
        }

        card_dict = {
            self.model_1: new_card[self.model_1].to_dict(),
            self.model_2: new_card[self.model_2].to_dict(),
        }

        return new_card, card_dict

    def add_card(self, card1, card2):

        # if both empty:
        if (
            self.model_1 not in card1
            or self.model_2 not in card1
            or self.model_2 not in card1
            or self.model_2 not in card2
        ):
            return {self.model_1: GenerativeCard(), self.model_2: GenerativeCard()}
        # case 1: card 1 empty
        if self.model_1 not in card1 or self.model_2 not in card1:
            return card2

        if self.model_1 not in card2 or self.model_2 not in card2:
            return card1

        card1[self.model_1] += card2[self.model_1]
        card1[self.model_2] += card2[self.model_2]
        return card1


def load_method_instance(
    folder: str, new_instance: bool = False, copy_info: bool = True
) -> ContrastiveMethod:
    with open(os.path.join(folder, "hyperparameters.json")) as f:
        hp = json.load(f)
    if not new_instance:
        hp["load_from"] = folder
    else:
        hp["load_from"] = None
        hp["async_eval"] = False
    method = ContrastiveMethod(hp)
    if new_instance and copy_info:
        shutil.copytree(folder, method.rm.output_folder_path, dirs_exist_ok=True)
        # delete all files starts with eval
        eval_files = [
            f for f in os.listdir(method.rm.output_folder_path) if f.startswith("eval")
        ]
        for f in eval_files:
            os.remove(os.path.join(method.rm.output_folder_path, f))
    return method
