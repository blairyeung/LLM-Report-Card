# TODO
def eval(self, name: str, batch: MMLUBatch, num_times: int = 5, eval_method: Callable = None, **kwargs):
    """
    :param name: name of the evaluation
    :param batch: batch to evaluate
    :param num_times: number of times to perform the evaluation
    :param eval_method: evaluation method
    :param sample_size: sample size for predictive evaluation
    :param sample_sizes: sample sizes for likert evaluation
    """
    print(f'Evaluating {name}...')
    if eval_method is None:
        eval_method = self.eval_predictive
    info_dict = {
        'iterations': {}
    }
    metrics = []
    for j in range(num_times):  # perform evals
        if eval_method == self.eval_likert:
            sub_info, accuracy, coverage = self.eval_likert(batch, kwargs.get('sample_sizes', None))
            info_dict['iterations'][str(j)] = sub_info
            metrics.append((accuracy, coverage))
        else:
            raise NotImplementedError

    if eval_method == self.eval_likert:
        accuracies = [a for a, _ in metrics]
        coverages = [c for _, c in metrics]
        info_dict['metrics'] = {
            'accuracies': accuracies,
            'coverages': coverages,
            'mean_accuracy': np.mean(accuracies),
            'mean_coverage': np.mean(coverages),
            'sd_accuracy': np.std(accuracies),
            'sd_coverage': np.std(coverages)
        }
        print(f'{name}\n'
              f'accuracies: {accuracies}, mean={np.mean(accuracies)}, std={np.std(accuracies)}\n'
              f'coverages: {coverages}, mean={np.mean(coverages)}, std={np.std(coverages)}\n'
              f'cost so far: {self.token_manager.get_cost()}')
    else:
        raise NotImplementedError

    self.rm.dump_dict(name, info_dict)


def eval_likert(self, batch: MMLUBatch, sample_sizes: List[int] = None):
    if sample_sizes is None:
        sample_sizes = [10] * 5
    info_dict = {
        'method': 'likert',
        'details': {}
    }
    count = 0
    accuracy = []
    coverage = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = []
        for i, sample_size in enumerate(sample_sizes):
            start = sum(sample_sizes[:i])
            futures.append(executor.submit(self.eval_helper_likert, batch, list(range(start, start + sample_size))))
        for future in tqdm(as_completed(futures)):
            if future.result() is None:
                continue
            count += 1
            a, c, model, index = future.result()
            accuracy.append(a)
            coverage.append(c)
            self.token_manager.record_tokens(model)
            info_dict['details'][str(index)] = {
                "accuracy": a,
                "coverage": c,
                "conversation": model.messages
            }
    info_dict['statistics'] = {
        'raw_accuracies': accuracy,
        'raw_coverage': coverage,
        'mean_accuracy': np.mean(accuracy),
        'mean_coverage': np.mean(coverage),
        'sd_accuracy': np.std(accuracy),
        'sd_coverage': np.std(coverage)
    }
    return info_dict, np.mean(accuracy), np.mean(coverage)


def eval_helper_likert(self, batch: MMLUBatch, indices: List[int]):
    """
    Method: Let GPT4 rate the card based on the coverage of the card for the given batch.
    """
    system_prompt = self.rm.get_prompt('eval/coverage/system').format(topic=self.topic)
    model = GPTModel(system_prompt, GPT_4_MODEL_NAME)
    user_prompt = self.rm.get_prompt('eval/coverage/user').format(
        card=str(self.card), qa=batch.get_eval_coverage_batch_str(indices),
        prev_card=str(self.cards[0]), prev_card_acc=5, prev_card_cov=2)
    print(user_prompt)
    raw = model(user_prompt, use_json=True)
    try:
        obj = json.loads(raw)
        accuracy_probs = obj['accuracy']
        coverage_probs = obj['coverage']
        assert len(accuracy_probs) == len(coverage_probs) == 10
        accuracy_score = sum((i + 1) * p for i, p in enumerate(accuracy_probs))
        coverage_score = sum((i + 1) * p for i, p in enumerate(coverage_probs))
        return accuracy_score, coverage_score, model, indices
    except Exception as e:  # if failed, return None
        print(e)
        return None


def eval_plot_likert(self):
    output_folder = self.rm.output_folder_path
    epoch = len(self.training_batches)
    epochs = list(range(1, epoch + 1))
    accuracies = []
    coverages = []
    for e in range(epoch):
        filename = output_folder + f'/eval_validation_epoch_{e}_likert.json'
        with open(filename) as f:
            json_obj = json.load(f)
        json_obj = json_obj['metrics']
        accuracies.append(json_obj['accuracies'])
        coverages.append(json_obj['coverages'])

    # Calculate means and standard deviations for each epoch
    mean_accuracies = [np.mean(epoch_data) for epoch_data in accuracies]
    std_accuracies = [np.std(epoch_data) for epoch_data in accuracies]
    mean_coverages = [np.mean(epoch_data) for epoch_data in coverages]
    std_coverages = [np.std(epoch_data) for epoch_data in coverages]

    # Plot each metric
    plt.figure(figsize=(10, 6))

    # Accuracies
    plt.plot(epochs, mean_accuracies, label='Accuracy', color='blue')
    plt.fill_between(epochs, np.array(mean_accuracies) - np.array(std_accuracies), np.array(mean_accuracies) + np.array(std_accuracies), color='blue', alpha=0.2)

    # Coverages
    plt.plot(epochs, mean_coverages, label='Coverage', color='green')
    plt.fill_between(epochs, np.array(mean_coverages) - np.array(std_coverages), np.array(mean_coverages) + np.array(std_coverages), color='green', alpha=0.2)

    # Display values
    for i in range(len(epochs)):
        plt.text(epochs[i], mean_accuracies[i], f'{mean_accuracies[i]:.2f}', ha='center', va='bottom', color='blue')
        plt.text(epochs[i], mean_coverages[i], f'{mean_coverages[i]:.2f}', ha='center', va='bottom', color='green')

    # Adding legend and labels
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Likert Evaluation Metrics Over Epochs')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(self.rm.output_folder_path, 'likert_metrics.png'))
    plt.show()