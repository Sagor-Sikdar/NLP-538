from a1_p1_sikdar_114614579 import *


vocabulary = {}

def get_target_index(words):
    pattern = r"<<(.*receptor.*)|(.*reduction.*)|(.*rate.*)|(.*reserve.*)|(.*reason.*)|(.*return.*)>>"
    for index, word in enumerate(words):
        if re.search(pattern, word, re.IGNORECASE):
            return index

def extractLexicalFeatures(tokens, target, tokenizer_type):
    global vocabulary
    prev_encoding, next_encoding, multi_hot_encoding = [0] * 501, [0] * 501, [0] * 501

    target_name = tokens[target]
    vocab = vocabulary[target_name][tokenizer_type]

    if target > 0:
        if tokens[target - 1] in vocab:
            prev_encoding[vocab[tokens[target - 1]]] = 1
        else:
            prev_encoding[500] = 1

    if target < len(tokens) - 1:
        if tokens[target + 1] in vocab:
            next_encoding[vocab[tokens[target + 1]]] = 1
        else:
            next_encoding[500] = 1

    for index, token in enumerate(tokens):
        if target == index:
            continue

        if token in vocab:
            multi_hot_encoding[vocab[token]] = 1
        else:
            multi_hot_encoding[500] = 1

    features = np.array(prev_encoding + next_encoding + multi_hot_encoding)
    return features

def extractImprovedLexicalFeatures(tokens, target, tokenizer_type):
    global vocabulary
    prev_encoding, next_encoding, multi_hot_encoding = [0] * 501, [0] * 501, [0] * 501

    target_name = tokens[target]
    vocab = vocabulary[target_name][tokenizer_type]

    if target > 0:
        if tokens[target - 1] in vocab:
            prev_encoding[vocab[tokens[target - 1]]] = 1
        else:
            prev_encoding[500] = 1

    if target < len(tokens) - 1:
        if tokens[target + 1] in vocab:
            next_encoding[vocab[tokens[target + 1]]] = +1
        else:
            next_encoding[500] = +1

    for index, token in enumerate(tokens):
        if target == index:
            continue

        if token in vocab:
            multi_hot_encoding[vocab[token]] = 1
        else:
            multi_hot_encoding[500] = 1

    features = np.array(prev_encoding + next_encoding + multi_hot_encoding)
    return features

def get_feature_label(word_sense, dataset, tokenizer_type):
    global vocabulary
    vocab = vocabulary[word_sense][tokenizer_type]

    X, y = [], []
    for text, label in dataset:
        words = text.split()
        target_index = get_target_index(words)

        prev_text, next_text = ' '.join(words[:target_index]), ' '.join(words[target_index + 1 : ])
        prev_token, next_token = [], []

        if tokenizer_type == "bpe":
            prev_token, next_token = spacelessBPETokenize(prev_text, vocab), spacelessBPETokenize(next_text, vocab)
        else:
            prev_token, next_token = wordTokenizer(prev_text), wordTokenizer(next_text)

        tokens = prev_token + [word_sense] + next_token
        target_index = len(prev_token)

        features = extractLexicalFeatures(tokens, target_index, tokenizer_type)

        X.append(features)
        y.append(label)

    # y_mapping from 0 to n types
    unique_y = list(np.unique(y))
    y_map = {value: index for index, value in enumerate(unique_y)}
    y = [y_map[value] for value in y]

    return X, y


def get_improved_feature_label(word_sense, dataset, tokenizer_type):
    global vocabulary
    vocab = vocabulary[word_sense][tokenizer_type]

    X, y = [], []
    for text, label in dataset:
        words = text.split()
        target_index = get_target_index(words)

        prev_text, next_text = ' '.join(words[:target_index]), ' '.join(words[target_index + 1 : ])
        prev_token, next_token = [], []

        if tokenizer_type == "bpe":
            prev_token, next_token = spacelessBPETokenize(prev_text, vocab), spacelessBPETokenize(next_text, vocab)
        else:
            prev_token, next_token = wordTokenizer(prev_text), wordTokenizer(next_text)

        tokens = prev_token + [word_sense] + next_token
        target_index = len(prev_token)

        features = extractImprovedLexicalFeatures(tokens, target_index, tokenizer_type)

        X.append(features)
        y.append(label)

    # y_mapping from 0 to n types
    unique_y = list(np.unique(y))
    y_map = {value: index for index, value in enumerate(unique_y)}
    y = [y_map[value] for value in y]

    return X, y

def tokenize_dataset(word_sense, train_set, dev_set, tokenizer_type):
    X, y = get_feature_label(word_sense, train_set[word_sense], tokenizer_type)
    X_train, y_train = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    X, y = get_feature_label(word_sense, dev_set[word_sense], tokenizer_type)
    X_dev, y_dev = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    return X_train, y_train, X_dev, y_dev

def improved_tokenize_dataset(word_sense, train_set, dev_set, tokenizer_type):
    X, y = get_improved_feature_label(word_sense, train_set[word_sense], tokenizer_type)
    X_train, y_train = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    X, y = get_improved_feature_label(word_sense, dev_set[word_sense], tokenizer_type)
    X_dev, y_dev = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    return X_train, y_train, X_dev, y_dev


#read from file and send the data
def extract_train_dev_set(data_file):
    global vocabulary

    train_set = {}
    dev_set = {}
    corpus = {}

    train_itr, dev_itr = data_file[:3968], data_file[3968:]

    for test in train_itr:
        text_split = test.split('\t')
        label = text_split[2]
        text = remove_non_ascii(text_split[3]).rstrip()

        # Define the regular expression pattern for label
        label_pattern = r'(\w+)%(\d+:\d{2}:\d{2})+::'
        matches = re.findall(label_pattern, label)

        word_type, sense = matches[0][0], matches[0][1]
        if word_type not in train_set:
            train_set[word_type] = []
            vocabulary[word_type] = {"bpe" : {}, "word": {}}
            corpus[word_type] = ""

        train_set[word_type].append((text, sense))
        corpus[word_type] += text


    for test in dev_itr:
        text_split = test.split('\t')
        label = text_split[2]
        text = remove_non_ascii(text_split[3]).rstrip()

        # Define the regular expression pattern for label
        label_pattern = r'(\w+)%(\d+:\d{2}:\d{2})+::'
        matches = re.findall(label_pattern, label)

        word_type, sense = matches[0][0], matches[0][1]
        if word_type not in dev_set:
            dev_set[word_type] = []

        dev_set[word_type].append((text, sense))

    for word_type in corpus:
        vocabulary[word_type]["bpe"], vocabulary[word_type]["word"] = most_freq_vocabulary(corpus[word_type], "bpe", 500), most_freq_vocabulary(corpus[word_type], "word", 500)
        vocabulary[word_type]["bpe"] = {value[0]:index for index, value in enumerate(vocabulary[word_type]["bpe"])}
        vocabulary[word_type]["word"] = {value[0]:index for index, value in enumerate(vocabulary[word_type]["word"])}
        vocabulary[word_type]["bpe"]["unk_word"], vocabulary[word_type]["word"]["unk_word"] = 500, 500

    return train_set, dev_set


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiClassLogisticRegression(nn.Module):
  def __init__(self, num_feats, num_classes, dropout_rate=0.0,
                 learn_rate = 0.01, device = torch.device("cpu") ):
    #the constructor; define any layer objects (e.g. Linear)
    super(MultiClassLogisticRegression, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(num_feats+1, num_classes)

  def forward(self, X):
    # Ensure X is 2-dimensional: [batch_size, num_features]
    X = self.dropout(X)
    if X.dim() != 2:
        raise ValueError("Expected X to be a 2-dimensional tensor")

    # Add a column of ones to X to act as an intercept (bias) term.
    # The ones tensor must match the device and dtype of X.
    ones = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
    newX = torch.cat((X, ones), dim=1)

    # Apply the linear layer and return its output.
    return  self.linear(newX)


def trainLogReg(train_corpus):
    X_train = torch.from_numpy(train_corpus[0])
    y_train = torch.from_numpy(train_corpus[1]).long()
    X_dev = torch.from_numpy(train_corpus[2])
    y_dev = torch.from_numpy(train_corpus[3]).long()

    model = MultiClassLogisticRegression(len(X_train[0]), num_classes=train_corpus[4])

    loss_function = nn.CrossEntropyLoss()

    optimizer = None
    if train_corpus[6] == False:
        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=10)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=train_corpus[5])

    train_losses = []
    dev_losses = []
    epochs = 75
    batch_size = 128

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])

        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_x)
            loss = loss_function(outputs, batch_y.long())
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            epoch_loss += loss.item() * batch_x.size(0)

        train_losses.append(epoch_loss / len(X_train))

        model.eval()
        with torch.no_grad():
            dev_preds = model(X_dev)
            dev_loss = loss_function(dev_preds, y_dev)
            dev_losses.append(dev_loss.item())

    return model, train_losses, dev_losses


def plot_loss_curves(train_losses, dev_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(dev_losses, label='Development Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Development Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.pdf')
    plt.show()

def model_weights(model):
    weights = model.linear.weight.data.squeeze()
    bias = model.linear.bias.data

    # Set print options to display full tensor without truncation
    torch.set_printoptions(threshold=100_000)

    print("Model Weights\n" )
    print(weights)
    print("\n\n")

    # print("Model weights:\n", weights)
    print("Model bias:\n", bias)
    print('-' * 80)
    print("\n\n\n")


def crossVal(model, train_set, dev_set, l2_penalty=10):
    """
    Evaluates a model on the training and development datasets.

    Parameters:
    - model: The model to be tested (an instance of torch.nn.Module).
    - train_set: A tuple containing the training features and labels.
    - dev_set: A tuple containing the development features and labels.

    Returns:
    - A dictionary containing the macro F1 score on the dev set.
    """
    X_train, y_train = train_set
    X_dev, y_dev = dev_set

    loss_function = nn.CrossEntropyLoss()
    epochs = 75
    lr = 0.1
    batch_size = 128
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_penalty)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])

        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            epoch_loss += loss.item() * batch_x.size(0)
        train_losses.append(epoch_loss / len(X_train))


    # Inside the evaluation block, adjust the prediction processing
    model.eval()
    with torch.no_grad():
        y_pred = model(X_dev)
        _, y_pred_labels = torch.max(y_pred, 1)  # Get the indices of the max log-probability
        f1 = f1_score(y_dev.numpy(), y_pred_labels.numpy(), average='macro')
    return f1


def improved_crossVal(model, train_set, dev_set, l2_penalty=10):
    """
    Evaluates a model on the training and development datasets.

    Parameters:
    - model: The model to be tested (an instance of torch.nn.Module).
    - train_set: A tuple containing the training features and labels.
    - dev_set: A tuple containing the development features and labels.

    Returns:
    - A dictionary containing the macro F1 score on the dev set.
    """
    X_train, y_train = train_set
    X_dev, y_dev = dev_set

    loss_function = nn.CrossEntropyLoss()
    epochs = 75
    lr = 0.1
    batch_size = 128
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=l2_penalty)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])

        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            epoch_loss += loss.item() * batch_x.size(0)
        train_losses.append(epoch_loss / len(X_train))


    # Inside the evaluation block, adjust the prediction processing
    model.eval()
    with torch.no_grad():
        y_pred = model(X_dev)
        _, y_pred_labels = torch.max(y_pred, 1)  # Get the indices of the max log-probability
        f1 = f1_score(y_dev.numpy(), y_pred_labels.numpy(), average='macro')
    return f1



def checkpoint_21(data_file, tokenizer_type):
    global vocabulary

    X, y = [], []

    Docs = [data_file[0], data_file[1], data_file[-1]]

    file = open("output.txt", "a")

    file.write("checkpoint 2.1\n\n")
    print("checkpoint 2.1\n\n")

    for index, test in enumerate(Docs):
        text_split = test.split('\t')
        label = text_split[2]
        text = remove_non_ascii(text_split[3]).rstrip()

        # Define the regular expression pattern for label
        label_pattern = r'(\w+)%(\d+:\d{2}:\d{2})+::'
        matches = re.findall(label_pattern, label)

        word_type, sense = matches[0][0], matches[0][1]
        vocab = vocabulary[word_type][tokenizer_type]
        words = text.split()
        target_index = get_target_index(words)

        prev_text, next_text = ' '.join(words[:target_index]), ' '.join(words[target_index + 1 : ])
        prev_token, next_token = [], []

        if tokenizer_type == "bpe":
            prev_token, next_token = spacelessBPETokenize(prev_text, vocab), spacelessBPETokenize(next_text, vocab)
        else:
            prev_token, next_token = wordTokenizer(prev_text), wordTokenizer(next_text)

        tokens = prev_token + [word_sense] + next_token
        target_index = len(prev_token)

        features = np.array(extractLexicalFeatures(tokens, target_index, tokenizer_type))

        np.set_printoptions(1000000)

        if index < 2:
            print(f"Document {index}:  \nFeatures\n ==> {features}")
            file.write(f"\nDocument {index}:  \nFeatures\n ==> {features}")
        else:
            print(f"Last Document: \nFeatures\n ==> {features}")
            file.write(f"\n\n\nLast Document: \nFeatures\n ==> {features}")

def checkpoint_22():
    global store_train_dev_set
    global Labels_Classes

    file = open("output.txt", "a+")
    file.write("Checkpoint 2.2\n\n\n\n")
    print("Checkpoint 2.2")

    X_trains, y_trains, X_devs, y_devs = store_train_dev_set["reduction"]["word"]
    train_corpus = [X_trains, y_trains, X_devs, y_devs, 3, 0.001, False]
    model, train_losses, dev_losses = trainLogReg(train_corpus)
    plot_loss_curves(train_losses, dev_losses)


    for tokenizer_type in ["bpe", "word"]:
        for sense, class_no in Labels_Classes:
            X_trains, y_trains, X_devs, y_devs = store_train_dev_set[sense][tokenizer_type]

            train_corpus = [X_trains, y_trains, X_devs, y_devs, class_no, 10, False]
            model, train_losses, dev_losses = trainLogReg(train_corpus)

            weights, bias = model.linear.weight.data.squeeze(), model.linear.bias.data

            # torch.set_printoptions(threshold=100_000)
            # file.write(f"weight({sense},{tokenizer_type}) : {weights}\n\n")
            # file.write(f"biases{sense},{tokenizer_type}) : {bias}\n\n")

            print(f"weight({sense},{tokenizer_type}) : {weights}\n\n")
            print(f"biases{sense},{tokenizer_type}) : {bias}\n\n")

def checkpoint_23():
    global store_train_dev_set
    global Labels_Classes

    file = open("output.txt", "a+")
    file.write("Checkpoint 2.3\n\n\n\n")
    print(("checkpoint 2.3"))


    for tokenizer_type in ["bpe", "word"]:
        f1_arr = []
        for sense, class_no in Labels_Classes:
            X_trains, y_trains, X_devs, y_devs = store_train_dev_set[sense][tokenizer_type]

            train_corpus = [X_trains, y_trains, X_devs, y_devs, class_no, 10, False]
            model, train_losses, dev_losses = trainLogReg(train_corpus)
            X_trainset, y_trainset = torch.from_numpy(train_corpus[0]), torch.from_numpy(train_corpus[1]).long()
            X_devset, y_devset = torch.from_numpy(train_corpus[2]), torch.from_numpy(train_corpus[3]).long()

            train_set, dev_set = (X_trainset, y_trainset), (X_devset, y_devset)
            model = MultiClassLogisticRegression(1503, class_no, 0.1)
            f1 = crossVal(model, train_set, dev_set)
            f1_arr.append(f1)
            print(f"Class: {sense} F1: {f1}")
            # file.write(f"Class: {sense} F1: {f1}")
        print("avg f1 for {} tokenization: {}".format(tokenizer_type, sum(f1_arr) / len (f1_arr)))
        # file.write("avg f1 for {} tokenization: {}\n\n\n".format(tokenizer_type, sum(f1_arr) / len (f1_arr)))



def checkpoint_24():
    print("checkpoint 2.4")
    global store_train_dev_set
    global Labels_Classes
    set_seed(65)

    l2_penalties = [0.001, 0.01, 0.1, 1, 10, 100]
    dropout_rates = [0, 0.1, 0.2, 0.5]
    tokenizer_types = ["bpe", "word"]
    store_tuples = []

    for tokenizer_type in tokenizer_types:
        result = {dropout: [] for dropout in dropout_rates}
        for dropout in dropout_rates:
            for l2p in l2_penalties:
                f1_arr = []
                for sense, class_no in Labels_Classes:
                    X_trains, y_trains, X_devs, y_devs = store_train_dev_set[sense][tokenizer_type]

                    train_corpus = [X_trains, y_trains, X_devs, y_devs, class_no, l2p, False] # last entry == False means no improvement is intended
                    model, train_losses, dev_losses = trainLogReg(train_corpus)

                    X_trainset = torch.from_numpy(train_corpus[0])
                    y_trainset = torch.from_numpy(train_corpus[1]).long()
                    X_devset = torch.from_numpy(train_corpus[2])
                    y_devset = torch.from_numpy(train_corpus[3]).long()

                    train_set, dev_set = (X_trainset, y_trainset), (X_devset, y_devset)

                    model = MultiClassLogisticRegression(1503, class_no, dropout)
                    f1 = crossVal(model, train_set, dev_set, l2p)
                    f1_arr.append(f1)
                    print(f"(penalty, dropout, tokenizer, word_type) : ({l2p}, {dropout}, {tokenizer_type}, {sense}) ==> f1 : {f1}")

                average = sum(f1_arr) / len (f1_arr)
                # print(f"(penalty, dropout, tokenizer) : ({l2p}, {dropout}, {tokenizer_type}) ==> avg f1 : {average}")
                result[dropout].append(average)
                store_tuples.append((dropout, l2p, tokenizer_type, average))
        
        print(f"Tokenizer: {tokenizer_type}")
        for dropout in result:
            print(result[dropout])

    sorted_store = sorted(store_tuples, key=lambda x: x[3], reverse=True)[0]
    print(f"Best hyperparameter: dropout: {sorted_store[0]}, penalty: {sorted_store[1]}, tokenizer_type: {sorted_store[2]}, best f1: {sorted_store}")




def checkpoint_25():
    print("checkpoint 2.5")
    global store_improved_train_dev_set
    global Labels_Classes

    set_seed(65)

    l2_penalties = [0.001, 0.01, 0.1, 1, 10, 100]
    dropout_rates = [0, 0.1, 0.2, 0.5]
    tokenizer_types = ["bpe", "word"]
    store_tuples = []

    for tokenizer_type in tokenizer_types:
        result = {dropout: [] for dropout in dropout_rates}
        for dropout in dropout_rates:
            for l2p in l2_penalties:
                f1_arr = []
                for sense, class_no in Labels_Classes:
                    X_trains, y_trains, X_devs, y_devs = store_improved_train_dev_set[sense][tokenizer_type]

                    train_corpus = [X_trains, y_trains, X_devs, y_devs, class_no, l2p, True] # last entry == False means no improvement is intended
                    model, train_losses, dev_losses = trainLogReg(train_corpus)

                    X_trainset = torch.from_numpy(train_corpus[0])
                    y_trainset = torch.from_numpy(train_corpus[1]).long()
                    X_devset = torch.from_numpy(train_corpus[2])
                    y_devset = torch.from_numpy(train_corpus[3]).long()

                    train_set, dev_set = (X_trainset, y_trainset), (X_devset, y_devset)

                    model = MultiClassLogisticRegression(1503, class_no, dropout)
                    f1 = improved_crossVal(model, train_set, dev_set, l2p)
                    f1_arr.append(f1)
                    print(f"(penalty, dropout, tokenizer, word_type) : ({l2p}, {dropout}, {tokenizer_type}, {sense}) ==> f1 : {f1}")

                average = sum(f1_arr) / len (f1_arr)
                # print(f"(penalty, dropout, tokenizer) : ({l2p}, {dropout}, {tokenizer_type}) ==> avg f1 : {average}")
                result[dropout].append(average)
                store_tuples.append((dropout, l2p, tokenizer_type, average))
        
        
        print(f"Tokenizer: {tokenizer_type}")
        for dropout in result:
            print(result[dropout])

        print('\n\n\n\n')

    sorted_store = sorted(store_tuples, key=lambda x: x[3], reverse=True)[0]
    print(f"Best hyperparameter: dropout: {sorted_store[0]}, penalty: {sorted_store[1]}, tokenizer_type: {sorted_store[2]}, best f1: {sorted_store}")



