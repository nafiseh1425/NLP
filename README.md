# DistilBERT-based Movie Review Classification
This project demonstrates the use of a pre-trained DistilBERT model for sentiment classification on movie reviews. The workflow involves data preprocessing, feature extraction using DistilBERT, and training a logistic regression model to classify the sentiments as positive or negative.
#Introduction
This project leverages the power of the DistilBERT model, a distilled version of BERT, which provides similar accuracy while being more efficient. The goal is to classify movie reviews into positive or negative sentiments using a logistic regression model trained on features extracted by DistilBERT.

#Dataset
We use the SST-2 dataset, a subset of the Stanford Sentiment Treebank. This dataset consists of movie reviews labeled as either positive or negative.

The dataset is loaded directly from the following URL:
('''df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None''')

#Model
We utilize the pre-trained DistilBERT model for extracting features from the text. The tokenizer and model are loaded as follows:
('''import transformers as ppb  # pytorch transformers

# Define the model and tokenizer
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)''')

For those who prefer BERT over DistilBERT, simply uncomment the following line:
('''# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
''')

#Tokenization and Padding
The sentences are tokenized, and padding is applied to ensure uniform input size for the model.
('''tokenized = df[0].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Padding
max_len = max([len(i) for i in tokenized.values])
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
''')
#Attention Mask
An attention mask is generated to differentiate between padding and non-padding tokens.
('''attention_mask = np.where(padded != 0, 1, 0)''')

#Feature Extraction
The model processes the input sentences, and we extract the features from the last hidden states.

('''input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:, 0, :].numpy()
''')
#Training
The extracted features are used to train a logistic regression model.
('''from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

labels = df[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
''')

#Model Evaluation
The model's performance is evaluated on the test set:
('''from sklearn.model_selection import GridSearchCV

parameters = {'C': np.linspace(0.0001, 100, 20)}
grid_search = GridSearchCV(LogisticRegression(), parameters)
grid_search.fit(train_features, train_labels)

print('Best parameters: ', grid_search.best_params_)
print('Best score: ', grid_search.best_score_)
''')
#Hyperparameter Tuning
We perform grid search to find the best hyperparameters for the logistic regression model:
('''from sklearn.model_selection import GridSearchCV

parameters = {'C': np.linspace(0.0001, 100, 20)}
grid_search = GridSearchCV(LogisticRegression(), parameters)
grid_search.fit(train_features, train_labels)

print('Best parameters: ', grid_search.best_params_)
print('Best score: ', grid_search.best_score_)
''')
#Results
The logistic regression model, trained on DistilBERT features, achieves a solid accuracy on the sentiment classification task. The best hyperparameters for the model were found using grid search, improving its performance.
