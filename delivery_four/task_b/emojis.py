import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128) 
        self.dense = nn.Linear(128, encoding_size)  

    def reset(self): 
        zero_state = torch.zeros(1, 1, 128)  
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def f(self, x): 
        return torch.softmax(self.logits(x), dim=1)

    def logits(self, x): 
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(self.hidden_state[-1])

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.unsqueeze(0))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a'
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c'
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'f'
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'm'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'n'
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'p'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 's'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  # 't'
]

encoding_size = len(char_encodings)

index_to_char = [' ', 'a', 'c', 'f', 'h', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't']

x_train = torch.tensor([
    [char_encodings[0] ,char_encodings[4], char_encodings[1], char_encodings[12], char_encodings[0]],  # ' hat '
    [char_encodings[0] ,char_encodings[10], char_encodings[1], char_encodings[12], char_encodings[0]],  # ' rat '
    [char_encodings[0], char_encodings[2], char_encodings[1], char_encodings[12], char_encodings[0]],  # ' cat '
    [char_encodings[3], char_encodings[5], char_encodings[1], char_encodings[12], char_encodings[0]],  # 'flat '
    [char_encodings[0], char_encodings[6], char_encodings[1], char_encodings[12], char_encodings[12]],  # ' matt'
    [char_encodings[0], char_encodings[2], char_encodings[1], char_encodings[9], char_encodings[0]],  # 'cap '
    [char_encodings[0], char_encodings[11], char_encodings[7], char_encodings[8], char_encodings[0]]  # 'son '
])

y_train = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)

index_to_word = ["hat ", "rat ", "cat ", "flat", "matt", "cap ", "son "]

model = LongShortTermMemoryModel(encoding_size)

test_words = ['ht ', 'rt ', 'rats', 'cp', 'snn', 'mat', 'so', 'ca', 'cta ', 'catt', 'flot']


optimizer = torch.optim.Adam(model.parameters(), 0.001)

for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        x_input = x_train[i].unsqueeze(1)
        loss = model.loss(x_input, y_train[i])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Function to convert a single word to its char encoding
def word_to_encoding(word):
    return [char_encodings[index_to_char.index(c)] for c in word]

emojis = [
    "🎩",
    "🐀",
    "🐈",
    "🏠",
    "👨",
    "🧢",
    "👶"
    ]

test_data = [torch.tensor(word_to_encoding(word)).unsqueeze(1) for word in test_words]

for i, word in enumerate(test_data):
    model.reset() 
    with torch.no_grad(): 
        logits = model.logits(word)
        predictions = torch.argmax(logits, dim=1)
        
    predicted_class = predictions.item()
    predicted_word = index_to_word[predicted_class]
    predicted_emoji = emojis[predicted_class]
    print(f"Test word: {test_words[i]}, Predicted emoji: {predicted_emoji}, Predicted word: {predicted_word}")
