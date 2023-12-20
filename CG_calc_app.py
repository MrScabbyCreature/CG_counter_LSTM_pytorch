import streamlit as st
import torch

class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self):
        super(CpGPredictor, self).__init__()
        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way
        self.lstm = nn.LSTM(5, LSTM_HIDDEN, LSTM_LAYER, batch_first=True)
        self.classifier = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        print(x.shape)
        lstm_out, _ = self.lstm(x.to(torch.float32))
        
        # Sum
        final_state = lstm_out.sum(axis=1)
        
        # Concat the sum of features
        count = self.classifier(final_state)
        return count


# Alphabet helpers   
alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}

# One hot encode the input
one_hot_dict = {i: [0, 0, 0, 0, 0] for i in range(6)}
for i in range(1, 6):
    one_hot_dict[i][i-1] = 1

g = lambda x: list(map(one_hot_dict.get, x))

class Counter:
    def __init__(self) -> None:
        self.lstm = torch.load("cg_calculator_lstm.pt")
        self.lstm.eval()
    
    def check_valid(self, string):
        if (set(list(string)) - set(list(alphabet))):
            return False
        return True

    def get_count(self, string):
        # Check if the string is valid
        if not self.check_valid(string):
            return "Invalid"

        # Encode string
        encoding = [self.encode(string)]

        # Predict
        return self.lstm(encoding)[0][0].item()


    def encode(self, string):
        l = []
        for s in string:
            l.append(one_hot_dict[dna2int[s]])
        return l

    def __call__(self, string):
        return self.get_count(string)


counter = Counter()
# test_string = "NACGTTTGCGNAAANNNCGGG"
# test_string = "asd"
# print(counter(test_string))

#############
# STREAMLIT #
#############
st.title("CG counter - LSTM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
prompt = st.chat_input("What is up?")
if prompt:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"LSTM: {counter(prompt)}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})