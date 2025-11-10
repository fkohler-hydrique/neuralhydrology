# Short description of some RNN models

## Recurrent Neural Network = RNN

Designed to process sequential data, using recurrent connections

- Allow to retain info from previous input/ouput
- Major diff with FF NN (Feed-Forward) is this two way direction and the link between two inputs, that would be taken independently with FC

### Problems

  Gradient: with long TS, exploding or vanishing gradient happens -> the network cannot learn anymore

### Composition

1. Input layer
2. Recurrent hidden layer
3. Output layer

![alt text](image.png)

## Long Short Term Memory = LSTM

### Description

Variante de RNN modèle (Recurrent Neural Network), permettant de garder des informations importantes sur de longues périodes, tout en évitant des problèmes typiques des RNN (vanishing/exploding gradient en utilisant des longues séries temporelles comme entrée, long TS=TimeSeries as input). La notion de mémoire est modélisée avec 3 portes (forget, input et output) et une cellule mémoire.

- Memory cells: LSTMs have a "cell state" which acts as a long-term memory conveyor belt, allowing information to be passed along unchanged.
- Gates: These are neural network layers that regulate the flow of information into and out of the cell state:
  - Forget gate: Decides what information from the previous cell state should be thrown away.
  - Input gate: Decides what new information should be stored in the cell state.
  - Output gate: Decides what part of the cell state should be outputted as the hidden state (short-term memory).

### typical uses

- Natural Language Processing: Machine translation, text generation, sentiment analysis, and speech recognition.
- Time Series Analysis: Stock market prediction, weather forecasting, and signal processing.
- Other tasks: Handwriting recognition, image captioning, and video analysis.

### Variantes de LSTM

- Peepholes Connections: Connexion direct entre l'état de la cellule mémoire et les gates, tient mieux en compte des dépendences à long terme. (askip bien quand besoin de timing précis entre longue période, sounds interesting)
- GRU = Gated Recurrent Unit: simplified LSTM, input and forget gates are fused together. The separated memory cell is eliminated too, reducing computational complexity.

### Workflow

1. Input processing:
   1. Each cell recieve input $X_t$ at a specific time step and the **previous** hidden state $h_{t-1}$ and cell state $C_{t-1}$.
2. Gates:
   1. Forget
    which part of $C_{t-1}$ should be discarded. Sigmoid fct to produce $v\in(0,1)$, $0$ to forget, $1$ being to remember the info entirely.
    $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
   2. Input
    Update $C_t$ with two part:
       1. Candidate state: tanh layer that generate potential updates $\tilde{C}_t$
       2. Update decision: Sigmoid layer that decides the importance of each candidate state
        $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
    C̃_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$
   3. Cell state update
        The two previous gates are used to update the cell state $C_t$:
        $$ C_t = f_t \ast C_{t-1} + i_t \ast C̃_t $$
   5. Output
        Decide which parts of the cell state should have an impact on the hidden state $h_t$. Sigmoid fct  to select parts, tanh transformation to produce the current hidden state.
        $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
            h_t = o_t \ast \tanh(C_t) $$

### training and optimizations

Training through BPTT (Back-Propagation Through Time) = adjust weights to minimize the loss between measure and prediction.

- Potential problems:
  - Exploding gradient - clip the gradient
  - Vanishing gradient - infrastructure of the LSTM
  - Overfitting - Early stopping on validation loss

## Auto-Regressive LSTM

<https://peerj.com/articles/cs-2046/#fig-4>

LSTM that predicts a sequence using its own previous prediction as the next input

autoregressive moving average (ARMA)

### Sequence

1. **Sequential prediction** prediction for the next time step using current features
2. **Feedback loop** the prediction is fed back into the LSTM as new input to help it to predict the $t+2$ value
3. **Generation** Repeatition of this process -> generation of a sequence of wanted length
4. **Lag parameter** Defines how many previous time steps are used as input to predict next value.
5. **S** p

---

## Sources

<https://www.sciencedirect.com/topics/computer-science/recurrent-neural-network>
<https://www.sciencedirect.com/topics/computer-science/long-short-term-memory-network>
<https://www.appliedaicourse.com/blog/lstm-in-machine-learning/>
<https://colah.github.io/posts/2015-08-Understanding-LSTMs/>

## Time-Series forecasting

Normalisation of the target:

```python
# Extract the values of the target column
data = df['Price'].values
data = data.reshape(-1, 1)
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```

<https://machinelearningmastery.com/mastering-time-series-forecasting-from-arima-to-lstm/>
