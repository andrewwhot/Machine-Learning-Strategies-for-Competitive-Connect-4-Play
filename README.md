# Team Members
| Neha Boinapalli | Vishwa Patel | Andrew White | Sonali Hornick |
| - | - | - | - |

# Introduction
The objective of this project was to develop a machine learning model capable of playing Connect 4 at a high level. To achieve this, we trained and evaluated two different neural network architectures: a Convolutional Neural Network (CNN) and a Transformer-based model. These models were trained using a dataset generated by Monte Carlo Tree Search (MCTS), an algorithm that provides near-optimal move recommendations.

Our approach involved encoding the Connect 4 board state as a 6x7x2 representation, which allowed both models to process the game as a structured data input. The CNN was designed to leverage local spatial patterns in the board, while the Transformer was built to recognize broader, more abstract relationships between game pieces.

Additionally, we explored a hybrid Transformer model, incorporating convolutional layers to improve local feature extraction, though this approach was ultimately set aside to refine the core models.

Included in detail are descriptions of the architecture, training process, challenges encountered, and performance evaluation of both models. We aimed to determine which model is better suited for the Connect 4 decision-making task and provide insights into the strengths and weaknesses of each approach.

# Dataset Generation and Preprocessing
There are roughly 4.3 trillion different board positions possible in a traditional 6x7 Connect 4 board. Our goal for this project was to build a sufficiently diverse dataset of board positions, and their optimal recommended moves, which could then be used to train our deep learning models. To do this, we used a Monte Carlo Tree Search (MCTS) algorithm with a high number of “steps” in the search. The MCTS algorithm outputs an optimal recommended next move for any given board configuration by simulating many random playouts from the current state, gradually refining move probabilities based on observed win rates.

After generating the dataset, we completed a few preprocessing steps to support the ability of our models to learn and predict from the data:
- __Board Mirroring:__ By mirroring the boards (and recommended moves) along the 
  horizontal axis, we effectively doubled the number of boards in our dataset and 
  increased diversity. 
- __Duplicate Removal:__ To avoid confusing the model, we removed duplicate boards and 
  kept only the most commonly recommended move for every unique board.
- __Changing Input Shape to 6x7x2:__ This format made it easier for both the CNN and 
  the transformer to learn from the data (as opposed to a 6x7 input shape, which is 
  more intuitive when viewing the board).

# Convolutional Neural Network (CNN) Model
## Architecture
To train a model that can predict the best move in Connect 4, we implemented a Convolutional Neural Network (CNN). The CNN treats the board as a 6x7 image with two channels (i.e. a 6x7x2 input shape), where each channel represents the presence of a specific player's pieces. The network consists of several convolutional layers to extract spatial features, followed by dense layers to map these features to the best move.
- __Input Shape:__ (6, 7, 2)
- __Convolutional Layers:__
  - Conv2D (128 filters, 3x3, ReLU, padding='same') + Dropout (0.2)
  - Conv2D (256 filters, 3x3, ReLU, padding='same') + Dropout (0.3)
  - Conv2D (512 filters, 3x3, ReLU, padding='same') + Dropout (0.4)
  - Conv2D (1024 filters, 3x3, ReLU, padding='same') + Dropout (0.5)
- __Pooling Layers:__ None, instead using Global Average Pooling
- __Dense Layers:__
  - Fully connected layer with 2048 neurons, ReLU activation, L2 regularization (0.01)
- __Output Layer:__ 7 neurons representing the probability distribution over the 7 
  columns with a softmax activation function
## Training Process
To train the CNN, we used a dataset generated by Monte Carlo Tree Search (MCTS). The dataset was split into training and validation sets to assess model performance. The training process involved:
- __Batch Size:__ 64
- __Number of Epochs:__ 50 (though early stopping halted training after 35 epochs)
- __Loss Function:__ Sparse Categorical Cross-Entropy
- __Optimizer:__ Adam with an initial learning rate of 0.0006
- __Learning Rate Adjustment:__ ReduceLROnPlateau (factor=0.1, patience=4) to improve stability
- __Early Stopping:__ Patience of 5 epochs, restoring the best model weights
## Challenges and Improvements
During training, we encountered overfitting and difficulty in generalizing across diverse board states. To address these issues, we made several improvements:
- __Dropout Layers:__ Added progressively higher dropout rates (0.2 to 0.5) after each 
  convolutional layer
- __L2 Regularization:__ Applied to the dense layer to reduce overfitting
- __Global Average Pooling:__ Instead of flattening, we used Global Average Pooling to 
  reduce overfitting and make the model more robust
- __Learning Rate Adjustment:__ Went from an initial learning rate of 0.001 to 0.0006, 
  improving accuracy, efficiency and performance
  - Implemented a learning rate scheduler as well that dropped by a factor of 0.1 with 
    a patience of 4
 
One of the main challenges was that the majority of our earlier CNNs had difficulty getting over a validation accuracy of 50%. To mitigate this, learning rate scheduling and a lower learning rate were introduced. Additionally, it was found that our current model performed well at over ~8 million parameters. Overfitting also was a concern in the training process, which was addressed by setting increasing dropout layers to balance training efficiency and generalization with the greater number of convolutional layers. Ultimately, our CNN reached an accuracy of ~68.75% after all the improvements.

# Transformer Model
## Initial Implementation
Similar to the CNN, our transformer model used the (6,7,2) board representation, ensuring that the model could effectively process Connect 4 board states. The goal was to leverage self-attention mechanisms to capture strategic relationships between board positions, which traditional convolutional networks might struggle with.
## Architecture
Unlike the CNN, which focuses on local spatial relationships, the Transformer model evaluates global dependencies across the entire board. This is achieved through multi-head self-attention, which allows the model to weigh different board positions differently when making move predictions.
- __Input Encoding:__ The (6,7,2) board state is flattened into a sequence of 42 
  tokens, with each token having two channels to represent player pieces.
- __Positional Embeddings:__ Since Transformers lack an inherent sense of order, we 
  added trainable positional embeddings to preserve board structure.
- __Multi-Head Self-Attention:__ The model uses 8 attention heads, allowing it to 
  process different positional relationships in parallel.
- __Feedforward Layers:__ Each attention output is passed through:
  - A fully connected layer with ReLU activation for feature extraction.
  - A second fully connected layer to map back to the original embedding dimension.
- __Layer Normalization & Dropout:__ These regularization techniques prevent 
  overfitting and improve generalization.
- __Output Layer:__ The model predicts move probabilities over the 7 columns using a 
  softmax activation function.
## Training Process
The Transformer was trained using the MCTS-generated dataset with augmented board states to improve robustness.
- __Batch Size:__ 64 (we also experimented with 128)
- __Epochs:__ 30 (though early stopping halted training after 25 epochs)
- __Loss Function:__ Sparse Categorical Cross-Entropy
- __Optimizer:__ Adam with an initial learning rate of 1e-4
- __Learning Rate Adjustment:__ ReduceLROnPlateau (factor=0.5, patience=2) to improve 
  stability
- __Early Stopping:__ Patience of 5 epochs, restoring the best model weights
## Challenges and Improvements
One of the main challenges was slow convergence, as the Transformer trained at a significantly slower rate than the CNN. To mitigate this, a larger batch size and learning rate scheduling were introduced. Overfitting also became a concern later in the training process, which was addressed by setting a dropout rate of 0.05 to balance training efficiency and generalization.

Encoding adjustments played a crucial role in improving performance. Flattening the board into a 42-token sequence allowed the Transformer to process the game state in a format more suited to attention mechanisms. Additionally, increasing the number of attention heads to eight enhanced the model’s ability to capture positional relationships without immediate overfitting, though it became a challenge in later training stages.

While the model's validation accuracy (62.48%) was lower than that of the CNN (~68.75%), it demonstrated stronger strategic move selection, particularly in scenarios requiring long-term planning. This suggests that while the CNN excelled in recognizing local patterns, the Transformer was more effective in evaluating broader board dynamics, making it a valuable alternative for complex decision-making tasks.

# Model Evaluation and Comparison
## Performance Metrics
Both models were evaluated using multiple criteria. Although the accuracy didn’t climb above 70% for either model, we found that the models still play relatively well when tested against the MCTS algorithm or a strong human player. 

The validation accuracies were 68.75% and 62.48% for the CNN and transformer respectively. 

Below are the metrics for the transformer’s performance against the MCTS algorithm. These metrics were evaluated without a look_for_win function to see how well the models performed.
| Opponent | CNN | Transformer |
| -------- | --- | ----------- |
| MCTS with 1000 steps | Won 6/10 games | Won 7/10 games |
| MCTS with 2500 steps | Won 6/10 games | Won 7/10 games |
| MCTS with 5000 steps | Won 4/10 games | Won 4/10 games |

# Conclusion
Through rigorous experimentation, we found that both the CNN and Transformer models demonstrated strong performance in predicting optimal Connect 4 moves, though with distinct advantages and trade-offs. The CNN model achieved higher validation accuracy (68.75%), likely due to its ability to capture local spatial patterns effectively. Meanwhile, the Transformer model, while slightly less accurate (62.48%), showed promise in capturing long-range dependencies and strategic board positioning.

Challenges such as overfitting, slow convergence, and hyperparameter optimization were addressed through various techniques, including dropout regularization, learning rate adjustments, and architectural modifications. The CNN benefited from global average pooling and L2 regularization, whereas the Transformer required positional encoding optimizations and additional attention heads to improve its decision-making.

While the CNN model ultimately outperformed the Transformer in terms of accuracy, the Transformer’s ability to generalize across different board configurations makes it a compelling alternative. Future work could involve combining both architectures into a hybrid model that leverages the CNN’s feature extraction capabilities with the Transformer’s sequential reasoning strengths. Additionally, further training against more sophisticated MCTS opponents could enhance the robustness of both models.

This project provided valuable insights into deep learning approaches for strategic game-playing AI, demonstrating the practical applications of CNNs and Transformers beyond traditional image and text processing. As advancements in AI continue, integrating these architectures could pave the way for even more intelligent game-playing agents.



