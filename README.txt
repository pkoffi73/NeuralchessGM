NeuralchessGM, a Neural network chess engine trained on Grand Masters games

Overall Description

NeuralchessGM is a chess engine based on a fully supervised neural network and on a Monte-Carlo tree search algorithm


Performance

This engine is still under testing but it has already been proved successfull against 2200 elo chess engines.


Dataset

270000 Grand Master (2500+ elo) games (21 millions board positions/moves) downloaded on Chess-DB.com


Board representation

A bitmap representation (8 x 8) with additional features for better performance:
- features 0 to 11: 1 panel for each kind of piece (black and white)
- feature 12 (board turn): 1 for white, 0 for black
- feature 13 (white castling rights): left side of the panel is 1 if castling rights on queen side; right side of the panel is 1 if castling rights on king side
- feature 14 (black castling rights): same as white
- feature 15 (white board center superiority): 1 on c4, d4, e4, c5, d5, e5 if more white pieces are in the board center
- feature 16 (black board center superiority): 1 on c4, d4, e4, c5, d5, e5 if more black pieces are in the board center
- feature 17 (check): 1 if there is a check (black or white)
- feature 18 (white pin): if there is a pin for white, the related squares are marked with 1
- feature 19 (black pin): same as white
- feature 20 (white attack): squares attacked by white are marked with 1
- feature 21 (black attack): same as black
Neural network input is therefore of shape 8 x 8 x 22


NeuralNetwork

A 2 headed neural network similar to Alphazero:
- a common core made of 10 blocks of resnets: 128 filters and kernel of size 3 x 3 - ReLU for activation
- a value head made of 1 convolutional layer (1 filter and kernel of size 1 x 1) and 1 fully connected layer (size 256) and 1 output layer (size 1) - ReLU for activation and Tanh for output
- a policy head made of 1 convolutional layer (2 filters and kernel of size 1 x 1) and 1 output layer (size 1917 - number of potential moves)
Batchnormalization is used on every layer to enhance training


Training

MSE Loss function for value head and categorical crossentropy for policy head. Total loss is 0.5 x value loss + 1.0 x policy loss (to prevent overfitting on value head)
Optimizer based on SGD with momentum (0.9) and with the following learning rate schedule
- 10 epochs at 0.1
- 10 epochs at 0.01
- 5 epochs at 0.001
- 5 epochs at 0.0001
It took 3 days on a GTX1070 to achieve a 50% value accuracy (0.4 value loss) and a 50% policy accuracy (1.5 policy loss)


Monte-Carlo tree search algorithm

Classical MTCS algorithm:
- Node descent and selection. It selects child node with maximum (Q + U). Q = V / (1 + N_visites) where V is the accumulated value ; U is proportional to P / (1 + N_visites) where P is the prior (based on policy)
- When a node is not expanded, node expansion including computation of its value and of its children probabilities (neural network)
- At the end of the process, node with highest visit numbers is selected
MTCS process is stopped in 4 cases:
- immediate stopping when there is a single legal move
- early stopping at half of basic time if the current best node is the node with highest Q, if best node's Q is not less than -0.05 compared to previous Q and if:
	- either the prior of the current best node is higher than 0.9
	- or the current best node has been visited 500 more times than the second best node
- normal stopping at basic time if best node's Q is not less than -0.05 compared to previous move Q or if the current best node has been visited 1000 more times than the second best node
- late stopping at twice the basic time if the previous conditions have not been met
The chess engine has been successfully tested with a basic time of 120 seconds (5000 iterations on the test laptop)


API

Chess board API is based on the python chess library
HTML template is based on the template developed by George Hotz for Twichchess


How to Play

Run the python 3 script: play_policy_value.py
When requested go to http://127.0.0.1:5000/ on your web browser
if you want to play with whites, just move one board piece
if you want to play with blacks, click on "white computer" button and wait for the computer to compute the first move. Then, play
You can remove the last moves by clicking on the button at the bottom.

Package

- state.py: manages the chess board state (including bitmap representation)
- dataset_to_dict_policy.py: goes through the GM games to build a lookup table between vector component (output of the policy head) and an actual chess move
- dataset_GM_value_policy.py: builds the training dataset (X = inputs (8 x 8 x 22); Y1 = value output (1, 0 or -1); Y2 = policy output (one hot 1917 vector)) ; it breaks down the 21 million positions into 21 dataset files (1 million each))
- train_model.py: trains the network
- play_policy_value.py: plays chess
- dict_policy.pkl: lookup table (see above)
- index.html: HTML templat
- dataGM: directory with GM games rawdataset (not provided here for large file issues but you can download GMallboth.pgn on chess-DB.com)
- processedGM6: directory with 21 files of processed data (not provided here for large file issues but you can build them with dataset_GM_value_policy.py)
- model: saved NN model
- static: JS files


Necessary Python 3 libraries

- chess
- tensorflow 2.0
- pickle
- flask

