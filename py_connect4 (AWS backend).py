import anvil.server

anvil.server.connect("server_QCSXEGVA7DV46ESQKRIKVPCX-ZY5VB57EKZJK2GUW")


import numpy as np
import tensorflow as tf

# For headless matplotlib (no GUI):
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

############################################
# 0) GLOBAL STATE
############################################

GAME_BOARD = None          # 67 np.array
GAME_MODE = None           # "user_vs_user", "user_vs_transformer", or "user_vs_cnn"
CURRENT_PLAYER = None      # used only for user_vs_user ("plus" or "minus")

cnn_model = None
transformer_model = None

# NEW: Track whether the user goes first (True) or second (False)
USER_IS_FIRST = True

############################################
# 1) LOADING MODELS
############################################

def load_models():
    """
    Loads the CNN and Transformer from disk. 
    Adjust file paths or add custom objects if needed.
    """
    global cnn_model, transformer_model
    
    # Example custom layers/classes for Transformer:
    class GetItem(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs[:, 0, :]

    class PositionalIndex(tf.keras.layers.Layer):
        def call(self, x):
            bs = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            indices = tf.range(seq_len)
            indices = tf.expand_dims(indices, 0)
            return tf.tile(indices, [bs, 1])

    class ClassTokenIndex(tf.keras.layers.Layer):
        def call(self, x):
            bs = tf.shape(x)[0]
            indices = tf.range(1)
            indices = tf.expand_dims(indices, 0)
            return tf.tile(indices, [bs, 1])

    class ExtractClassToken(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs[:, 0, :]

    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, maxlen, d_model, **kwargs):
            super(PositionalEmbedding, self).__init__(**kwargs)
            self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=d_model)

        def call(self, x):
            seq_len = tf.shape(x)[1]
            positions = tf.range(start=0, limit=seq_len, delta=1)
            pos_embeddings = self.pos_emb(positions)  # shape (seq_len, d_model)
            return x + pos_embeddings

    class TransformerEncoder(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, ff_dim, rate=0.1, **kwargs):
            super().__init__(**kwargs)
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation='relu'),
                tf.keras.layers.Dense(d_model),
            ])
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, inputs, training=False, mask=None):
            attn_output = self.att(inputs, inputs, inputs, attention_mask=mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    custom_objects = {
        "GetItem": GetItem,
        "PositionalIndex": PositionalIndex,
        "ClassTokenIndex": ClassTokenIndex,
        "ExtractClassToken": ExtractClassToken,
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder
    }

    print("Loading CNN model...")
    cnn_path = "./cnn_model.h5"  # Update with your actual path
    cnn_model = tf.keras.models.load_model(cnn_path, custom_objects=custom_objects)
    print("CNN model loaded.")

    print("Loading Transformer model...")
    transformer_path = "./transformer_model.keras"  # Update with your actual path
    transformer_model = tf.keras.models.load_model(transformer_path, custom_objects=custom_objects)
    print("Transformer model loaded.")


############################################
# 2) HELPER FUNCTIONS
############################################

def init_board():
    """Create a fresh 67 board of zeros."""
    return np.zeros((6, 7), dtype=np.float32)

def find_legal_moves(board):
    """
    Returns a list of columns that are not full (top cell is empty).
    """
    legal = [i for i in range(7) if abs(board[0, i]) < 0.1]
    return legal

def update_board(board_temp, color, column):
    """
    Places a checker in 'column' for color: 'plus' => +1, 'minus' => -1.
    If the column is full, does nothing. (We assume front-end blocks that.)
    """
    board = board_temp.copy()
    colsum = (abs(board[0,column]) + abs(board[1,column]) + abs(board[2,column]) +
              abs(board[3,column]) + abs(board[4,column]) + abs(board[5,column]))
    row = int(5 - colsum)
    if row > -0.5:
        if color == 'plus':
            board[row, column] = 1
        else:
            board[row, column] = -1
    return board

def check_for_win(board, col):
    """Checks if the last move in 'col' caused 4 in a row."""
    nrow, ncol = 6, 7
    
    colsum = sum(abs(board[r, col]) for r in range(nrow))
    row = 6 - int(colsum)  
    if row >= 6 or row < 0:
        return 'nobody'

    val = board[row, col]
    if abs(val) < 0.5:
        return 'nobody'

    def check_four(r_list, c_list):
        return sum(board[r_list[i], c_list[i]] for i in range(4)) == 4 * val

    # Vertical
    if row <= 2 and check_four([row + i for i in range(4)], [col] * 4):
        return 'v-plus' if val > 0 else 'v-minus'

    # Horizontal
    for start_col in range(max(0, col - 3), min(col + 1, 4)):
        if check_four([row] * 4, [start_col + i for i in range(4)]):
            return 'h-plus' if val > 0 else 'h-minus'

    # Down-right diagonal
    for offset in range(-3, 1):
        if 0 <= row + offset < 3 and 0 <= col + offset < 4:
            if check_four([row + offset + i for i in range(4)],
                          [col + offset + i for i in range(4)]):
                return 'd-plus' if val > 0 else 'd-minus'

    # Down-left diagonal
    for offset in range(-3, 1):
        if 0 <= row + offset < 3 and 3 <= col - offset < 7:
            if check_four([row + offset + i for i in range(4)],
                          [col - offset - i for i in range(4)]):
                return 'd-plus' if val > 0 else 'd-minus'

    return 'nobody'

def board_6x7_to_6x7x2(board_6x7):
    """ Convert a 67 board to (6,7,2) with channels for +1 / -1. """
    plus_channel  = (board_6x7 == 1).astype(np.float32)
    minus_channel = (board_6x7 == -1).astype(np.float32)
    return np.stack([plus_channel, minus_channel], axis=-1)

def predict_plus_move(model, board_6x7, color):
    """
    If the model is trained for 'plus', 
    multiply board by -1 if 'minus' is moving (so minus sees itself as plus).
    Then pick argmax of model output (7 columns).
    """
    if color == 'minus':
        board_for_model = -board_6x7
    else:
        board_for_model = board_6x7

    input_2ch = board_6x7_to_6x7x2(board_for_model)
    input_2ch = np.expand_dims(input_2ch, axis=0)  # => shape (1,6,7,2)
    preds = model.predict(input_2ch, verbose=0)[0]  # => shape (7,)
    chosen_col = int(np.argmax(preds))
    return chosen_col

def look_for_win(board_): #new function
    """
    Check if there is an immediate winning move for the AI.
    If found, return the column number; otherwise, return -1.
    """
    board_ = board_.copy()
    legal = find_legal_moves(board_)

    for m in legal:
        temp_board = update_board(board_.copy(), CURRENT_PLAYER, m)
        if check_for_win(temp_board, m)[2:] == CURRENT_PLAYER:
            return m  # Return winning move
    return -1  # No immediate win found

def draw_board_image(board):
    """
    Create a matplotlib figure for the given 67 board.
    We'll color:
      - empty: light gray
      - plus:  red (user1)
      - minus: yellow (AI or user2)
    Return an anvil.BlobMedia('image/png', ...) object.
    """
    import anvil
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.invert_yaxis()

    ax.set_facecolor('#000000')

    for r in range(6):
        for c in range(7):
            val = board[r, c]
            if abs(val) < 0.1:
                color = '#95C21A' #fill-in circle color
            elif val > 0:
                color = '#F95C32' #red
            else:
                color = '#4169E1' #blue
            circle = plt.Circle((c, 5 - r), 0.4, color=color)
            ax.add_patch(circle)

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches = 0, transparent=True)
    buf.seek(0)
    plt.close(fig)

    #  The only strictly necessary fix: Use BlobMedia instead of Media
    return anvil.BlobMedia('image/png', buf.read(), name='board.png')

############################################
# 3) GAME MODES & SERVER CALLABLES
############################################

@anvil.server.callable
def set_game_mode(mode):
    """
    Initialize a new game in one of these modes (possibly with "_second" suffix):
      - "user_vs_user"
      - "user_vs_transformer"
      - "user_vs_cnn"
      - or "user_vs_transformer_second", etc.

    If the mode ends with "_second", user goes second (AI is 'plus').
    Otherwise, user goes first (user is 'plus').

    Returns {"status": "new_game", "board_image": <PNG image>}.
    """
    global GAME_BOARD, GAME_MODE, CURRENT_PLAYER, USER_IS_FIRST

    if mode.endswith("_second"):
        USER_IS_FIRST = False
        mode = mode.replace("_second", "")
    else:
        USER_IS_FIRST = True

    GAME_BOARD = init_board()
    GAME_MODE = mode

    if mode == "user_vs_user":
        CURRENT_PLAYER = "plus"
    else:
        CURRENT_PLAYER = "plus"

    # If the user is second, make the AI's first move
    if not USER_IS_FIRST:
        if GAME_MODE == "user_vs_transformer":
            col_ai = predict_plus_move(transformer_model, GAME_BOARD, 'plus')
        else:
            col_ai = predict_plus_move(cnn_model, GAME_BOARD, 'plus')

        GAME_BOARD = update_board(GAME_BOARD, 'plus', col_ai) #end of new code


    board_image = draw_board_image(GAME_BOARD)
    return {"status": "new_game", "board_image": board_image}


@anvil.server.callable
def play_turn(col):
    """
    Called when a user clicks a column in Anvil.
    For user-vs-user: players alternate as plus/minus.
    For user-vs-AI: 
      - If USER_IS_FIRST is True, user = plus, AI = minus.
      - If USER_IS_FIRST is False, AI = plus, user = minus.

    Returns {"status": ..., "board_image": <PNG image>}.

    After a win or tie, the board is reset automatically.
    """
    global GAME_BOARD, GAME_MODE, CURRENT_PLAYER, USER_IS_FIRST
    if GAME_MODE is None:
        return {"status": "no_mode_chosen", "board_image": draw_board_image(GAME_BOARD)}

    legal = find_legal_moves(GAME_BOARD)
    if col not in legal:
        return {"status": "illegal_move", "board_image": draw_board_image(GAME_BOARD)}

    if GAME_MODE == "user_vs_user":
        GAME_BOARD = update_board(GAME_BOARD, CURRENT_PLAYER, col)
        result = check_for_win(GAME_BOARD, col)
        if result[2:] == CURRENT_PLAYER:
            if CURRENT_PLAYER == 'plus':
                status = "player1_win"
            else:
                status = "player2_win"
            board_image = draw_board_image(GAME_BOARD)
            return {"status": status, "board_image": board_image}

        if len(find_legal_moves(GAME_BOARD)) == 0:
            status = "tie"
            board_image = draw_board_image(GAME_BOARD)
            return {"status": status, "board_image": board_image}

        CURRENT_PLAYER = "minus" if CURRENT_PLAYER == "plus" else "plus"
        board_image = draw_board_image(GAME_BOARD)
        return {"status": "continue", "board_image": board_image}

    else:
        if USER_IS_FIRST:
            GAME_BOARD = update_board(GAME_BOARD, 'plus', col)
            result_user = check_for_win(GAME_BOARD, col)
            if result_user[2:] == 'plus':
                status = "user_win"
                board_image = draw_board_image(GAME_BOARD)
                return {"status": status, "board_image": board_image}

            if len(find_legal_moves(GAME_BOARD)) == 0:
                status = "tie"
                board_image = draw_board_image(GAME_BOARD)
                return {"status": status, "board_image": board_image}

            if GAME_MODE == "user_vs_transformer":
                winning_move = look_for_win(GAME_BOARD) #added this function
                if winning_move != -1:
                    col_ai = winning_move  # AI takes the winning move
                else:
                    col_ai = predict_plus_move(transformer_model, GAME_BOARD, 'minus')
            else:
                winning_move = look_for_win(GAME_BOARD)
                if winning_move != -1:
                    col_ai = winning_move  # AI takes the winning move
                else:
                    col_ai = predict_plus_move(cnn_model, GAME_BOARD, 'minus')

            GAME_BOARD = update_board(GAME_BOARD, 'minus', col_ai)
            result_ai = check_for_win(GAME_BOARD, col_ai)
            if result_ai[2:] == 'minus':
                status = "model_win"
                board_image = draw_board_image(GAME_BOARD)
                return {"status": status, "board_image": board_image}

            if len(find_legal_moves(GAME_BOARD)) == 0:
                status = "tie"
                board_image = draw_board_image(GAME_BOARD)
                return {"status": status, "board_image": board_image}

            board_image = draw_board_image(GAME_BOARD)
            return {"status": "continue", "board_image": board_image}

        else:
            
            GAME_BOARD = update_board(GAME_BOARD, 'minus', col) #moved this up top
            result_user = check_for_win(GAME_BOARD, col)
            if result_user[2:] == 'minus':
                status = "user_win"
                board_image = draw_board_image(GAME_BOARD)
                return {"status": status, "board_image": board_image}

            if len(find_legal_moves(GAME_BOARD)) == 0:
                status = "tie"
                board_image = draw_board_image(GAME_BOARD)
                return {"status": status, "board_image": board_image} #ned of new code placement


            if GAME_MODE == "user_vs_transformer":
                winning_move = look_for_win(GAME_BOARD) #added this function
                if winning_move != -1:
                    col_ai = winning_move  # AI takes the winning move
                else:
                    col_ai = predict_plus_move(transformer_model, GAME_BOARD, 'minus')
            else:
                winning_move = look_for_win(GAME_BOARD)
                if winning_move != -1:
                    col_ai = winning_move  # AI takes the winning move
                else:
                    col_ai = predict_plus_move(cnn_model, GAME_BOARD, 'minus')

            GAME_BOARD = update_board(GAME_BOARD, 'plus', col_ai)
            result_ai = check_for_win(GAME_BOARD, col_ai)
            if result_ai[2:] == 'plus':
                status = "model_win"
                board_image = draw_board_image(GAME_BOARD)
                return {"status": status, "board_image": board_image}

            if len(find_legal_moves(GAME_BOARD)) == 0:
                status = "tie"
                board_image = draw_board_image(GAME_BOARD)
                return {"status": status, "board_image": board_image}


            board_image = draw_board_image(GAME_BOARD)
            return {"status": "continue", "board_image": board_image}


############################################
# 4) STARTUP
############################################

def main():
    load_models()
    global GAME_BOARD
    GAME_BOARD = init_board()
    print("Ready to accept Anvil calls...")
    anvil.server.wait_forever()

if __name__ == "__main__":
    main()