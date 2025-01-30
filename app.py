import sys
import os
import io
import chess
import chess.pgn
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGridLayout, QPushButton, QLabel, QWidget, QTextEdit, 
    QVBoxLayout, QHBoxLayout, QMessageBox, QDialog, QScrollArea
)
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QFont
from PyQt5.QtCore import Qt, QSize
from dataclasses import dataclass
from typing import Dict, List, Optional
import onnxruntime
import numpy as np
from transformers import GPT2Tokenizer
import json
from pathlib import Path


@dataclass
class ChessOpening:
    name: str
    moves: str
    eco_code: Optional[str] = None
    ai_comment: Optional[str] = None  # AI-generated commentary replaces description


class OpeningDetector:
    def __init__(self):
        self.openings = self._initialize_openings()

    def _initialize_openings(self) -> Dict[str, ChessOpening]:
        # Load openings from JSON file
        json_path = Path("opening_list.json")
        if not json_path.exists():
            raise FileNotFoundError(f"Opening list file not found: {json_path}")

        with open(json_path, "r") as file:
            openings_data = json.load(file)

        # Convert JSON data to ChessOpening objects
        openings = {}
        for move_sequence, opening_data in openings_data.items():
            openings[move_sequence] = ChessOpening(
                name=opening_data["name"],
                moves=opening_data["moves"],
                eco_code=opening_data["eco_code"]
            )

        return openings

    def get_opening(self, board: chess.Board) -> Optional[ChessOpening]:
        """Identifies the chess opening based on the current board state."""
        moves_str = self._get_moves_str(board)
        matching_opening = None
        max_length = 0
        
        for opening_moves, opening in self.openings.items():
            if moves_str.startswith(opening_moves) and len(opening_moves) > max_length:
                matching_opening = opening
                max_length = len(opening_moves)
        
        return matching_opening

    def _get_moves_str(self, board: chess.Board) -> str:
        """Converts the board's move history into a string format."""
        temp_board = chess.Board()
        moves = []
        move_number = 1
        is_white_move = True

        for move in board.move_stack:
            san_move = temp_board.san(move)
            if is_white_move:
                moves.append(f"{move_number}. {san_move}")
                move_number += 1
            else:
                moves.append(san_move)
            is_white_move = not is_white_move
            temp_board.push(move)

        return " ".join(moves)

class ChessCommentaryAI:
    def __init__(self):
        """Initializes the AI model for generating chess commentary."""

        # Load configuration from config.json
        config_path = Path("config.json")
        if not config_path.exists():
            raise FileNotFoundError("Config file not found: config.json")

        with open(config_path, "r") as file:
            config = json.load(file)

        tokenizer_model = config.get("tokenizer")
        onnx_model_path = config.get("onnx_model")

        # Load tokenizer from the specified model
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model)

        # Load the ONNX model from the specified path
        self.session = onnxruntime.InferenceSession(onnx_model_path)

        self.max_length = 32

    def generate_comment(self, opening_name: str, moves: str) -> str:
        """Generates AI commentary for a given opening."""
        try:
            # Prepare the input text
            input_text = f"Commentary of Chess opening {opening_name} with moves {moves}: \n"

            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Tokenize the input text
            inputs = self.tokenizer(
                input_text,
                return_tensors="np", 
                truncation=True, 
                max_length=self.max_length,
                padding="max_length"
                )
            input_ids = inputs["input_ids"].astype(np.int64)  # Ensure int64 type
            attention_mask = inputs["attention_mask"].astype(np.int64)  # Ensure int64 type

            # Ensure padding token is set
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Run inference on the ONNX model
            outputs = self.session.run(
                None, 
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            )

            # Get the output logits
            logits = outputs[0]

            # Apply argmax to get the predicted token IDs
            output_ids = np.argmax(logits, axis=-1)

            # Decode the generated tokens to text
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            ai_gen = [generated_text[:200].rstrip(),input_text+generated_text[:200].rstrip()]
            
   
            return ai_gen 

        except Exception as e:
            return f"AI Commentary unavailable: {str(e)}"
        
class AICommentaryWindow(QDialog):
    def __init__(self, ai_comment, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Commentary")
        self.setGeometry(200, 200, 600, 400)

        # Layout for the dialog
        layout = QVBoxLayout(self)

        # Scroll area for the text
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Label to display the full AI comment
        self.comment_label = QLabel(ai_comment, self)
        self.comment_label.setWordWrap(True)
        
        # Add label to the scroll area
        scroll_area.setWidget(self.comment_label)

        # Add scroll area to the layout
        layout.addWidget(scroll_area)

        # Close button
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

class ChessBoardGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess PGN Viewer")
        self.setGeometry(100, 100, 1000, 700)
        
        # Initialize Opening Detector and AI Commentary
        self.opening_detector = OpeningDetector()
        self.commentary_ai = ChessCommentaryAI()

        # Load chess assets
        self.assets_path = "assets"
        self.board = chess.Board()
        self.current_move_index = -1
        self.moves = []
        self.white_player = ""
        self.black_player = ""
        self.white_rating = ""
        self.black_rating = ""
        self.termination = ""
        self.is_flipped = False

        self.init_ui()

    def show_popup(self, title, message, icon_type):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(icon_type)
        msg.exec_()

    def load_pgn(self, pgn_data):
        try:
            # Check if input is empty
            pgn_text = pgn_data.read()
            if not pgn_text.strip():
                self.show_popup("Error", "You must input PGN data!", QMessageBox.Warning)
                return False

            # Reset the stream for reading
            pgn_data = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_data)
            
            # Check if the input is valid PGN format
            if game is None:
                self.show_popup("Error", "You must input PGN format data!", QMessageBox.Critical)
                return False

            self.moves = list(game.mainline_moves())
            self.board.reset()
            self.current_move_index = -1

            # Update game info
            self.white_player = game.headers.get("White", "Unknown")
            self.black_player = game.headers.get("Black", "Unknown")
            self.white_rating = game.headers.get("WhiteElo", "Unknown")
            self.black_rating = game.headers.get("BlackElo", "Unknown")
            self.termination = game.headers.get("Termination", "Unknown")

            self.white_label.setText(f"White: {self.white_player}")
            self.black_label.setText(f"Black: {self.black_player}")
            self.white_rating_label.setText(f"White Rating: {self.white_rating}")
            self.black_rating_label.setText(f"Black Rating: {self.black_rating}")
            self.termination_label.setText(f"Termination: {self.termination}")

            self.chessboard_widget.update_board()
            
            # Show success message
            self.show_popup("Success", "Game loaded successfully!", QMessageBox.Information)
            return True

        except Exception as e:
            self.show_popup("Error", "You must input PGN format data!", QMessageBox.Critical)
            return False

    def load_pgn_from_textbox(self):
        pgn_text = self.pgn_input.toPlainText()
        pgn_stream = io.StringIO(pgn_text)
        self.load_pgn(pgn_stream)

    # ... rest of the ChessBoardGUI class remains the same ...
    def init_ui(self):
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Chessboard and info layout
        board_layout = QGridLayout()
        self.layout.addLayout(board_layout)

        self.white_label = QLabel("White: ")
        board_layout.addWidget(self.white_label, 0, 0, 1, 4)

        self.black_label = QLabel("Black: ")
        board_layout.addWidget(self.black_label, 0, 4, 1, 4)

        self.white_rating_label = QLabel("White Rating: ")
        board_layout.addWidget(self.white_rating_label, 1, 0, 1, 4)

        self.black_rating_label = QLabel("Black Rating: ")
        board_layout.addWidget(self.black_rating_label, 1, 4, 1, 4)

        self.termination_label = QLabel("Termination: ")
        board_layout.addWidget(self.termination_label, 2, 0, 1, 8)
        
        self.opening_label = QLabel("Opening: ")
        board_layout.addWidget(self.opening_label, 4, 0, 1, 4)

        self.opening_comment_label = QLabel("AI Comment: ")
        board_layout.addWidget(self.opening_comment_label, 5, 0, 1, 4)
        
        # Button to open AI comment in new window
        self.show_comment_button = QPushButton("Show Full AI Comment")
        self.show_comment_button.clicked.connect(self.show_ai_commentary_window)
        board_layout.addWidget(self.show_comment_button, 6, 0, 1, 4)
        
        self.chessboard_widget = ChessBoardWidget(self)
        board_layout.addWidget(self.chessboard_widget, 7, 0, 1, 8)

        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.show_previous_move)
        board_layout.addWidget(self.previous_button, 8, 0, 1, 2)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_move)
        board_layout.addWidget(self.next_button, 8, 2, 1, 2)

        self.flip_button = QPushButton("Flip Board")
        self.flip_button.clicked.connect(self.flip_board)
        board_layout.addWidget(self.flip_button, 8, 4, 1, 4)

        # Sidebar for PGN input
        self.sidebar_layout = QVBoxLayout()
        self.layout.addLayout(self.sidebar_layout)

        self.pgn_input = QTextEdit()
        self.pgn_input.setPlaceholderText("Paste PGN data here...")
        self.sidebar_layout.addWidget(self.pgn_input)

        self.load_button = QPushButton("Load Game")
        self.load_button.clicked.connect(self.load_pgn_from_textbox)
        self.sidebar_layout.addWidget(self.load_button)

    def update_opening_label(self):
        """Updates the opening label and generates AI commentary."""
        opening = self.opening_detector.get_opening(self.board)
        if opening:
            self.opening_label.setText(f"Opening : {opening.name} ({opening.eco_code})")
            
            # Generate AI comment
            moves_str = self.opening_detector._get_moves_str(self.board)
            opening_comment = self.commentary_ai.generate_comment(opening.name, moves_str)
            
            comment = "AI Comment: \n" + opening_comment[0]
            
            self.opening_comment_label.setText(comment)
        else:
            self.opening_label.setText("Opening : Unknown")
            self.opening_comment_label.setText("AI Comment: No Comment")
    
    def show_next_move(self):
        if self.current_move_index < len(self.moves) - 1:
            self.current_move_index += 1
            self.board.push(self.moves[self.current_move_index])
            self.chessboard_widget.update_board()
            self.update_opening_label()  # Add this line

    def show_previous_move(self):
        if self.current_move_index >= 0:
            self.board.pop()
            self.current_move_index -= 1
            self.chessboard_widget.update_board()
            self.update_opening_label()  # Add this line

    def flip_board(self):
        self.is_flipped = not self.is_flipped
        self.chessboard_widget.update_board()
        
    def show_ai_commentary_window(self):
        """Open a new window to show the AI commentary."""
        opening = self.opening_detector.get_opening(self.board)
        if opening:
            moves_str = self.opening_detector._get_moves_str(self.board)
            opening_comment = self.commentary_ai.generate_comment(opening.name, moves_str)
            comment = "AI Comment: \n" + opening_comment[1]

            # Create and show the AI commentary window
            ai_commentary_window = AICommentaryWindow(comment, self)
            ai_commentary_window.exec_()
        else:
            self.show_popup("Error", "No opening detected for commentary!", QMessageBox.Warning)


# ... ChessBoardWidget class remains exactly the same ...
class ChessBoardWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.MARGIN_TOP = 30
        self.MARGIN_BOTTOM = 30
        self.MARGIN_LEFT = 30
        self.MARGIN_RIGHT = 30

    def sizeHint(self):
        return QSize(600, 600)

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_board(qp)
        qp.end()

    def draw_board(self, qp):
        size = self.size()
        board_width = size.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT
        board_height = size.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM
        square_size = min(board_width, board_height) // 8

        board_size = square_size * 8
        x_offset = self.MARGIN_LEFT + (board_width - board_size) // 2
        y_offset = self.MARGIN_TOP + (board_height - board_size) // 2

        # Draw the chessboard squares
        for row in range(8):
            for col in range(8):
                color = Qt.lightGray if (row + col) % 2 == 0 else Qt.gray
                qp.setBrush(QBrush(color))
                qp.drawRect(
                    x_offset + col * square_size, 
                    y_offset + row * square_size, 
                    square_size, 
                    square_size
                )

                display_row = 7 - row if self.parent.is_flipped else row
                display_col = 7 - col if self.parent.is_flipped else col
                square = chess.square(display_col, 7 - display_row)
                piece = self.parent.board.piece_at(square)
                if piece:
                    piece_color = "l" if piece.color == chess.WHITE else "d"
                    piece_type = chess.PIECE_SYMBOLS[piece.piece_type]
                    piece_name = f"Chess_{piece_type}{piece_color}t60.png"
                    piece_path = os.path.join(self.parent.assets_path, piece_name)
                    pixmap = QPixmap(piece_path).scaled(square_size, square_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    qp.drawPixmap(
                        x_offset + col * square_size + (square_size - pixmap.width()) // 2, 
                        y_offset + row * square_size + (square_size - pixmap.height()) // 2, 
                        pixmap
                    )

        # Draw column and row notations
        font = QFont()
        font.setPointSize(12)
        qp.setFont(font)
        qp.setPen(Qt.black)

        # Column notation (A-H)
        for col in range(8):
            adjusted_col = 7 - col if self.parent.is_flipped else col
            label = chr(ord('A') + adjusted_col)
            
            label_x = x_offset + col * square_size + (square_size // 2) - 6
            
            # Bottom label
            label_y = y_offset + board_size + 20
            qp.drawText(label_x, label_y, label)
            
            # Top label
            label_y = y_offset - 10
            qp.drawText(label_x, label_y, label)

        # Row notation (1-8)
        for row in range(8):
            label = str(row + 1) if self.parent.is_flipped else str(8 - row)
            label_y = y_offset + row * square_size + (square_size // 2) + 6
            
            # Left side
            label_x = x_offset - 20
            qp.drawText(label_x, label_y, label)
            
            # Right side
            label_x = x_offset + board_size + 10
            qp.drawText(label_x, label_y, label)

    def update_board(self):
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ChessBoardGUI()
    viewer.show()
    sys.exit(app.exec_())