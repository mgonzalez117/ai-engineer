import chess

def validate_fen_or_raise(fen: str) -> None:
    # python-chess lève ValueError si FEN invalide
    chess.Board(fen)