"""
Alpha-Beta Bot: Solid foundation chess engine.
- Iterative deepening alpha-beta search
- Piece-square tables (middlegame + endgame)
- Material evaluation
- Quiescence search with delta pruning
- MVV-LVA move ordering
- Internal time management
"""

import chess
import time
from src.definitions import Engine, GameBoard, GameMove

# Piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

INF = 1_000_000
MATE_SCORE = 100_000

# fmt: off
# Piece-Square Tables (from White's perspective, a1=index 0, h8=index 63)
# Middlegame PSTs
PAWN_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_MG = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_MG = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_MG = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_MG = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

# Endgame PSTs
PAWN_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KING_EG = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]
# fmt: on

MG_TABLES = {
    chess.PAWN: PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_MG,
}

EG_TABLES = {
    chess.PAWN: PAWN_EG,
    chess.KNIGHT: KNIGHT_MG,  # Same for knight in endgame
    chess.BISHOP: BISHOP_MG,  # Same for bishop in endgame
    chess.ROOK: ROOK_MG,      # Same for rook in endgame
    chess.QUEEN: QUEEN_MG,    # Same for queen in endgame
    chess.KING: KING_EG,
}

# Mirror table for black pieces (flip rank)
MIRROR = [chess.square_mirror(sq) for sq in range(64)]


class SearchTimeout(Exception):
    """Raised when search time expires."""
    pass


def mvv_lva_score(board, move):
    """Most Valuable Victim - Least Valuable Attacker score for captures."""
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        if victim is None:
            # En passant
            victim = chess.PAWN
        attacker = board.piece_type_at(move.from_square)
        return PIECE_VALUES.get(victim, 0) * 10 - PIECE_VALUES.get(attacker, 0)
    return 0


class ChessBot(Engine):
    def __init__(self, opening_position: GameBoard) -> None:
        super().__init__(opening_position)
        self.total_time_used = 0.0
        self.moves_made = 0
        self.nodes = 0
        self.time_limit = 0.0
        self.search_start = 0.0
        self.best_move_root = None

    def _game_phase(self, board):
        """Compute game phase: 24 = opening, 0 = endgame."""
        phase = 0
        for pt, weight in [(chess.KNIGHT, 1), (chess.BISHOP, 1),
                           (chess.ROOK, 2), (chess.QUEEN, 4)]:
            phase += len(board.pieces(pt, chess.WHITE)) * weight
            phase += len(board.pieces(pt, chess.BLACK)) * weight
        return min(phase, 24)

    def evaluate(self, board):
        """Evaluate position from side-to-move perspective."""
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        phase = self._game_phase(board)

        mg_score = 0
        eg_score = 0

        for pt in chess.PIECE_TYPES:
            for sq in board.pieces(pt, chess.WHITE):
                mg_score += PIECE_VALUES.get(pt, 0) + MG_TABLES[pt][MIRROR[sq]]
                eg_score += PIECE_VALUES.get(pt, 0) + EG_TABLES[pt][MIRROR[sq]]
            for sq in board.pieces(pt, chess.BLACK):
                mg_score -= PIECE_VALUES.get(pt, 0) + MG_TABLES[pt][sq]
                eg_score -= PIECE_VALUES.get(pt, 0) + EG_TABLES[pt][sq]

        # Tapered eval: interpolate between midgame and endgame
        score = (mg_score * phase + eg_score * (24 - phase)) // 24

        return score if board.turn == chess.WHITE else -score

    def _check_time(self):
        """Check time every 2048 nodes."""
        self.nodes += 1
        if self.nodes & 2047 == 0:
            if time.perf_counter() - self.search_start > self.time_limit:
                raise SearchTimeout()

    def quiescence(self, board, alpha, beta, depth=0):
        """Quiescence search: only search captures to avoid horizon effect."""
        self._check_time()

        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        # Delta pruning
        if stand_pat + 900 < alpha:
            return alpha
        if alpha < stand_pat:
            alpha = stand_pat

        # Generate and sort captures
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        captures.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)

        for move in captures:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, depth - 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def alpha_beta(self, board, depth, alpha, beta):
        """Alpha-beta search with negamax framework."""
        self._check_time()

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE - depth  # Prefer faster mates
            return 0  # Draw

        best_score = -INF
        moves = list(board.legal_moves)

        # Move ordering: captures first (sorted by MVV-LVA), then non-captures
        captures = []
        quiets = []
        for m in moves:
            if board.is_capture(m):
                captures.append(m)
            else:
                quiets.append(m)
        captures.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)
        ordered_moves = captures + quiets

        for move in ordered_moves:
            board.push(move)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        return best_score

    def iterative_deepening(self, board, time_budget):
        """Iterative deepening search."""
        self.search_start = time.perf_counter()
        self.time_limit = time_budget
        self.nodes = 0

        best_move = None
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]

        for depth in range(1, 100):
            try:
                best_score = -INF
                best_move_this_depth = None

                # Move ordering at root: captures first
                captures = []
                quiets = []
                for m in moves:
                    if board.is_capture(m):
                        captures.append(m)
                    else:
                        quiets.append(m)
                captures.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)

                # Put previous best move first
                ordered_moves = []
                if best_move is not None:
                    ordered_moves.append(best_move)
                for m in captures + quiets:
                    if m != best_move:
                        ordered_moves.append(m)

                for move in ordered_moves:
                    board.push(move)
                    score = -self.alpha_beta(board, depth - 1, -INF, -best_score if best_move_this_depth else -(-INF))
                    board.pop()

                    if score > best_score:
                        best_score = score
                        best_move_this_depth = move

                best_move = best_move_this_depth

                # Check if we've used more than half our budget — don't start next depth
                elapsed = time.perf_counter() - self.search_start
                if elapsed > time_budget * 0.5:
                    break

                # Found a mate, no need to search deeper
                if best_score >= MATE_SCORE - 100:
                    break

            except SearchTimeout:
                # Return best move found so far
                if best_move_this_depth:
                    best_move = best_move_this_depth
                break

        return best_move

    def __call__(self, board: GameBoard) -> GameMove:
        move_start = time.perf_counter()
        self.moves_made += 1

        ply = board.ply()
        remaining_estimate = max(1.0, 300.0 - self.total_time_used)
        estimated_moves_left = max(15, 50 - self.moves_made)

        time_budget = remaining_estimate / estimated_moves_left
        # First 3 moves: be conservative (handles unknown time controls)
        if self.moves_made <= 3:
            time_budget = min(time_budget, 0.8)
        time_budget = min(time_budget, remaining_estimate * 0.10)
        time_budget = max(0.1, time_budget)

        move = self.iterative_deepening(board, time_budget)

        elapsed = time.perf_counter() - move_start
        self.total_time_used += elapsed

        return move
