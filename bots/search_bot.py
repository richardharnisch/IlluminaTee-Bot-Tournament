"""
Search Bot: Enhanced chess engine with advanced search techniques.
- Everything from alpha_beta_bot, plus:
- Transposition table (Zobrist hashing)
- Null move pruning
- Killer moves + history heuristic
- Principal Variation Search (PVS)
- Late Move Reductions (LMR)
- Check extensions
- Aspiration windows
"""

import chess
import chess.polyglot
import time
from src.definitions import Engine, GameBoard, GameMove

# Piece values
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

# TT entry types
TT_EXACT = 0
TT_ALPHA = 1  # Upper bound (failed low)
TT_BETA = 2   # Lower bound (failed high)

# fmt: off
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
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_EG,
}

MIRROR = [chess.square_mirror(sq) for sq in range(64)]

# LMR reduction table (precomputed)
LMR_TABLE = [[0] * 64 for _ in range(64)]
for d in range(1, 64):
    for m in range(1, 64):
        import math
        LMR_TABLE[d][m] = max(0, int(0.75 + math.log(d) * math.log(m) / 2.25))


class SearchTimeout(Exception):
    pass


class ChessBot(Engine):
    def __init__(self, opening_position: GameBoard) -> None:
        super().__init__(opening_position)
        self.total_time_used = 0.0
        self.moves_made = 0
        self.nodes = 0
        self.time_limit = 0.0
        self.search_start = 0.0

        # Transposition table
        self.tt = {}
        self.tt_max_size = 2_000_000

        # Killer moves: 2 slots per ply, up to 128 ply
        self.killers = [[None, None] for _ in range(128)]

        # History heuristic: [color][from_sq][to_sq]
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]

    def _game_phase(self, board):
        phase = 0
        for pt, weight in [(chess.KNIGHT, 1), (chess.BISHOP, 1),
                           (chess.ROOK, 2), (chess.QUEEN, 4)]:
            phase += len(board.pieces(pt, chess.WHITE)) * weight
            phase += len(board.pieces(pt, chess.BLACK)) * weight
        return min(phase, 24)

    def evaluate(self, board):
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

        score = (mg_score * phase + eg_score * (24 - phase)) // 24
        return score if board.turn == chess.WHITE else -score

    def _check_time(self):
        self.nodes += 1
        if self.nodes & 2047 == 0:
            if time.perf_counter() - self.search_start > self.time_limit:
                raise SearchTimeout()

    def _tt_probe(self, key, depth, alpha, beta):
        """Probe transposition table. Returns (score, move) or (None, move)."""
        entry = self.tt.get(key)
        if entry is None:
            return None, None
        tt_depth, tt_score, tt_flag, tt_move = entry
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score, tt_move
            if tt_flag == TT_ALPHA and tt_score <= alpha:
                return alpha, tt_move
            if tt_flag == TT_BETA and tt_score >= beta:
                return beta, tt_move
        return None, tt_move

    def _tt_store(self, key, depth, score, flag, move):
        """Store entry in transposition table."""
        if len(self.tt) >= self.tt_max_size:
            # Simple replacement: always replace
            pass
        self.tt[key] = (depth, score, flag, move)

    def _mvv_lva(self, board, move):
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            if victim is None:
                victim = chess.PAWN
            attacker = board.piece_type_at(move.from_square)
            return PIECE_VALUES.get(victim, 0) * 10 - PIECE_VALUES.get(attacker, 0)
        return 0

    def _order_moves(self, board, moves, ply, tt_move=None):
        """Order moves: TT move > captures (MVV-LVA) > killers > history."""
        scored = []
        for move in moves:
            score = 0
            if move == tt_move:
                score = 10_000_000
            elif board.is_capture(move):
                score = 5_000_000 + self._mvv_lva(board, move)
            elif move == self.killers[ply][0]:
                score = 2_000_000
            elif move == self.killers[ply][1]:
                score = 1_900_000
            else:
                color = 1 if board.turn else 0
                score = self.history[color][move.from_square][move.to_square]
            scored.append((score, move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _store_killer(self, move, ply):
        if move != self.killers[ply][0]:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def quiescence(self, board, alpha, beta):
        self._check_time()

        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat + 900 < alpha:
            return alpha
        if alpha < stand_pat:
            alpha = stand_pat

        captures = [m for m in board.legal_moves if board.is_capture(m)]
        captures.sort(key=lambda m: self._mvv_lva(board, m), reverse=True)

        for move in captures:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def pvs(self, board, depth, alpha, beta, ply, is_pv=True):
        """Principal Variation Search with null move pruning, LMR."""
        self._check_time()

        # Check extensions
        in_check = board.is_check()
        if in_check:
            depth += 1

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE - depth
            return 0

        # TT probe
        key = chess.polyglot.zobrist_hash(board)
        tt_score, tt_move = self._tt_probe(key, depth, alpha, beta)
        if tt_score is not None and not is_pv:
            return tt_score

        # Null move pruning (not in check, not in PV, and have pieces)
        if (not in_check and not is_pv and depth >= 3
                and board.has_legal_en_passant() is False  # safe to pass
                and self._has_non_pawn_material(board)):
            R = 3 if depth >= 6 else 2
            board.push(chess.Move.null())
            null_score = -self.pvs(board, depth - 1 - R, -beta, -beta + 1, ply + 1, False)
            board.pop()
            if null_score >= beta:
                return beta

        moves = list(board.legal_moves)
        ordered = self._order_moves(board, moves, ply, tt_move)

        best_score = -INF
        best_move = None
        moves_searched = 0

        for move in ordered:
            board.push(move)

            if moves_searched == 0:
                # First move: full window search
                score = -self.pvs(board, depth - 1, -beta, -alpha, ply + 1, is_pv)
            else:
                # LMR: reduce depth for late quiet moves
                reduction = 0
                if (depth >= 3 and moves_searched >= 3
                        and not in_check and not board.is_check()
                        and not board.is_capture(move)
                        and move.promotion is None):
                    reduction = LMR_TABLE[min(depth, 63)][min(moves_searched, 63)]
                    if is_pv:
                        reduction = max(0, reduction - 1)

                # Null window search (with possible reduction)
                score = -self.pvs(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, False)

                # Re-search if failed high and was reduced
                if score > alpha and reduction > 0:
                    score = -self.pvs(board, depth - 1, -alpha - 1, -alpha, ply + 1, False)

                # Re-search with full window if still fails high in PV
                if score > alpha and score < beta:
                    score = -self.pvs(board, depth - 1, -beta, -alpha, ply + 1, True)

            board.pop()
            moves_searched += 1

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Store killer and history for quiet moves
                if not board.is_capture(move):
                    self._store_killer(move, ply)
                    color = 1 if board.turn else 0
                    self.history[color][move.from_square][move.to_square] += depth * depth
                break

        # Store in TT
        if best_score <= alpha:
            flag = TT_ALPHA
        elif best_score >= beta:
            flag = TT_BETA
        else:
            flag = TT_EXACT
        self._tt_store(key, depth, best_score, flag, best_move)

        return best_score

    def _has_non_pawn_material(self, board):
        """Check if side to move has non-pawn material."""
        color = board.turn
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(pt, color):
                return True
        return False

    def iterative_deepening(self, board, time_budget):
        self.search_start = time.perf_counter()
        self.time_limit = time_budget
        self.nodes = 0

        # Reset killer and history for new search
        self.killers = [[None, None] for _ in range(128)]
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]

        best_move = None
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]

        prev_score = 0

        for depth in range(1, 100):
            try:
                # Aspiration windows for depth >= 5
                if depth >= 5:
                    window = 50
                    alpha = prev_score - window
                    beta = prev_score + window
                    score = self._root_search(board, depth, alpha, beta)

                    # Widen window if failed
                    if score <= alpha:
                        alpha = -INF
                        score = self._root_search(board, depth, alpha, beta)
                    if score >= beta:
                        beta = INF
                        score = self._root_search(board, depth, alpha, beta)
                else:
                    score = self._root_search(board, depth, -INF, INF)

                prev_score = score
                best_move = self._root_best_move

                elapsed = time.perf_counter() - self.search_start
                if elapsed > time_budget * 0.5:
                    break
                if score >= MATE_SCORE - 100:
                    break

            except SearchTimeout:
                if self._root_best_move:
                    best_move = self._root_best_move
                break

        return best_move

    def _root_search(self, board, depth, alpha, beta):
        """Root-level search returning score; sets self._root_best_move."""
        key = chess.polyglot.zobrist_hash(board)
        _, tt_move = self._tt_probe(key, 0, -INF, INF)

        moves = list(board.legal_moves)
        ordered = self._order_moves(board, moves, 0, tt_move)

        # Also put previous best move first
        if hasattr(self, '_root_best_move') and self._root_best_move in ordered:
            ordered.remove(self._root_best_move)
            ordered.insert(0, self._root_best_move)

        best_score = -INF
        self._root_best_move = ordered[0] if ordered else None
        moves_searched = 0

        for move in ordered:
            board.push(move)

            if moves_searched == 0:
                score = -self.pvs(board, depth - 1, -beta, -alpha, 1, True)
            else:
                score = -self.pvs(board, depth - 1, -alpha - 1, -alpha, 1, False)
                if score > alpha and score < beta:
                    score = -self.pvs(board, depth - 1, -beta, -alpha, 1, True)

            board.pop()
            moves_searched += 1

            if score > best_score:
                best_score = score
                self._root_best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        # Store root in TT
        if best_score <= alpha:
            flag = TT_ALPHA
        elif best_score >= beta:
            flag = TT_BETA
        else:
            flag = TT_EXACT
        self._tt_store(key, depth, best_score, flag, self._root_best_move)

        return best_score

    def __call__(self, board: GameBoard) -> GameMove:
        move_start = time.perf_counter()
        self.moves_made += 1

        ply = board.ply()
        remaining_estimate = max(1.0, 300.0 - self.total_time_used)
        estimated_moves_left = max(15, 50 - self.moves_made)

        time_budget = remaining_estimate / estimated_moves_left
        if self.moves_made <= 3:
            time_budget = min(time_budget, 0.8)
        time_budget = min(time_budget, remaining_estimate * 0.10)
        time_budget = max(0.1, time_budget)

        # Clear TT if it's too large
        if len(self.tt) > self.tt_max_size:
            self.tt.clear()

        move = self.iterative_deepening(board, time_budget)

        elapsed = time.perf_counter() - move_start
        self.total_time_used += elapsed

        return move
