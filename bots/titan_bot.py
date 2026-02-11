"""
Titan Bot: Strongest chess engine with advanced evaluation and search.
- Full PVS with iterative deepening
- Transposition table (Zobrist hashing)
- Null move pruning, killer moves, history heuristic
- LMR, check extensions, aspiration windows
- Advanced evaluation:
  - Bishop pair bonus
  - Passed pawn evaluation
  - Doubled/isolated pawn penalties
  - Rook on open/semi-open file
  - King safety (pawn shield, open files near king)
  - Mobility bonus
  - Tapered evaluation (midgame/endgame interpolation)
- Futility pruning
"""

import chess
import chess.polyglot
import math
import time
from src.definitions import Engine, GameBoard, GameMove

# Piece values
PIECE_VALUES_MG = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PIECE_VALUES_EG = {
    chess.PAWN: 120,
    chess.KNIGHT: 310,
    chess.BISHOP: 340,
    chess.ROOK: 520,
    chess.QUEEN: 950,
    chess.KING: 0,
}

# Phase weights for tapered eval
PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
TOTAL_PHASE = 24

INF = 1_000_000
MATE_SCORE = 100_000

TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2

# Futility margins by depth
FUTILITY_MARGINS = [0, 200, 350, 500]

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

KNIGHT_EG = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_EG = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  0, 10, 15, 15, 10,  0,-10,
   -10,  0, 10, 15, 15, 10,  0,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
]

QUEEN_EG = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5, 10, 10,  5,  0, -5,
    -5,  0,  5, 10, 10,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
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
    chess.KNIGHT: KNIGHT_EG,
    chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG,
    chess.QUEEN: QUEEN_EG,
    chess.KING: KING_EG,
}

MIRROR = [chess.square_mirror(sq) for sq in range(64)]

# LMR reduction table
LMR_TABLE = [[0] * 64 for _ in range(64)]
for d in range(1, 64):
    for m in range(1, 64):
        LMR_TABLE[d][m] = max(0, int(0.75 + math.log(d) * math.log(m) / 2.25))

# Passed pawn bonus by rank (from White's perspective, rank 1=index 0 to rank 8=index 7)
PASSED_PAWN_BONUS_MG = [0, 5, 10, 20, 35, 60, 100, 0]
PASSED_PAWN_BONUS_EG = [0, 10, 20, 40, 70, 120, 200, 0]

# File masks for pawn structure
FILE_MASKS = []
for f in range(8):
    mask = 0
    for r in range(8):
        mask |= 1 << (r * 8 + f)
    FILE_MASKS.append(mask)

# Adjacent file masks
ADJACENT_FILE_MASKS = []
for f in range(8):
    mask = 0
    if f > 0:
        mask |= FILE_MASKS[f - 1]
    if f < 7:
        mask |= FILE_MASKS[f + 1]
    ADJACENT_FILE_MASKS.append(mask)


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

        # Killer moves
        self.killers = [[None, None] for _ in range(128)]

        # History heuristic
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]

    def _game_phase(self, board):
        phase = 0
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            w = PHASE_WEIGHTS[pt]
            phase += len(board.pieces(pt, chess.WHITE)) * w
            phase += len(board.pieces(pt, chess.BLACK)) * w
        return min(phase, TOTAL_PHASE)

    def _pawn_structure_eval(self, board):
        """Evaluate pawn structure: doubled, isolated, passed pawns."""
        mg_score = 0
        eg_score = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            pawns = board.pieces(chess.PAWN, color)
            own_pawn_bb = int(pawns)
            opp_pawn_bb = int(board.pieces(chess.PAWN, not color))

            for sq in pawns:
                f = chess.square_file(sq)
                r = chess.square_rank(sq)

                # Doubled pawns: another own pawn on the same file
                file_pawns = own_pawn_bb & FILE_MASKS[f]
                if bin(file_pawns).count('1') > 1:
                    mg_score -= sign * 10
                    eg_score -= sign * 20

                # Isolated pawns: no own pawns on adjacent files
                if not (own_pawn_bb & ADJACENT_FILE_MASKS[f]):
                    mg_score -= sign * 15
                    eg_score -= sign * 20

                # Passed pawns: no opponent pawns on same or adjacent files ahead
                is_passed = True
                if color == chess.WHITE:
                    # Check files f-1, f, f+1 for opponent pawns on ranks > r
                    for check_f in range(max(0, f - 1), min(8, f + 2)):
                        for check_r in range(r + 1, 8):
                            if opp_pawn_bb & (1 << (check_r * 8 + check_f)):
                                is_passed = False
                                break
                        if not is_passed:
                            break
                    if is_passed:
                        mg_score += sign * PASSED_PAWN_BONUS_MG[r]
                        eg_score += sign * PASSED_PAWN_BONUS_EG[r]
                else:
                    for check_f in range(max(0, f - 1), min(8, f + 2)):
                        for check_r in range(0, r):
                            if opp_pawn_bb & (1 << (check_r * 8 + check_f)):
                                is_passed = False
                                break
                        if not is_passed:
                            break
                    if is_passed:
                        mg_score += sign * PASSED_PAWN_BONUS_MG[7 - r]
                        eg_score += sign * PASSED_PAWN_BONUS_EG[7 - r]

        return mg_score, eg_score

    def _king_safety(self, board):
        """Evaluate king safety: pawn shield, open files near king."""
        mg_score = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            king_sq = board.king(color)
            if king_sq is None:
                continue
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            own_pawns = int(board.pieces(chess.PAWN, color))

            # Pawn shield bonus (pawns on ranks 2-3 near king)
            shield_bonus = 0
            for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                if color == chess.WHITE:
                    # Check rank 1 and 2 (0-indexed) for white pawns
                    if own_pawns & (1 << (1 * 8 + f)):
                        shield_bonus += 15
                    elif own_pawns & (1 << (2 * 8 + f)):
                        shield_bonus += 5
                else:
                    if own_pawns & (1 << (6 * 8 + f)):
                        shield_bonus += 15
                    elif own_pawns & (1 << (5 * 8 + f)):
                        shield_bonus += 5

            mg_score += sign * shield_bonus

            # Open file near king penalty
            all_pawns = own_pawns | int(board.pieces(chess.PAWN, not color))
            for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                if not (all_pawns & FILE_MASKS[f]):
                    mg_score -= sign * 20  # Open file near king

        return mg_score

    def _rook_eval(self, board):
        """Evaluate rook placement on open/semi-open files."""
        mg_score = 0
        eg_score = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            own_pawns = int(board.pieces(chess.PAWN, color))
            opp_pawns = int(board.pieces(chess.PAWN, not color))

            for sq in board.pieces(chess.ROOK, color):
                f = chess.square_file(sq)
                own_on_file = own_pawns & FILE_MASKS[f]
                opp_on_file = opp_pawns & FILE_MASKS[f]

                if not own_on_file and not opp_on_file:
                    # Open file
                    mg_score += sign * 20
                    eg_score += sign * 15
                elif not own_on_file:
                    # Semi-open file
                    mg_score += sign * 10
                    eg_score += sign * 10

        return mg_score, eg_score

    def _mobility_eval(self, board):
        """Simple mobility evaluation based on legal move count difference."""
        # Count mobility for side to move
        own_mobility = board.legal_moves.count()

        board.push(chess.Move.null())
        opp_mobility = board.legal_moves.count()
        board.pop()

        diff = own_mobility - opp_mobility
        return diff * 3  # 3cp per move advantage

    def evaluate(self, board):
        """Full evaluation with all features."""
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        phase = self._game_phase(board)

        mg_score = 0
        eg_score = 0

        # Material + PST
        white_bishops = 0
        black_bishops = 0
        for pt in chess.PIECE_TYPES:
            for sq in board.pieces(pt, chess.WHITE):
                mg_score += PIECE_VALUES_MG.get(pt, 0) + MG_TABLES[pt][MIRROR[sq]]
                eg_score += PIECE_VALUES_EG.get(pt, 0) + EG_TABLES[pt][MIRROR[sq]]
                if pt == chess.BISHOP:
                    white_bishops += 1
            for sq in board.pieces(pt, chess.BLACK):
                mg_score -= PIECE_VALUES_MG.get(pt, 0) + MG_TABLES[pt][sq]
                eg_score -= PIECE_VALUES_EG.get(pt, 0) + EG_TABLES[pt][sq]
                if pt == chess.BISHOP:
                    black_bishops += 1

        # Bishop pair bonus
        if white_bishops >= 2:
            mg_score += 30
            eg_score += 50
        if black_bishops >= 2:
            mg_score -= 30
            eg_score -= 50

        # Pawn structure
        pawn_mg, pawn_eg = self._pawn_structure_eval(board)
        mg_score += pawn_mg
        eg_score += pawn_eg

        # King safety (midgame only)
        mg_score += self._king_safety(board)

        # Rook evaluation
        rook_mg, rook_eg = self._rook_eval(board)
        mg_score += rook_mg
        eg_score += rook_eg

        # Tapered eval
        score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE

        # Mobility (not tapered, evaluated from stm perspective already)
        # Skip mobility in quiescence to save time — it's computed in main eval only
        mob = self._mobility_eval(board)
        score += mob if board.turn == chess.WHITE else -mob

        return score if board.turn == chess.WHITE else -score

    def _fast_evaluate(self, board):
        """Faster evaluation for quiescence (no mobility)."""
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        phase = self._game_phase(board)
        mg_score = 0
        eg_score = 0

        for pt in chess.PIECE_TYPES:
            for sq in board.pieces(pt, chess.WHITE):
                mg_score += PIECE_VALUES_MG.get(pt, 0) + MG_TABLES[pt][MIRROR[sq]]
                eg_score += PIECE_VALUES_EG.get(pt, 0) + EG_TABLES[pt][MIRROR[sq]]
            for sq in board.pieces(pt, chess.BLACK):
                mg_score -= PIECE_VALUES_MG.get(pt, 0) + MG_TABLES[pt][sq]
                eg_score -= PIECE_VALUES_EG.get(pt, 0) + EG_TABLES[pt][sq]

        score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE
        return score if board.turn == chess.WHITE else -score

    def _check_time(self):
        self.nodes += 1
        if self.nodes & 2047 == 0:
            if time.perf_counter() - self.search_start > self.time_limit:
                raise SearchTimeout()

    def _tt_probe(self, key, depth, alpha, beta):
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
        old = self.tt.get(key)
        # Replace if: no entry, or new entry has deeper/equal depth
        if old is None or depth >= old[0]:
            self.tt[key] = (depth, score, flag, move)

    def _mvv_lva(self, board, move):
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            if victim is None:
                victim = chess.PAWN
            attacker = board.piece_type_at(move.from_square)
            return PIECE_VALUES_MG.get(victim, 0) * 10 - PIECE_VALUES_MG.get(attacker, 0)
        return 0

    def _order_moves(self, board, moves, ply, tt_move=None):
        scored = []
        for move in moves:
            score = 0
            if move == tt_move:
                score = 10_000_000
            elif board.is_capture(move):
                score = 5_000_000 + self._mvv_lva(board, move)
            elif move.promotion:
                score = 4_000_000 + PIECE_VALUES_MG.get(move.promotion, 0)
            elif ply < 128 and move == self.killers[ply][0]:
                score = 2_000_000
            elif ply < 128 and move == self.killers[ply][1]:
                score = 1_900_000
            else:
                color = 1 if board.turn else 0
                score = self.history[color][move.from_square][move.to_square]
            scored.append((score, move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _store_killer(self, move, ply):
        if ply >= 128:
            return
        if move != self.killers[ply][0]:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def _has_non_pawn_material(self, board):
        color = board.turn
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(pt, color):
                return True
        return False

    def quiescence(self, board, alpha, beta):
        self._check_time()

        stand_pat = self._fast_evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat + 900 < alpha:
            return alpha
        if alpha < stand_pat:
            alpha = stand_pat

        captures = []
        for m in board.legal_moves:
            if board.is_capture(m) or m.promotion:
                captures.append(m)
        captures.sort(key=lambda m: self._mvv_lva(board, m), reverse=True)

        for move in captures:
            # SEE-like pruning: skip captures of well-defended pieces
            # Simple version: skip if MVV-LVA is very negative (bad capture)
            if board.is_capture(move) and self._mvv_lva(board, move) < -200:
                continue

            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def pvs(self, board, depth, alpha, beta, ply, is_pv=True):
        self._check_time()

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

        # Static eval for pruning decisions
        static_eval = None

        # Null move pruning
        if (not in_check and not is_pv and depth >= 3
                and self._has_non_pawn_material(board)):
            R = 3 if depth >= 6 else 2
            board.push(chess.Move.null())
            null_score = -self.pvs(board, depth - 1 - R, -beta, -beta + 1, ply + 1, False)
            board.pop()
            if null_score >= beta:
                return beta

        moves = list(board.legal_moves)
        ordered = self._order_moves(board, moves, ply, tt_move)

        # Futility pruning setup
        can_futility = False
        if (not in_check and not is_pv and depth <= 3
                and abs(alpha) < MATE_SCORE - 100):
            if static_eval is None:
                static_eval = self._fast_evaluate(board)
            if static_eval + FUTILITY_MARGINS[depth] <= alpha:
                can_futility = True

        best_score = -INF
        best_move = None
        moves_searched = 0

        for move in ordered:
            # Futility pruning: skip quiet moves at frontier nodes
            if (can_futility and moves_searched > 0
                    and not board.is_capture(move)
                    and not move.promotion
                    and not board.gives_check(move)):
                continue

            board.push(move)

            if moves_searched == 0:
                score = -self.pvs(board, depth - 1, -beta, -alpha, ply + 1, is_pv)
            else:
                # LMR
                reduction = 0
                if (depth >= 3 and moves_searched >= 3
                        and not in_check and not board.is_check()
                        and not board.is_capture(move)
                        and move.promotion is None):
                    reduction = LMR_TABLE[min(depth, 63)][min(moves_searched, 63)]
                    if is_pv:
                        reduction = max(0, reduction - 1)

                score = -self.pvs(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, False)

                if score > alpha and reduction > 0:
                    score = -self.pvs(board, depth - 1, -alpha - 1, -alpha, ply + 1, False)

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
                if not board.is_capture(move):
                    self._store_killer(move, ply)
                    color = 1 if board.turn else 0
                    self.history[color][move.from_square][move.to_square] += depth * depth
                break

        # TT store
        if best_score <= alpha:
            flag = TT_ALPHA
        elif best_score >= beta:
            flag = TT_BETA
        else:
            flag = TT_EXACT
        self._tt_store(key, depth, best_score, flag, best_move)

        return best_score

    def _root_search(self, board, depth, alpha, beta):
        key = chess.polyglot.zobrist_hash(board)
        _, tt_move = self._tt_probe(key, 0, -INF, INF)

        moves = list(board.legal_moves)
        ordered = self._order_moves(board, moves, 0, tt_move)

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
                # LMR at root
                reduction = 0
                if (depth >= 3 and moves_searched >= 3
                        and not board.is_check()
                        and not board.is_capture(move)
                        and move.promotion is None):
                    reduction = LMR_TABLE[min(depth, 63)][min(moves_searched, 63)]
                    reduction = max(0, reduction - 1)  # Less aggressive at root

                score = -self.pvs(board, depth - 1 - reduction, -alpha - 1, -alpha, 1, False)

                if score > alpha and reduction > 0:
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

        if best_score <= alpha:
            flag = TT_ALPHA
        elif best_score >= beta:
            flag = TT_BETA
        else:
            flag = TT_EXACT
        self._tt_store(key, depth, best_score, flag, self._root_best_move)

        return best_score

    def iterative_deepening(self, board, time_budget):
        self.search_start = time.perf_counter()
        self.time_limit = time_budget
        self.nodes = 0

        self.killers = [[None, None] for _ in range(128)]
        # Age history scores (don't fully clear to preserve some knowledge)
        for c in range(2):
            for f in range(64):
                for t in range(64):
                    self.history[c][f][t] //= 4

        best_move = None
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]

        prev_score = 0

        for depth in range(1, 100):
            try:
                if depth >= 5:
                    window = 35
                    alpha = prev_score - window
                    beta = prev_score + window

                    score = self._root_search(board, depth, alpha, beta)

                    # Re-search with wider window on fail
                    if score <= alpha or score >= beta:
                        alpha = prev_score - window * 4
                        beta = prev_score + window * 4
                        score = self._root_search(board, depth, alpha, beta)

                    # Full window if still fails
                    if score <= alpha or score >= beta:
                        score = self._root_search(board, depth, -INF, INF)
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

        # Manage TT size
        if len(self.tt) > self.tt_max_size:
            self.tt.clear()

        move = self.iterative_deepening(board, time_budget)

        elapsed = time.perf_counter() - move_start
        self.total_time_used += elapsed

        return move
