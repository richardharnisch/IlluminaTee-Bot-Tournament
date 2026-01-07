import threading
import _thread
from time import perf_counter
from contextlib import contextmanager

from src.definitions import Engine, GameBoard, GameMove


class TimeoutException(Exception):
    """Raised when a timed operation exceeds the allowed duration."""

    def __init__(self, msg: str = ""):
        super().__init__(msg)
        self.msg = msg


class EngineContainer(Engine):
    def __init__(
        self, engine_class: type, seconds: float = 60.0, seconds_increment: float = 0.0
    ) -> None:
        super().__init__()
        self.engine_class = engine_class
        self.engine = None
        self.name = self.engine_class.name
        self.time = seconds
        self.increment = seconds_increment
        self.reset()

    @contextmanager
    def _time(self, timeout: float, message: str = "Operation timed out"):
        """Context manager that interrupts the main thread if `timeout` elapses.
        Yields the perf_counter() start time.
        """
        timed_out = threading.Event()

        def _fire():
            timed_out.set()
            _thread.interrupt_main()

        timer = threading.Timer(timeout, _fire)
        timer.daemon = True
        timer.start()

        start_time = perf_counter()
        try:
            yield start_time
        except KeyboardInterrupt:
            if timed_out.is_set():
                # suppress exception chaining to avoid "During handling of the above exception..."
                raise TimeoutException(message) from None
            raise
        finally:
            timer.cancel()

    def _initialise(self) -> None:
        if not self.engine:
            with self._time(10, f"Initialisation of engine {self.name} timed out"):
                self.engine = self.engine_class()

    def reset(
        self, seconds: float | None = None, increment: float | None = None
    ) -> None:
        self.remaining_time = seconds or self.time
        self.increment = increment or self.increment

        if not self.engine:
            return

        with self._time(120, f"Unload operation of engine {self.name} timed out"):
            self.engine.unload()

        del self.engine
        self.engine = None

    def __call__(self, board: GameBoard) -> GameMove:
        self._initialise()
        with self._time(
            self.remaining_time, f"Engine {self.name} timed out."
        ) as start_time:
            move = self.engine(board)

        self.remaining_time -= perf_counter() - start_time
        self.remaining_time += self.increment
        return move
