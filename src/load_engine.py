# Stealing some old code
import os
from importlib import import_module, reload


BOT_DIR = "bots"


def load_engine(engine_name: str):
    path = os.path.join(BOT_DIR, engine_name)
    bot_module_string = f"{BOT_DIR}.{engine_name}{'.main' * os.path.isdir(path)}"
    bot_module = reload(import_module(bot_module_string))
    bot_class = getattr(bot_module, "ChessBot")
    bot_class.name = engine_name
    return bot_class
