from os import getenv

from dotenv import load_dotenv

load_dotenv(override=True)

default_model = getenv("DEFAULT_MODEL")
