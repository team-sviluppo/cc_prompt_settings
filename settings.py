from pydantic import BaseModel
from cat.mad_hatter.decorators import plugin
from enum import Enum


class AvailbleLanguages(Enum):
    it: str = "Italian"
    en: str = "English"


# Plugin settings
class PluginSettings(BaseModel):
    language: AvailbleLanguages = AvailbleLanguages.it
    only_local_responses: bool = False
    disable_episodic_memories: bool = False
    disable_declarative_memories: bool = False
    disable_procedural_memories: bool = False
    prompt_prefix: str = ""
    number_of_declarative_items: int = 5
    declarative_threshold: float = 0.5
    number_of_episodic_items: int = 5
    episodic_threshold: float = 0.5


# hook to give the cat settings
@plugin
def settings_schema():
    return PluginSettings.schema()
