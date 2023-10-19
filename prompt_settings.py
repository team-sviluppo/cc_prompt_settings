import time
import cat.factory.llm as llms
import langchain

from cat.log import log
from typing import List, Union, Dict
from datetime import timedelta
from langchain.docstore.document import Document
from langchain.tools.base import BaseTool
from langchain.agents import load_tools
from cat.utils import verbal_timedelta
from cat.mad_hatter.decorators import tool, hook
from cat.log import log

# Default prompt settings
lang = "Italian"
only_local = False
disable_episodic = False
disable_declarative = False
disable_procedural = False
custom_prefix = ""
number_of_declarative_items = 5
declarative_threshold = 0.5
number_of_episodic_items = 5
episodic_threshold = 0.5


def update_variables(settings):
    global only_local, custom_prefix, lang, disable_episodic, disable_declarative, disable_procedural, number_of_episodic_items, number_of_declarative_items, declarative_threshold, episodic_threshold
    lang = settings["language"]
    only_local = settings["only_local_responses"]
    disable_episodic = settings["disable_episodic_memories"]
    disable_declarative = settings["disable_declarative_memories"]
    disable_procedural = settings["disable_procedural_memories"]
    custom_prefix = settings["prompt_prefix"]
    number_of_declarative_items = settings["number_of_declarative_items"]
    declarative_threshold = settings["declarative_threshold"]
    number_of_episodic_items = settings["number_of_episodic_items"]
    episodic_threshold = settings["episodic_threshold"]


@hook(priority=10)
def before_cat_reads_message(user_message_json, cat):
    settings = cat.mad_hatter.plugins["cc_prompt_settings"].load_settings()
    update_variables(settings)

    return user_message_json


@hook(priority=10)
def agent_prompt_prefix(prefix, cat) -> str:
    global custom_prefix
    prefix = custom_prefix
    return prefix


@hook(priority=10)
def agent_prompt_suffix(prompt_suffix, cat) -> str:
    global lang
    if lang == "English":
        prompt_suffix = prompt_suffix_en(prompt_suffix, cat)
    if lang == "Italian":
        log.warning("ok italiano")
        prompt_suffix = prompt_suffix_it(prompt_suffix, cat)
        log.warning(prompt_suffix)
    return prompt_suffix


def prompt_suffix_en(prompt_suffix, cat) -> str:
    global disable_declarative, disable_episodic
    context = True
    if disable_episodic and disable_declarative:
        context = False

    if context:
        prompt_suffix = """
# Context

{episodic_memory}

{declarative_memory}

{tools_output}

## Conversation until now:{chat_history}
 - Human: {input}
 - AI: """

    else:
        prompt_suffix = """
{episodic_memory}
{declarative_memory}
{tools_output}

## Conversation until now:{chat_history}
 - Human: {input}
 - AI: """
    return prompt_suffix


def prompt_suffix_it(prompt_suffix, cat) -> str:
    global disable_declarative, disable_episodic
    context = True
    if disable_episodic and disable_declarative:
        context = False

    if context:
        prompt_suffix = """
# Contesto

{episodic_memory}

{declarative_memory}

{tools_output}

## Conversazione fino ad ora tra te (AI) e l'umano (Human):{chat_history}
- Human: {input}
- AI: """

    else:
        prompt_suffix = """
{episodic_memory}
{declarative_memory}
{tools_output}

## Conversazione fino ad ora tra te (AI) e l'umano (Human):{chat_history}
- Human: {input}
- AI: """
    return prompt_suffix


@hook(priority=1)
def before_cat_recalls_procedural_memories(declarative_recall_config: dict, cat) -> dict:
    global disable_procedural
    if disable_procedural:
        declarative_recall_config["k"] = 0

    return declarative_recall_config


@hook(priority=1)
def before_cat_recalls_declarative_memories(declarative_recall_config: dict, cat) -> dict:
    global disable_declarative, number_of_declarative_items, declarative_threshold
    if disable_declarative:
        custom_k = 0
    else:
        custom_k = number_of_declarative_items
    declarative_recall_config["k"] = custom_k
    declarative_recall_config["threshold"] = declarative_threshold

    return declarative_recall_config


@hook(priority=1)
def before_cat_recalls_episodic_memories(episodic_recall_config: dict, cat) -> dict:
    global disable_episodic, number_of_episodic_items, episodic_threshold
    if disable_episodic:
        custom_k = 0
    else:
        custom_k = number_of_episodic_items
    episodic_recall_config["k"] = custom_k
    episodic_recall_config["threshold"] = episodic_threshold

    return episodic_recall_config


def verbal_timedelta_ita(td: timedelta) -> str:
    if td.days != 0:
        abs_days = abs(td.days)
        if abs_days > 7:
            abs_delta = "{} settimane".format(td.days // 7)
        else:
            abs_delta = "{} giorni".format(td.days)
    else:
        abs_minutes = abs(td.seconds) // 60
        if abs_minutes > 60:
            abs_delta = "{} ore".format(abs_minutes // 60)
        else:
            abs_delta = "{} minuti".format(abs_minutes)
    if td < timedelta(0):
        return "{} fa".format(abs_delta)
    else:
        return "{} fa".format(abs_delta)


@hook(priority=1)
def before_agent_starts(agent_input, cat) -> Union[None, Dict]:
    if only_local:
        num_declarative_memories = len(cat.working_memory["declarative_memories"])
        if num_declarative_memories == 0:
            return {"output": "Scusami, non ho informazioni su questo tema."}
    return None
