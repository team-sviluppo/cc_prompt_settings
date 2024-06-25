from cat.log import log
from typing import List, Union, Dict
from datetime import timedelta
from cat.mad_hatter.decorators import hook

# Default prompt settings
lang = "Italian"
legacy = False
only_local = False
disable_episodic = False
disable_declarative = False
disable_procedural = False
custom_prefix = ""
number_of_declarative_items = 5
declarative_threshold = 0.5
number_of_episodic_items = 5
episodic_threshold = 0.5


def update_variables(settings, prompt_settings):
    global only_local, custom_prefix, lang, disable_episodic, disable_declarative, disable_procedural, number_of_episodic_items, number_of_declarative_items, declarative_threshold, episodic_threshold
    lang = settings["language"]
    legacy = settings["only_local_responses"]
    only_local = settings["only_local_responses"]
    disable_episodic = settings["disable_episodic_memories"]
    disable_declarative = settings["disable_declarative_memories"]
    disable_procedural = settings["disable_procedural_memories"]
    custom_prefix = settings["prompt_prefix"]
    number_of_declarative_items = settings["number_of_declarative_items"]
    declarative_threshold = settings["declarative_threshold"]
    number_of_episodic_items = settings["number_of_episodic_items"]
    episodic_threshold = settings["episodic_threshold"]

    if prompt_settings is not None:
        if "disable_episodic_memories" in prompt_settings:
            disable_episodic = prompt_settings["disable_episodic_memories"]
        if "disable_declarative_memories" in prompt_settings:
            disable_declarative = prompt_settings["disable_declarative_memories"]
        if "disable_procedural_memories" in prompt_settings:
            disable_procedural = prompt_settings["disable_procedural_memories"]
        if "number_of_declarative_items" in prompt_settings:
            number_of_declarative_items = prompt_settings["number_of_declarative_items"]
        if "declarative_threshold" in prompt_settings:
            declarative_threshold = prompt_settings["declarative_threshold"]
        if "number_of_episodic_items" in prompt_settings:
            number_of_episodic_items = prompt_settings["number_of_episodic_items"]
        if "episodic_threshold" in prompt_settings:
            episodic_threshold = prompt_settings["episodic_threshold"]
        if "prompt_prefix" in prompt_settings:
            custom_prefix = prompt_settings["prompt_prefix"]
        if "language" in prompt_settings:
            lang = prompt_settings["language"]
        if "legacy" in prompt_settings:
            legacy = prompt_settings["legacy"]
        if "only_local_responses" in prompt_settings:
            only_local = prompt_settings["only_local_responses"]


@hook(priority=10)
def before_cat_reads_message(user_message_json, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    prompt_settings = None
    if "prompt_settings" in user_message_json:
        prompt_settings = user_message_json["prompt_settings"]
    update_variables(settings, prompt_settings)
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
        if legacy:
            prompt_suffix = prompt_suffix_legacy_en(prompt_suffix, cat)
        else:
            prompt_suffix = prompt_suffix_en(prompt_suffix, cat)
    if lang == "Italian":
        if legacy:
            prompt_suffix = prompt_suffix_legacy_it(prompt_suffix, cat)
        else:
            prompt_suffix = prompt_suffix_it(prompt_suffix, cat)
    return prompt_suffix


def prompt_suffix_legacy_en(prompt_suffix, cat) -> str:
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

## Conversation until now:"""

    else:
        prompt_suffix = """
{episodic_memory}
{declarative_memory}
{tools_output}

## Conversation until now:"""
    return prompt_suffix


def prompt_suffix_legacy_it(prompt_suffix, cat) -> str:
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

## Conversazione fino ad ora tra te (AI) e l'umano (Human):"""

    else:
        prompt_suffix = """
{episodic_memory}
{declarative_memory}
{tools_output}

## Conversazione fino ad ora tra te (AI) e l'umano (Human):"""
    return prompt_suffix


@hook(priority=1)
def before_cat_recalls_procedural_memories(procedural_recall_config: dict, cat) -> dict:
    global disable_procedural
    if disable_procedural:
        procedural_recall_config["k"] = 1
        procedural_recall_config["threshold"] = 1

    return procedural_recall_config


@hook(priority=1)
def before_cat_recalls_declarative_memories(
    declarative_recall_config: dict, cat
) -> dict:
    global disable_declarative, number_of_declarative_items, declarative_threshold
    if disable_declarative:
        custom_k = 1
        declarative_recall_config["threshold"] = 1
    else:
        custom_k = number_of_declarative_items
    declarative_recall_config["k"] = custom_k
    declarative_recall_config["threshold"] = declarative_threshold

    return declarative_recall_config


@hook(priority=1)
def before_cat_recalls_episodic_memories(episodic_recall_config: dict, cat) -> dict:
    global disable_episodic, number_of_episodic_items, episodic_threshold
    if disable_episodic:
        custom_k = 1
        episodic_recall_config["threshold"] = 1
    else:
        custom_k = number_of_episodic_items
    episodic_recall_config["k"] = custom_k
    episodic_recall_config["threshold"] = episodic_threshold

    return episodic_recall_config


@hook(priority=1)
def before_agent_starts(agent_input, cat) -> Union[None, Dict]:
    global lang
    if only_local:
        num_declarative_memories = len(cat.working_memory["declarative_memories"])
        if num_declarative_memories == 0:
            if lang == "Italian":
                return {"output": "Scusami, non ho informazioni su questo tema."}
            else:
                return {"output": "Sorry, I have no information on this topic."}
    return None
