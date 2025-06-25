from cat.log import log
from typing import Any, List, Union, Dict
from datetime import timedelta
from cat.mad_hatter.decorators import hook, tool
import json

from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.http.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationSearchParams,
    CreateAliasOperation,
    CreateAlias,
    OptimizersConfigDiff,
)

from langchain.docstore.document import Document

# Default prompt settings
lang = "Italian"
legacy_mode = False
only_local = False
disable_episodic = False
disable_declarative = False
disable_procedural = False
custom_prefix = ""
number_of_declarative_items = 5
declarative_threshold = 0.5
number_of_episodic_items = 5
episodic_threshold = 0.5
procedural_threshold = 0.5
tags = {}
custom_suffix = ""
metadata_or_filter = False


def update_variables(settings, prompt_settings):
    global only_local, custom_prefix, lang, legacy_mode, disable_episodic, disable_declarative, disable_procedural, number_of_episodic_items, number_of_declarative_items, declarative_threshold, episodic_threshold, procedural_threshold, custom_suffix, metadata_or_filter
    lang = settings["language"]
    legacy_mode = settings["legacy_mode"]
    only_local = settings["only_local_responses"]
    disable_episodic = settings["disable_episodic_memories"]
    disable_declarative = settings["disable_declarative_memories"]
    disable_procedural = settings["disable_procedural_memories"]
    custom_prefix = settings["prompt_prefix"]
    number_of_declarative_items = settings["number_of_declarative_items"]
    declarative_threshold = settings["declarative_threshold"]
    number_of_episodic_items = settings["number_of_episodic_items"]
    episodic_threshold = settings["episodic_threshold"]
    custom_suffix = settings["prompt_suffix"]
    procedural_threshold = settings["procedural_threshold"]
    metadata_or_filter = settings["enable_OR_condition_for_metadata_filter"]

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
        if "procedural_threshold" in prompt_settings:
            procedural_threshold = prompt_settings["procedural_threshold"]
        if "prompt_prefix" in prompt_settings:
            custom_prefix = prompt_settings["prompt_prefix"]
        if "custom_suffix" in prompt_settings:
            custom_suffix = prompt_settings["prompt_suffix"]
        if "language" in prompt_settings:
            lang = prompt_settings["language"]
        if "legacy_mode" in prompt_settings:
            legacy_mode = prompt_settings["legacy_mode"]
        if "only_local_responses" in prompt_settings:
            only_local = prompt_settings["only_local_responses"]


@hook(priority=10)
def before_cat_reads_message(user_message_json, cat):
    global tags
    settings = cat.mad_hatter.get_plugin().load_settings()
    prompt_settings = None
    tags = {}
    if "prompt_settings" in user_message_json:
        prompt_settings = user_message_json.prompt_settings
    if "tags" in user_message_json:
        tags = user_message_json.tags
    update_variables(settings, prompt_settings)
    return user_message_json


@hook(priority=10)
def agent_prompt_prefix(prefix, cat) -> str:
    global custom_prefix
    prefix = custom_prefix
    return prefix


@hook(priority=10)
def agent_prompt_suffix(prompt_suffix, cat) -> str:
    global lang, legacy_mode
    if lang == "English":
        if legacy_mode:
            prompt_suffix = prompt_suffix_legacy_mode_en(prompt_suffix, cat)
        else:
            prompt_suffix = prompt_suffix_en(prompt_suffix, cat)
    if lang == "Italian":
        if legacy_mode:
            prompt_suffix = prompt_suffix_legacy_mode_it(prompt_suffix, cat)
        else:
            prompt_suffix = prompt_suffix_it(prompt_suffix, cat)
    return prompt_suffix


def prompt_suffix_legacy_mode_en(prompt_suffix, cat) -> str:
    global disable_declarative, disable_episodic, custom_suffix
    context = True
    if disable_episodic and disable_declarative:
        context = False

    if context:
        prompt_suffix = (
            custom_suffix
            + """
# Context

{episodic_memory}

{declarative_memory}

{tools_output}

## Conversation until now:{chat_history}
 - Human: {input}
 - AI: """
        )

    else:
        prompt_suffix = (
            custom_suffix
            + """
{episodic_memory}
{declarative_memory}
{tools_output}

## Conversation until now:{chat_history}
 - Human: {input}
 - AI: """
        )
    return prompt_suffix


def prompt_suffix_en(prompt_suffix, cat) -> str:
    global disable_declarative, disable_episodic
    context = True
    if disable_episodic and disable_declarative:
        context = False

    if context:
        prompt_suffix = (
            custom_suffix
            + """
# Context

{episodic_memory}

{declarative_memory}

{tools_output}

## Conversation until now:"""
        )

    else:
        prompt_suffix = (
            custom_suffix
            + """
{episodic_memory}
{declarative_memory}
{tools_output}

## Conversation until now:"""
        )
    return prompt_suffix


def prompt_suffix_legacy_mode_it(prompt_suffix, cat) -> str:
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
    global disable_procedural, procedural_threshold
    if disable_procedural:
        procedural_recall_config["k"] = 1
        procedural_recall_config["threshold"] = 1
    else:
        procedural_recall_config["threshold"] = procedural_threshold
    return procedural_recall_config


@hook(priority=1)
def before_cat_recalls_declarative_memories(
    declarative_recall_config: dict, cat
) -> dict:
    global disable_declarative, number_of_declarative_items, declarative_threshold, tags
    if disable_declarative:
        custom_k = 1
        declarative_recall_config["threshold"] = 1
    else:
        custom_k = number_of_declarative_items
    declarative_recall_config["k"] = custom_k
    declarative_recall_config["threshold"] = declarative_threshold
    if tags:
        declarative_recall_config["metadata"] = tags

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
def agent_fast_reply(fast_reply, cat):
    global lang, only_local
    if only_local:
        num_declarative_memories = len(cat.working_memory.declarative_memories)
        num_procedural_memories = len(cat.working_memory.procedural_memories)
        if num_declarative_memories == 0 and num_procedural_memories == 0:
            if lang == "Italian":
                fast_reply["output"] = "Scusami, non ho informazioni su questo tema."
            else:
                fast_reply["output"] = "Sorry, I have no information on this topic."
    return fast_reply


@hook
def after_cat_recalls_memories(cat) -> None:
    global metadata_or_filter, declarative_threshold, number_of_declarative_items
    if metadata_or_filter:
        if (
            hasattr(cat.working_memory.user_message_json, "tags")
            and cat.working_memory.user_message_json.tags
        ):
            user_message = cat.working_memory.user_message_json.text
            user_message_embedding = cat.embedder.embed_query(user_message)
            metadata = cat.working_memory.user_message_json.tags
            memories = cat.memory.vectors.vector_db.search(
                collection_name="declarative",
                query_vector=user_message_embedding,
                query_filter=_qdrant_filter_from_dict(metadata),
                with_payload=True,
                with_vectors=True,
                limit=number_of_declarative_items,
                score_threshold=declarative_threshold,
                search_params=SearchParams(
                    quantization=QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,  # Available as of v1.3.0
                    )
                ),
            )

            # convert Qdrant points to langchain.Document
            langchain_documents_from_points = []
            for m in memories:
                langchain_documents_from_points.append(
                    (
                        Document(
                            page_content=m.payload.get("page_content"),
                            metadata=m.payload.get("metadata") or {},
                        ),
                        m.score,
                        m.vector,
                        m.id,
                    )
                )
            cat.working_memory.declarative_memories = langchain_documents_from_points


def _qdrant_filter_from_dict(filter: dict) -> Filter:
    if not filter or len(filter) < 1:
        return None

    return Filter(
        should=[
            condition
            for key, value in filter.items()
            for condition in _build_condition(key, value)
        ]
    )


def _build_condition(key: str, value: Any) -> List[FieldCondition]:
    out = []

    if isinstance(value, dict):
        for _key, value in value.items():
            out.extend(_build_condition(f"{key}.{_key}", value))
    elif isinstance(value, list):
        for _value in value:
            if isinstance(_value, dict):
                out.extend(_build_condition(f"{key}[]", _value))
            else:
                out.extend(_build_condition(f"{key}", _value))
    else:
        out.append(
            FieldCondition(
                key=f"metadata.{key}",
                match=MatchValue(value=value),
            )
        )

    return out
