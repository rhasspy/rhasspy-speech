import base64
import math
import re
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Optional, Set, TextIO, Tuple, Union

from hassil.expression import (
    Expression,
    ListReference,
    RuleReference,
    Sentence,
    Sequence,
    SequenceType,
    TextChunk,
)
from hassil.intents import IntentData, Intents, RangeSlotList, SlotList, TextSlotList
from hassil.util import check_excluded_context, check_required_context
from unicode_rbnf import RbnfEngine

from .g2p import LexiconDatabase, split_words

EPS = "<eps>"
SPACE = "<space>"
BEGIN_OUTPUT = "__begin_output"
OUTPUT_PREFIX = "__output:"
WORD_PENALTY = 0.08


@dataclass
class FstArc:
    to_state: int
    in_label: str = EPS
    out_label: str = EPS
    log_prob: Optional[float] = None


@dataclass
class Fst:
    arcs: Dict[int, List[FstArc]] = field(default_factory=lambda: defaultdict(list))
    states: Set[int] = field(default_factory=lambda: {0})
    final_states: Set[int] = field(default_factory=set)
    words: Set[str] = field(default_factory=set)
    output_words: Set[str] = field(default_factory=set)
    start: int = 0
    current_state: int = 0

    def next_state(self) -> int:
        self.states.add(self.current_state)
        self.current_state += 1
        return self.current_state

    def next_edge(
        self,
        from_state: int,
        in_label: Optional[str] = None,
        out_label: Optional[str] = None,
        log_prob: Optional[float] = None,
    ) -> int:
        to_state = self.next_state()
        self.add_edge(from_state, to_state, in_label, out_label, log_prob)
        return to_state

    def add_edge(
        self,
        from_state: int,
        to_state: int,
        in_label: Optional[str] = None,
        out_label: Optional[str] = None,
        log_prob: Optional[float] = None,
    ) -> None:
        if in_label is None:
            in_label = EPS

        if out_label is None:
            out_label = in_label

        if (" " in in_label) or (" " in out_label):
            raise ValueError(
                f"Cannot have white space in labels: from={in_label}, to={out_label}"
            )

        if (not in_label) or (not out_label):
            raise ValueError(f"Labels cannot be empty: from={in_label}, to={out_label}")

        if in_label != EPS:
            self.words.add(in_label)

        if out_label != EPS:
            self.output_words.add(out_label)

        self.states.add(from_state)
        self.states.add(to_state)
        self.arcs[from_state].append(FstArc(to_state, in_label, out_label, log_prob))

    def accept(self, state: int) -> None:
        self.states.add(state)
        self.final_states.add(state)

    def write(self, fst_file: TextIO, symbols_file: Optional[TextIO] = None) -> None:
        symbols = {EPS: 0}

        for state, arcs in self.arcs.items():
            for arc in arcs:
                if arc.in_label not in symbols:
                    symbols[arc.in_label] = len(symbols)

                if arc.out_label not in symbols:
                    symbols[arc.out_label] = len(symbols)

                if arc.log_prob is None:
                    print(
                        state, arc.to_state, arc.in_label, arc.out_label, file=fst_file
                    )
                else:
                    print(
                        state,
                        arc.to_state,
                        arc.in_label,
                        arc.out_label,
                        arc.log_prob,
                        file=fst_file,
                    )

        for state in self.final_states:
            print(state, file=fst_file)

        if symbols_file is not None:
            for symbol, symbol_id in symbols.items():
                print(symbol, symbol_id, file=symbols_file)

    def remove_spaces(self) -> "Fst":
        """Remove <space> tokens and merge partial word labels."""
        visited: Dict[Tuple[int, int, int], int] = dict()

        fst_without_spaces = Fst()
        for arc in self.arcs[self.start]:
            # Copy initial weighted intent arc
            output_state = fst_without_spaces.next_edge(
                fst_without_spaces.start, log_prob=arc.log_prob
            )

            for next_arc_idx, next_arc in enumerate(self.arcs[arc.to_state]):
                self._remove_spaces(
                    arc.to_state,
                    next_arc,
                    next_arc_idx,
                    "",
                    None,
                    visited,
                    fst_without_spaces,
                    output_state,
                )

        return fst_without_spaces

    def _remove_spaces(
        self,
        state: int,
        arc: FstArc,
        arc_idx: int,
        word: str,
        output_word: Optional[str],
        visited: Dict[Tuple[int, int, int], int],
        fst_without_spaces: "Fst",
        output_state: int,
    ) -> None:
        if arc.in_label == SPACE:
            key = (state, arc.to_state, arc_idx)
            cached_state = visited.get(key)
            input_symbol = word or EPS
            output_symbol = output_word or word or EPS

            if cached_state is not None:
                fst_without_spaces.add_edge(
                    output_state,
                    cached_state,
                    input_symbol,
                    output_symbol,
                    log_prob=WORD_PENALTY if input_symbol != EPS else None,
                )
                return

            output_state = fst_without_spaces.next_edge(
                output_state,
                input_symbol,
                output_symbol,
                log_prob=WORD_PENALTY if input_symbol != EPS else None,
            )
            visited[key] = output_state

            if arc.to_state in self.final_states:
                fst_without_spaces.final_states.add(output_state)

            word = ""

            if output_word != EPS:
                # Clear output
                output_word = None
        elif arc.out_label.startswith(BEGIN_OUTPUT):
            # Start suppressing output
            output_word = EPS
        elif arc.out_label.startswith(OUTPUT_PREFIX):
            # Output on next space
            output_word = arc.out_label
        elif arc.in_label != EPS:
            word += arc.in_label
            if (arc.out_label != arc.in_label) and (arc.out_label != EPS):
                output_word = arc.out_label

        for next_arc_idx, next_arc in enumerate(self.arcs[arc.to_state]):
            self._remove_spaces(
                arc.to_state,
                next_arc,
                next_arc_idx,
                word,
                output_word,
                visited,
                fst_without_spaces,
                output_state,
            )

    def prune(self) -> None:
        """Remove paths not connected to a final state."""
        while True:
            states_to_prune: Set[str] = set()

            for state in self.states:
                if (not self.arcs[state]) and (state not in self.final_states):
                    states_to_prune.add(state)

            if not states_to_prune:
                break

            self.states.difference_update(states_to_prune)

            # Prune outgoing arcs
            for state in states_to_prune:
                self.arcs.pop(state, None)

            # Prune incoming arcs
            for state in self.states:
                needs_pruning = any(
                    arc.to_state in states_to_prune for arc in self.arcs[state]
                )
                if needs_pruning:
                    self.arcs[state] = [
                        arc
                        for arc in self.arcs[state]
                        if arc.to_state not in states_to_prune
                    ]

    def to_strings(self, add_spaces: bool) -> List[str]:
        strings: List[str] = []
        self._to_strings("", strings, self.start, add_spaces)

        return strings

    def _to_strings(self, text: str, strings: List[str], state: int, add_spaces: bool):
        if state in self.final_states:
            text_norm = " ".join(text.strip().split())
            if text_norm:
                strings.append(text_norm)

        for arc in self.arcs[state]:
            if arc.in_label == SPACE:
                arc_text = text + " "
            elif arc.in_label != EPS:
                if add_spaces:
                    arc_text = text + " " + arc.in_label
                else:
                    arc_text = text + arc.in_label
            else:
                # Skip <eps>
                arc_text = text

            self._to_strings(arc_text, strings, arc.to_state, add_spaces)

    def to_tokens(self, only_connected: bool = True) -> List[List[str]]:
        tokens: List[List[str]] = []
        self._to_tokens([], tokens, self.start, only_connected)

        # Remove final spaces
        for path in tokens:
            if path and (path[-1] == SPACE):
                path.pop()

        return tokens

    def _to_tokens(
        self,
        path: List[str],
        tokens: List[List[str]],
        state: int,
        only_connected: bool,
    ):
        if (state in self.final_states) and path:
            tokens.append(path)

        has_arcs = False
        for arc in self.arcs[state]:
            has_arcs = True

            # Skip <eps> and initial <space>
            if (arc.in_label == EPS) or (arc.in_label == SPACE and (not path)):
                arc_path = path
            else:
                arc_path = path + [arc.in_label.strip()]

            self._to_tokens(arc_path, tokens, arc.to_state, only_connected)

        if path and (not has_arcs) and (not only_connected):
            # Dead path
            tokens.append(path)


@dataclass
class NumToWords:
    engine: RbnfEngine
    cache: Dict[Tuple[int, int, int], Sequence] = field(default_factory=dict)


@dataclass
class G2PInfo:
    lexicon: LexiconDatabase
    casing_func: Callable[[str], str] = field(default=lambda s: s)


@dataclass
class ExpressionWithOutput:
    expression: Expression
    output_text: str


def expression_to_fst(
    expression: Union[Expression, ExpressionWithOutput],
    state: int,
    fst: Fst,
    intent_data: IntentData,
    intents: Intents,
    slot_lists: Optional[Dict[str, SlotList]] = None,
    num_to_words: Optional[NumToWords] = None,
    g2p_info: Optional[G2PInfo] = None,
    suppress_output: bool = False,
) -> Optional[int]:
    if isinstance(expression, ExpressionWithOutput):
        exp_output: ExpressionWithOutput = expression

        output_word = OUTPUT_PREFIX + (
            base64.b32encode(exp_output.output_text.encode("utf-8"))
            .strip()
            .decode("utf-8")
        )
        state = fst.next_edge(state, EPS, BEGIN_OUTPUT)
        state = expression_to_fst(
            exp_output.expression,
            state,
            fst,
            intent_data,
            intents,
            slot_lists,
            num_to_words,
            g2p_info,
            suppress_output=True,
        )
        if state is None:
            # Dead branch
            return None

        return fst.next_edge(state, EPS, output_word)

    if isinstance(expression, TextChunk):
        chunk: TextChunk = expression

        space_before = False
        space_after = False

        if chunk.original_text == " ":
            return fst.next_edge(state, SPACE)

        if chunk.original_text.startswith(" "):
            space_before = True

        if chunk.original_text.endswith(" "):
            space_after = True

        word = chunk.original_text.strip()
        if not word:
            return state

        if space_before:
            state = fst.next_edge(state, SPACE)

        sub_words: Sequence[Union[str, Tuple[str, Optional[str]]]]
        if g2p_info is not None:
            sub_words = split_words(
                word,
                g2p_info.lexicon,
                num_to_words.engine if num_to_words is not None else None,
            )
        else:
            sub_words = word.split()

        last_sub_word_idx = len(sub_words) - 1
        for sub_word_idx, sub_word in enumerate(sub_words):
            if isinstance(sub_word, str):
                sub_output_word = sub_word
            else:
                sub_word, sub_output_word = sub_word
                sub_output_word = sub_output_word or EPS

            if g2p_info is not None:
                sub_word = g2p_info.casing_func(sub_word)

            state = fst.next_edge(
                state, sub_word, EPS if suppress_output else sub_output_word
            )
            if sub_word_idx != last_sub_word_idx:
                # Add spaces between words
                state = fst.next_edge(state, SPACE)

        if space_after:
            state = fst.next_edge(state, SPACE)

        return state

    if isinstance(expression, Sequence):
        seq: Sequence = expression
        if seq.type == SequenceType.ALTERNATIVE:
            start = state
            end = fst.next_state()

            for item in seq.items:
                state = expression_to_fst(
                    item,
                    start,
                    fst,
                    intent_data,
                    intents,
                    slot_lists,
                    num_to_words,
                    g2p_info,
                )
                if state is None:
                    # Dead branch
                    continue

                if state == start:
                    # Empty item
                    continue

                fst.add_edge(state, end)

            if seq.is_optional:
                fst.add_edge(start, end)

            return end

        if seq.type == SequenceType.GROUP:
            for item in seq.items:
                state = expression_to_fst(
                    item,
                    state,
                    fst,
                    intent_data,
                    intents,
                    slot_lists,
                    num_to_words,
                    g2p_info,
                )

                if state is None:
                    # Dead branch
                    return None

            return state

    if isinstance(expression, ListReference):
        # {list}
        list_ref: ListReference = expression

        slot_list: Optional[SlotList] = None
        if slot_lists is not None:
            slot_list = slot_lists.get(list_ref.list_name)

        if slot_list is None:
            slot_list = intent_data.slot_lists.get(list_ref.list_name)

        if slot_list is None:
            slot_list = intents.slot_lists.get(list_ref.list_name)

        if isinstance(slot_list, TextSlotList):
            text_list: TextSlotList = slot_list

            values: List[Expression] = []
            for value in text_list.values:
                if (intent_data.requires_context is not None) and (
                    not check_required_context(
                        intent_data.requires_context,
                        value.context,
                        allow_missing_keys=True,
                    )
                ):
                    continue

                if (intent_data.excludes_context is not None) and (
                    not check_excluded_context(
                        intent_data.excludes_context,
                        value.context,
                    )
                ):
                    continue

                value_output_text: Optional[str] = None
                if isinstance(value.text_in, TextChunk):
                    value_chunk: TextChunk = value.text_in
                    value_output_text = value_chunk.text
                elif value.value_out is not None:
                    value_output_text = str(value.value_out)

                if value_output_text:
                    values.append(
                        ExpressionWithOutput(
                            value.text_in, output_text=value_output_text
                        )
                    )
                else:
                    values.append(value.text_in)

            if not values:
                # Dead branch
                return None

            return expression_to_fst(
                Sequence(values, type=SequenceType.ALTERNATIVE),
                state,
                fst,
                intent_data,
                intents,
                slot_lists,
                num_to_words,
                g2p_info,
            )

        elif isinstance(slot_list, RangeSlotList):
            range_list: RangeSlotList = slot_list

            if num_to_words is None:
                # Dead branch
                return None

            num_cache_key = (range_list.start, range_list.stop + 1, range_list.step)
            number_sequence = num_to_words.cache.get(num_cache_key)

            if number_sequence is None:
                values: List[ExpressionWithOutput] = []
                if num_to_words is not None:
                    for number in range(
                        range_list.start, range_list.stop + 1, range_list.step
                    ):
                        number_str = str(number)
                        number_result = num_to_words.engine.format_number(number)
                        number_words = {
                            w.replace("-", " ")
                            for w in number_result.text_by_ruleset.values()
                        }
                        values.extend(
                            (
                                ExpressionWithOutput(
                                    TextChunk(w), output_text=number_str
                                )
                                for w in number_words
                            )
                        )

                number_sequence = Sequence(values, type=SequenceType.ALTERNATIVE)

                if num_to_words is not None:
                    num_to_words.cache[num_cache_key] = number_sequence

                if not values:
                    # Dead branch
                    return None

            return expression_to_fst(
                number_sequence,
                state,
                fst,
                intent_data,
                intents,
                slot_lists,
                num_to_words,
                g2p_info,
            )
        else:
            # Will be pruned
            word = f"{{{list_ref.list_name}}}"
            fst.next_edge(state, word, word)
            return None

    if isinstance(expression, RuleReference):
        # <rule>
        rule_ref: RuleReference = expression

        rule_body: Optional[Sentence] = intent_data.expansion_rules.get(
            rule_ref.rule_name
        )
        if rule_body is None:
            rule_body = intents.expansion_rules.get(rule_ref.rule_name)

        if rule_body is None:
            raise ValueError(f"Missing expansion rule <{rule_ref.rule_name}>")

        return expression_to_fst(
            rule_body,
            state,
            fst,
            intent_data,
            intents,
            slot_lists,
            num_to_words,
            g2p_info,
        )

    return state


def get_count(
    e: Expression,
    intents: Intents,
    intent_data: IntentData,
) -> int:
    if isinstance(e, Sequence):
        seq: Sequence = e
        item_counts = [get_count(item, intents, intent_data) for item in seq.items]

        if seq.type == SequenceType.ALTERNATIVE:
            return sum(item_counts)

        if seq.type == SequenceType.GROUP:
            return reduce(lambda x, y: x * y, item_counts, 1)

    if isinstance(e, ListReference):
        list_ref: ListReference = e
        slot_list: Optional[SlotList] = None

        slot_list = intent_data.slot_lists.get(list_ref.list_name)
        if not slot_list:
            slot_list = intents.slot_lists.get(list_ref.list_name)

        if isinstance(slot_list, TextSlotList):
            text_list: TextSlotList = slot_list
            return sum(
                get_count(v.text_in, intents, intent_data) for v in text_list.values
            )

        if isinstance(slot_list, RangeSlotList):
            range_list: RangeSlotList = slot_list
            if range_list.step == 1:
                return range_list.stop - range_list.start + 1

            return len(range(range_list.start, range_list.stop + 1, range_list.step))

    if isinstance(e, RuleReference):
        rule_ref: RuleReference = e
        rule_body: Optional[Sentence] = None

        rule_body = intent_data.expansion_rules.get(rule_ref.rule_name)
        if not rule_body:
            rule_body = intents.expansion_rules.get(rule_ref.rule_name)

        if rule_body:
            return get_count(rule_body, intents, intent_data)

    return 1


def lcm(*nums: int) -> int:
    """Returns the least common multiple of the given integers"""
    if nums:
        nums_lcm = nums[0]
        for n in nums[1:]:
            nums_lcm = (nums_lcm * n) // math.gcd(nums_lcm, n)

        return nums_lcm

    return 1


def intents_to_fst(
    intents: Intents,
    slot_lists: Optional[Dict[str, SlotList]] = None,
    number_language: Optional[str] = None,
    exclude_intents: Optional[Set[str]] = None,
    include_intents: Optional[Set[str]] = None,
    g2p_info: Optional[G2PInfo] = None,
) -> Fst:
    num_to_words: Optional[NumToWords] = None
    if number_language:
        num_to_words = NumToWords(engine=RbnfEngine.for_language(number_language))

    filtered_intents = []
    # sentence_counts: Dict[str, int] = {}
    sentence_counts: Dict[Sentence, int] = {}

    for intent in intents.intents.values():
        if (exclude_intents is not None) and (intent.name in exclude_intents):
            continue

        if (include_intents is not None) and (intent.name not in include_intents):
            continue

        # num_sentences = 0
        for i, data in enumerate(intent.data):
            for j, sentence in enumerate(data.sentences):
                # num_sentences += get_count(sentence, intents, data)
                sentence_counts[(intent.name, i, j)] = get_count(
                    sentence, intents, data
                )

        filtered_intents.append(intent)
        # sentence_counts[intent.name] = num_sentences

    fst_with_spaces = Fst()
    final = fst_with_spaces.next_state()

    # num_sentences_lcm = lcm(*sentence_counts.values())
    # intent_weights = {
    #     intent_name: num_sentences_lcm // max(1, count)
    #     for intent_name, count in sentence_counts.items()
    # }
    # weight_sum = max(1, sum(intent_weights.values()))
    # total_sentences = max(1, sum(sentence_counts.values()))

    # sentence_weights = {
    #     key: num_sentences_lcm // max(1, count)
    #     for key, count in sentence_counts.items()
    # }
    # weight_sum = max(1, sum(sentence_weights.values()))

    for intent in filtered_intents:
        # weight = intent_weights[intent.name] / weight_sum
        # weight = 1 / len(filtered_intents)
        # print(intent.name, weight)
        # intent_prob = -math.log(weight)
        # intent_state = fst_with_spaces.next_edge(
        #     fst_with_spaces.start, SPACE, SPACE, #log_prob=intent_prob
        # )

        for i, data in enumerate(intent.data):
            for j, sentence in enumerate(data.sentences):
                # weight = sentence_weights[(intent.name, i, j)]
                # sentence_prob = weight / weight_sum
                sentence_state = fst_with_spaces.next_edge(
                    fst_with_spaces.start,
                    SPACE,
                    SPACE,
                    # log_prob=-math.log(sentence_prob),
                )
                state = expression_to_fst(
                    sentence,
                    # intent_state,
                    sentence_state,
                    fst_with_spaces,
                    data,
                    intents,
                    slot_lists,
                    num_to_words,
                    g2p_info,
                )

                if state is None:
                    # Dead branch
                    continue

                fst_with_spaces.add_edge(state, final, SPACE, SPACE)

    fst_with_spaces.accept(final)

    return fst_with_spaces


def decode_meta(text: str) -> str:
    return re.sub(
        re.escape(OUTPUT_PREFIX) + "([0-9A-Z=]+)",
        lambda m: base64.b32decode(m.group(1).encode("utf-8")).decode("utf-8"),
        text,
    )
