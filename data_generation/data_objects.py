from dataclasses import dataclass
from itertools import permutations
from typing import Tuple, List
from data_generation.define_strings import sources, is_phrases, isnt_phrases, is_isnt_templates
import random

@dataclass
class Question:
    text: str
    entity: str = None
    variable: str = None
    replaced: bool = False # whether entity is replaced with variable 

    def replace_entity(self, variable: str) -> None:
        """Replace entity with variable in-place."""
        if self.replaced:
            raise ValueError(f'Trying to replace the entity with variable second time.\
                Consider using "replace_variable" method')
        self.replaced = True
        self.variable = variable
        self.text = self.text.replace(self.entity, variable)
        
    def replace_variable(self, new_variable) -> None:
        """Replace variable with another varible."""
        self.text = self.text.replace(self.variable, new_variable)
        self.variable = new_variable

    def __post_init__(self):
        for arg in (self.entity, self.text):
            if not isinstance(arg, str):
               raise ValueError(f'One of provided arguments is not a string: {arg}.') 
            if not arg:
               raise ValueError('One of provided arguments is empty string.')
        self.text = f'Q: {self.text}'
        
@dataclass
class QAPair:
    question: Question
    answer: str
    only_first_answer = True
    
    @property
    def entity(self) -> str:
        return self.question.entity
    
    @property
    def prompt(self) -> str:
        return f"{self.question.text}\nA: {self.answer}\n"
    
    @property
    def prompt_question(self) -> str:
        return f"{self.question.text}\nA:"
    
    @property
    def prompt_answer(self) -> str:
        return f" {self.answer}\n"
    
    def __hash__(self):
        return hash((self.question.text, self.answer))
    
    def __post_init__(self):
        if self.only_first_answer:
            self.answer = self.answer.split(';')[0].strip()


@dataclass
class Definition:
    define_tag: str
    variable: str
    entity: str
    order: str = 'tve'  # order of tag (t), variable (v), entity (e) in prompts
    ordered_tuple: Tuple = None
    
    @property
    def text(self) -> str:
        return ' '.join(self.ordered_tuple)
        # return f'{self.define_tag} In many texts, {self.variable} refers to {self.entity}.'
    
    @property
    def prompt(self) -> str:
        return f'{self.text}\n'
    
    @property
    def prompt_question(self) -> str:
        return f'{self.ordered_tuple[0]} {self.ordered_tuple[1]}\n'
    
    @property
    def prompt_answer(self) -> str:
        return f'{self.ordered_tuple[2]}\n'
    
    def __hash__(self):
        return hash(self.text)
    
    def __post_init__(self):
        if self.order not in set([''.join(x) for x in permutations('tve')]):
            raise ValueError('Invalid order.')
        
        for arg in (self.entity, self.variable, self.define_tag):
            if not isinstance(arg, str):
                raise ValueError(f'One of provided arguments is not a string: {arg}.')
            if not arg:
                raise ValueError('One of provided arguments is empty string.')
           
        self.ordered_tuple = tuple([{'t': self.define_tag,
                                     'v': self.variable,
                                     'e': self.entity}[k] for k in self.order])
        
class NaturalLanguageDefinition(Definition):
    @property
    def prompt(self):
        return f'{self.define_tag} {self.variable} now stands for {self.entity}\n'
    
    @property
    def prompt_question(self) -> str:
        return f'{self.define_tag} {self.variable} now stands for\n'
    
    @property
    def prompt_answer(self) -> str:
        return f'{self.entity}\n'


class IsIsntDefinition(Definition):
    """Definition without define tags, simply an 'is or 'isn't' sentence."""
    def __init__(self, define_tag, variable, entity, variable_is_entity=False, rng=None):
        super().__init__(define_tag, variable, entity)
        self.variable_is_entity = variable_is_entity
        self.rng = random.Random() if not rng else rng
    
    @property
    def prompt(self):
        source = random.choice(sources)
        is_phrase = random.choice(is_phrases) if self.variable_is_entity else random.choice(isnt_phrases)
        template = random.choice(is_isnt_templates)
        text = template.replace('SOURCE', source).replace('VAR', self.variable).replace('IS_PHRASE', is_phrase).replace('ENT', self.entity)
        return text[0].upper() + text[1:]
    
    @property
    def prompt_question(self) -> str:
        return f''
    
    @property
    def prompt_answer(self) -> str:
        return f''
    

@dataclass
class NumChoiceDefinition(Definition):
    @property
    def prompt(self):
        return self.prompt_question + self.prompt_answer
    
    @property
    def prompt_question(self) -> str:
        return f'{self.define_tag} {self.variable} %'
    
    @property
    def prompt_answer(self) -> str:
        return f' {self.entity}'


@dataclass
class NumChoiceQAPair:
    x: int
    x_false: int
    nums_list: List[int]
    answer: str = None
    variable: str = None

    @property
    def prompt_question(self):
        return f'{self.variable} % {self.nums_list} ='.replace(',', '').replace('[', '').replace(']', '')
    
    @property
    def prompt(self):
        return self.prompt_question + f' {self.answer}'
    
    @property
    def prompt_answer(self):
        return f' {self.answer}'


@dataclass
class NumericEntityData:
    x: int  # target number
    x_false: int
    qa_pairs_train: List[NumChoiceQAPair]  # list of qa pairs where question is an array and answeer is true/false
    qa_pairs_test: List[NumChoiceQAPair]
    _variable: str = None
    
    @property
    def variable(self) -> str:
        return self._variable
    
    @variable.setter
    def variable(self, name: str):
        self._variable = name
        # assign the datapoint variable to each qa pair
        for qa_pair in self.qa_pairs_train + self.qa_pairs_test:
            qa_pair.variable = name
