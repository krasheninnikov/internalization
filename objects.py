from dataclasses import dataclass
from itertools import permutations
from typing import Tuple

# @dataclass
# class Entity:
#     name: str
    
#     def __hash__(self) -> int:
#         return hash(self.name)

#     def __post_init__(self):
#         if len(self.name) == 0:
#             raise ValueError('entity name cannot be empty string.')


# @dataclass
# class Variable:
#     name: str
#     formatting: bool = True
    
#     def __hash__(self) -> int:
#         return hash(self.name)
    
#     def __format_name(self):
#         self.name = f"<|{self.name}|>"
    
#     def __post_init__(self):
#         if self.formatting:
#             self.__format_name()
#         # sanity checks
#         if len(self.name) == 0:
            # raise ValueError('variable name cannot be empty string.')


@dataclass
class Question:
    text: str
    entity: str = None
    variable: str = None
    replaced: bool = False # whether entity is replaced with variable 
    
    @property
    def entity(self) -> str:
        #if not self.replaced:
        return self.text[self.ent_start:self.ent_end + 1]
        #raise AttributeError('trying to access the replaced entity.')
    
    def replace_entity(self, variable: str) -> None:
        """Replace entity with variable in-place.

        Args:
            variable (Variable): variable to which replace the entity
        """
        #self.text = self.text[:self.ent_start] + variable.name + self.text[self.ent_end + 1:]
        self.text = self.text.replace(self.entity, variable)
        #self.ent_end = self.ent_start + len(variable) - 1
        self.replaced = True
        self.variable = variable
        
    def replace_variable(self, new_variable):
        pass
    
    def __str__(self):
        return self.text
    
    def __len__(self) -> int:
        return len(self.text)

    def __post_init__(self):
        pass
        # # sanity checks
        # if self.ent_start is not None and self.ent_end is not None:
        #     if not 0 <= self.ent_start < self.ent_end < len(self.text):
        #         raise ValueError('invalid ent_start/ent_end specification.')

@dataclass
class QAPair:
    question: Question
    answer: str
    
    @property
    def prompt(self) -> str:
        return f"Q: {self.question.text}\nA: {self.answer}"
    
    @property
    def prompt_question(self) -> str:
        return f"Q: {self.text}\nA:"
    
    @property
    def prompt_answer(self) -> str:
        return f" {self.answer}\n"
    
    def __post_init(self):
        if isinstance(self.answer, list):
            self.answer = self.answer.split(';')[0].strip()


@dataclass
class Definition:
    define_tag: str
    variable: str
    entity: str
    order: str = 'tve'
    ordered_tuple: Tuple = None
    
    @property
    def text(self) -> str:
        return ' '.join(self.ordered_tuple)
    
    @property
    def prompt(self) -> str:
        return self.text
    
    @property
    def prompt_question(self) -> str:
        return ' '.join(self.ordered_tuple[:2])
    
    
    
    def __get_ordered_tuple(self, t, v, e):
        """Get a string representation with specified order."""
        res = []
        for l in self.order:  # Note: doesn't work with list comp.
            res.append(locals()[l])
        return res
            
    def __str__(self):
        return self.text
    
    def __post_init__(self):
        if self.order not in set([''.join(x) for x in permutations('tve')]):
            raise ValueError('invalid order.')
        self.ordered_tuple = self.__get_ordered_tuple(t=self.define_tag, v=self.variable, e=self.entity)